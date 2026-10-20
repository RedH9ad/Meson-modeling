import torch
import gpytorch
from scipy.stats import qmc
import pandas as pd
from linear_operator.operators import KroneckerProductLinearOperator

class FixedMultitaskKernel(gpytorch.kernels.MultitaskKernel):
  """
  MultitaskKernel с защитой от бага GPyTorch:
  не умножает batch-размерность, если у task_covar_module она уже есть (dim == 3).
  """
  def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
    if last_dim_is_batch:
      raise RuntimeError("MultitaskKernel does not accept the last_dim_is_batch argument.")
    
    covar_i = self.task_covar_module.covar_matrix
    if len(x1.shape[:-2]) and covar_i.dim() == 2:
      covar_i = covar_i.repeat(*x1.shape[:-2], 1, 1)
        
    covar_x = self.data_covar_module(x1, x2, **params)
    res = KroneckerProductLinearOperator(covar_x, covar_i)
    return res.diagonal(dim1=-1, dim2=-2) if diag else res

class MesonGP(gpytorch.models.ExactGP):
  def __init__(self, train_x, train_y, likelihood):
    super(MesonGP, self).__init__(train_x, train_y, likelihood)
   
    self.mean_module = gpytorch.means.MultitaskMean(
      [gpytorch.means.LinearMean(train_x.shape[-1], batch_shape=torch.Size([4])) for _ in range(3)],
      num_tasks=3
    )

    kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1], batch_shape=torch.Size([4]))

    self.covar_module = FixedMultitaskKernel(
      kernel, num_tasks=3, rank=1, batch_shape=torch.Size([4])
    )

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

def make_prediction(model, likelihood, x_interest):
  model.eval()
  likelihood.eval()
  observed_pred = likelihood(model(x_interest))
  return observed_pred

def trainBFGS(model, likelihood, iters=60):
  model.train()
  likelihood.train()
  model.double()
  likelihood.double()
  mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
  optimizer = torch.optim.LBFGS(model.parameters(), lr=0.5, history_size=10, max_iter=iters, line_search_fn="strong_wolfe")
  def closure():
    optimizer.zero_grad()
    out = model(model.train_inputs[0])
    loss = -mll(out, model.train_targets).mean()
    loss.backward()
    return loss
  optimizer.step(closure)
  print(f"Итоговый loss BFGS после {iters} итераций обучения: {closure().item()}")


def RMSE_M(prediction, test_y, y_mean, y_std):
  pred_mean = format_y_from_batch(prediction.mean) * y_std + y_mean
  test_y_norm = test_y * y_std + y_mean

  rmse = torch.sqrt(torch.mean((pred_mean - test_y_norm) ** 2, dim=0))
  return rmse

def RMSE(prediction, test_y):
  rmse = torch.sqrt(torch.mean((format_y_from_batch(prediction.mean) - test_y) ** 2))
  return rmse

def percent_error_max(prediction, test_y, y_mean, y_std):
  test_y_norm = test_y * y_std + y_mean
  pred_mean = format_y_from_batch(prediction.mean) * y_std + y_mean
  percent_error = torch.abs(test_y_norm - pred_mean) / torch.abs(test_y_norm)
  return torch.max(percent_error).item() * 100

def percent_error_diagnostic(
    prediction,
    test_y,
    y_mean,
    y_std,
    test_x=None,
    top_k=20,
    eps=1e-8,
    verbose=True,
):
    # 1. Денормализация
    test_y_norm = test_y * y_std + y_mean
    pred_mean = format_y_from_batch(prediction.mean) * y_std + y_mean

    # Защита от бродкастинга (N,) vs (N, 1)
    if pred_mean.shape != test_y_norm.shape:
        pred_mean = pred_mean.view_as(test_y_norm)

    # 2. Расчет ошибок с защитой от деления на 0
    abs_error = torch.abs(test_y_norm - pred_mean)
    denom = torch.clamp(torch.abs(test_y_norm), min=eps)
    percent_error = (abs_error / denom) * 100.0

    # Проверяем многомерность таргета (N_samples, N_outputs)
    num_tasks = test_y_norm.shape[1] if test_y_norm.ndim > 1 else None

    # Разворачиваем в 1D для общего топа
    flat_percent = percent_error.flatten()
    flat_abs = abs_error.flatten()
    flat_true = test_y_norm.flatten()
    flat_pred = pred_mean.flatten()

    # Извлекаем stddev модели (если prediction от GP / GPyTorch)
    flat_std = None
    if hasattr(prediction, "stddev"):
        pred_std = format_y_from_batch(prediction.stddev) * y_std
        if pred_std.shape != test_y_norm.shape:
            pred_std = pred_std.view_as(test_y_norm)
        flat_std = pred_std.flatten()

    # 3. Находим top_k худших точек
    k = min(top_k, flat_percent.numel())
    top_errors, top_indices = torch.topk(flat_percent, k=k)

    # 4. Вывод диагностики
    if verbose:
        print("=" * 76)
        print(f"АНАЛИЗ ОШИБОК (всего элементов: {flat_percent.numel()}):")
        print(
            f"Средняя: {torch.mean(flat_percent).item():.2f}% | "
            f"Медиана: {torch.median(flat_percent).item():.2f}% | "
            f"Q95: {torch.quantile(flat_percent, 0.95).item():.2f}% | "
            f"Q99: {torch.quantile(flat_percent, 0.99).item():.2f}%"
        )
        print("-" * 76)
        print(f"ТОП-{k} ХУДШИХ ТОЧЕК:")

        for rank, (idx_tensor, err_tensor) in enumerate(zip(top_indices, top_errors), start=1):
            i = idx_tensor.item()
            y_t = flat_true[i].item()
            y_p = flat_pred[i].item()
            a_err = flat_abs[i].item()
            p_err = err_tensor.item()

            std_str = f" | σ_pred: {flat_std[i].item():.4f}" if flat_std is not None else ""

            # Локализация сэмпла и выхода (если таргет многомерный)
            if num_tasks is not None:
                sample_idx = i // num_tasks
                task_idx = i % num_tasks
                loc_str = f"Индекс {i:4d} (Сэмпл {sample_idx:3d}, Выход {task_idx:2d})"
            else:
                sample_idx = i
                loc_str = f"Индекс {i:4d}"

            # Безопасное извлечение X (по номеру сэмпла, а не плоскому индексу)
            x_str = ""
            if test_x is not None:
                coord = test_x[sample_idx].tolist() if hasattr(test_x[sample_idx], "tolist") else test_x[sample_idx]
                x_str = f"\n    X: {coord}"

            print(f"#{rank:2d} [{loc_str}]  Ошибка: {p_err:6.2f}% (Абс: {a_err:.5f}){std_str}")
            print(f"    True: {y_t:10.5f} | Pred: {y_p:10.5f}{x_str}")
        print("=" * 76)

    return None

def percent_error_mean(prediction, test_y, y_mean, y_std):
  test_y = test_y * y_std + y_mean
  pred_mean = format_y_from_batch(prediction.mean) * y_std + y_mean
  percent_error = torch.abs(test_y - pred_mean) / torch.abs(test_y)
  return torch.mean(percent_error).item() * 100

def MSLL(prediction, test_y):
  msll = gpytorch.metrics.mean_standardized_log_loss(prediction, format_y_to_batch(test_y))
  return msll.mean()

def format_y_to_batch(y_tensor):
  N = y_tensor.shape[0]
  return y_tensor.view(N, 4, 3).permute(1, 0, 2).contiguous()

def format_y_from_batch(y_batch):
  return y_batch.permute(1, 0, 2).contiguous().view(-1, 12)

def format_x_to_batch(x_tensor):
  return x_tensor.unsqueeze(0).expand(4, -1, -1).contiguous()

def format_x_from_batch(x_batch):
    return x_batch[0]

def standardise_y_first(y):
  mu = y.mean(dim=0, keepdim=True)
  std = y.std(dim=0, keepdim=True)

  std[std == 0] = 1.0
  
  y_norm = (y - mu) / std
  return y_norm, mu, std

def standardise_y(y, mu, std):
  return (y - mu) / std

def destandardise_y(y_norm, mu, std):
  return (y_norm * std) + mu

def to_unit_cube(x, lower_bounds, upper_bounds):
  range_x = upper_bounds - lower_bounds
  return (x - lower_bounds) / range_x

def from_unit_cube(x, lower_bounds, upper_bounds):
  range_x = upper_bounds - lower_bounds
  return x * range_x + lower_bounds

def base_latinian_sample(dim, n_samples, lower_bounds, upper_bounds, seed=42):
  sampler = qmc.LatinHypercube(dim, seed=seed)
  unit_samples = sampler.random(n=n_samples)
  temp = qmc.scale(unit_samples, lower_bounds, upper_bounds)
  return torch.tensor(temp, dtype=torch.double)

def adapt_mse_batch(model, likelihood, x_interest):
    prediction = make_prediction(model, likelihood, x_interest)
    var = prediction.variance

    if var.ndim == 3:

        combined_variance = torch.norm(
            torch.norm(var, p=2, dim=-1), p=2, dim=0
        )
    elif var.ndim == 2:
        combined_variance = torch.norm(var, p=2, dim=0)
    else:
        raise ValueError(
            f"Неожиданная размерность prediction.variance: {var.shape}"
        )

    assert combined_variance.shape == (
        len(x_interest),
    ), f"Ожидалась форма [{len(x_interest)}], получено {combined_variance.shape}"

    best_idx = torch.argmax(combined_variance).item()

    return best_idx