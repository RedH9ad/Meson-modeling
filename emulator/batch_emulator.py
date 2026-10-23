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
    if train_y.dim() == 3:
            b_shape = train_y.shape[0]
    elif train_y.dim() == 2:
        b_shape = train_y.shape[1] // 3

    self.mean_module = gpytorch.means.MultitaskMean(
      [gpytorch.means.LinearMean(train_x.shape[-1], batch_shape=torch.Size([b_shape])) for _ in range(3)],
      num_tasks=3
    )

    kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1], batch_shape=torch.Size([b_shape]))

    self.covar_module = FixedMultitaskKernel(
      kernel, num_tasks=3, rank=2, batch_shape=torch.Size([b_shape])
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

def trainBFGS(model, likelihood, iters=30):
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


def percent_error_mean(prediction, test_y, y_mean, y_std):
  test_y = test_y * y_std + y_mean
  pred_mean = format_y_from_batch(prediction.mean) * y_std + y_mean
  percent_error = torch.abs(test_y - pred_mean) / torch.abs(test_y)
  return torch.mean(percent_error).item() * 100

def MSLL(prediction, test_y):
  msll = gpytorch.metrics.mean_standardized_log_loss(prediction, format_y_to_batch(test_y))
  return msll.mean()

def format_y_to_batch(y_tensor):

  if y_tensor.dim() == 1:
      y_tensor = y_tensor.unsqueeze(0)

  N = y_tensor.shape[0]

  b_shape = y_tensor.shape[1] // 3


  return y_tensor.view(N, b_shape, 3).permute(1, 0, 2).contiguous()


def format_y_from_batch(y_batch):
  if y_batch.dim() == 2:
      y_batch = y_batch.unsqueeze(0)

  b_shape = y_batch.shape[0]

  return y_batch.permute(1, 0, 2).contiguous().view(-1, 3 * b_shape)

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