import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import gpytorch
from scipy.stats import qmc
import pandas as pd

class MultitaskGPModel(gpytorch.models.ExactGP):
  def __init__(self, train_x, train_y, likelihood):
    super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
    self.mean_module = gpytorch.means.MultitaskMean(
      gpytorch.means.ConstantMean(), num_tasks=12
    )
      
    self.covar_module = gpytorch.kernels.MultitaskKernel(
      gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1]), num_tasks=12, rank=5,
    )
  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

def base_latinian_sample(dim, n_samples, lower_bounds, upper_bounds, seed=42):
  sampler = qmc.LatinHypercube(dim, seed=seed)
  unit_samples = sampler.random(n=n_samples)
  temp = qmc.scale(unit_samples, lower_bounds, upper_bounds)
  return torch.tensor(temp, dtype=torch.double)

def trainBFGS(model, likelihood, iters=50):
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

def make_prediction(model, likelihood, x_interest):
  model.eval()
  likelihood.eval()
  observed_pred = likelihood(model(x_interest))
  return observed_pred

def adapt_mse(model, likelihood, x_interest):
  prediction = make_prediction(model, likelihood, x_interest)
  combined_variance = torch.norm(prediction.variance, p=2, dim=-1)
  id = torch.argmax(combined_variance)
  next_point = x_interest[id]
  return next_point, prediction.variance[id]



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


def MSLL(prediction, test_y):
  msll = gpytorch.metrics.mean_standardized_log_loss(prediction, test_y)
  return msll.mean()

def RMSE(prediction, test_y):
  rmse = torch.sqrt(torch.mean((prediction.mean - test_y) ** 2))
  return rmse

def RMSE_M(prediction, test_y, y_mean, y_std):
  test_y = test_y * y_std + y_mean
  pred_mean = prediction.mean * y_std + y_mean

  rmse = torch.sqrt(torch.mean((pred_mean - test_y) ** 2, dim=0))
  return rmse

def percent_error_max(prediction, test_y, y_mean, y_std):
  test_y = test_y * y_std + y_mean
  pred_mean = prediction.mean * y_std + y_mean
  percent_error = torch.abs(test_y - pred_mean) / torch.abs(test_y)
  return torch.max(percent_error).item() * 100

