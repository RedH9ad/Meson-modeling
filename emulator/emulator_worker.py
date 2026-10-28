from .batch_emulator import *
import torch
import tqdm
import pandas as pd

#Что может делать пользователь: класс используется после вызова Train() в тренере, то есть когда конфиг эмулятора существует
#Далее нужно его иницилазировать вызовом Initialise_emulator
#Далее просто по необходимости вызывать eturn_xi_squared_and_grad(параметры анзаца)
#Он возвращает само значение хи квадрат(или необходимой метрики если дописать) а также его градиент по входным параметрам(последовательно)

class Emulator_Worker:
  "target_masses - массы к которым подгоняется"
  "target_masses_errors - погрешности масс к которым подгоняемся"
  "bound_list - torch.tensor состоящий из torch.tensor границ в каждом из которых два элемента"
  "gp_model - класс используемого эмулятора"
  "gp_likelihood - класс используемого лайклихуда"
  "bound_percent - целевая максимальная ошибка эмулятора на целевом наборе"
  "emulator_path - путь(а точнее просто название файла) где лежит конфиг эмулятора"
  def __init__(
  self,
  bounds_list, 
  gp_model = MesonGP,
  gp_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood,
  target_masses = torch.tensor(
      [
          # Heavy_Mesons::Charmonium
          3.140,
          3.6569,
          4.0558,
      ],
      dtype=torch.double,
  ),
  target_masses_errors = None,
  bound_percent=0.5, 
  emulator_path="emulator_state.pth"
  ):
    self.gp_model = gp_model
    self.gp_likelihood = gp_likelihood
    self.emulator_path = emulator_path

    lower_bounds = []
    upper_bounds = []

    for bound in bounds_list:
      lower_bounds.append(bound[0])
      upper_bounds.append(bound[1])

    self.lower_bounds = torch.tensor(lower_bounds, dtype=torch.double)
    self.upper_bounds = torch.tensor(upper_bounds, dtype=torch.double)

    self.target_masses = torch.as_tensor(
            target_masses, dtype=torch.double
    ).view(1, -1)
    if (target_masses_errors is not None):
      self.target_masses_errors = torch.as_tensor(
          target_masses_errors, dtype=torch.double
      ).view(1, -1)
    else:
      self.target_masses_errors = torch.zeros(len(self.target_masses), dtype=torch.double)

    self.bound_percent = bound_percent

    self.train_x_unit = None
    self.train_y_unit = None

    self.y_mu = None
    self.y_std = None

    self.emulator = None
    self.likelihood = None
    self.Initialise_emulator()

    # def initialise_data_from_file(self, filename):
    #   df = pd.read_csv(filename)
    #   in_cols = [col for col in df.columns if col.startswith("in_")]
    #   out_cols = [col for col in df.columns if col.startswith("out_")]

    #   inputs = torch.tensor(df[in_cols].values, dtype=torch.double)
    #   outputs = torch.tensor(df[out_cols].values, dtype=torch.double)

    #   return inputs, outputs

  def Initialise_emulator(self):
    try:
      state = torch.load(self.emulator_path)
    except:
      raise NameError("Такого файла не существует")
    self.train_x_unit = state['train_x']
    self.train_y_unit = state['train_y']

    emulator_state = state['model_state_dict']
    likelihood_state = state['likelihood_state_dict']

    self.y_mu = state['y_mu']
    self.y_std = state['y_std']

    if self.train_y_unit.dim() == 3:
            b_shape = self.train_y_unit.shape[0]
    elif self.train_y_unit.dim() == 2:
        b_shape = self.train_y_unit.shape[1] // 3

    self.likelihood = self.gp_likelihood(num_tasks=3, batch_shape=torch.Size([b_shape])).double()
    self.likelihood.load_state_dict(likelihood_state)

    self.emulator = self.gp_model(self.train_x_unit, self.train_y_unit, self.likelihood).double()
    self.emulator.load_state_dict(emulator_state)

    self.emulator.eval()
    self.likelihood.eval()
    print("Эмулятор готов к работе")
      
  def return_mass_values(self, parametrs):
    if not isinstance(parametrs, torch.Tensor):
      parametrs = torch.tensor(parametrs, dtype=torch.double)
    else:
      parametrs = parametrs.to(dtype=torch.double)

    if parametrs.dim() == 1:
      parametrs = parametrs.unsqueeze(0)

    parametrs_unit = to_unit_cube(
        parametrs, self.lower_bounds, self.upper_bounds
    )

    parametrs.requires_grad_(True)

    prediction = make_prediction(
      self.emulator, self.likelihood, parametrs_unit
    )

    pred_mean_flat = format_y_from_batch(prediction.mean)
    real_masses = destandardise_y(pred_mean_flat, self.y_mu, self.y_std)

    return real_masses.squeeze(0).detach()

  def return_mse_and_grad(self, parametrs):
    if not isinstance(parametrs, torch.Tensor):
      parametrs = torch.tensor(parametrs, dtype=torch.double)
    else:
      parametrs = parametrs.clone().detach().to(dtype=torch.double)

    if parametrs.dim() == 1:
      parametrs = parametrs.unsqueeze(0)

    parametrs.requires_grad_(True)

    parametrs_unit = to_unit_cube(parametrs, self.lower_bounds, self.upper_bounds)
    prediction = make_prediction(self.emulator, self.likelihood, parametrs_unit)
    mass_values = destandardise_y(format_y_from_batch(prediction.mean), self.y_mu, self.y_std)

    # pred_var_unit = format_y_from_batch(prediction.variance)
    # emulator_var = pred_var_unit * (self.y_std**2)
    # total_var = (self.target_masses_errors**2) + emulator_var.detach()

    diff = (mass_values - self.target_masses)
    mse = torch.mean((diff / self.target_masses)**2)

    grad = torch.autograd.grad(
          outputs=mse,
          inputs=parametrs,
          retain_graph=False,
          create_graph=False
      )[0]
    return mse.item(), grad.squeeze(0)