from batch_emulator import *
from emulator_wrapper import *
from simulator import *
import torch
from tqdm.auto import tqdm
import pandas as pd
import sys
import matplotlib.pyplot as plt

#В функцию передаются два параметра относящихся к мезонам
#k_sw_impact и L и они фиксируются(пока с Аней обсудили что просто единицами и если захотим и их подгонять чета чета сделаю)

#Что может делать пользователь: нужно создать объект класса, после чего просто вызвать Train() и оно обучит эмулятор
#Также при создании класса есть возможность выбора - можно либо задать bound_percent_max если нужно чтобы максимальное процентное отклонение предсказания не превысило
#Или задать bound_percent_mean - если нужно чтобы среднее не превысило
class Emulator_Trainer:
  "Класс обучающий эмулятор на основе симулятора"
  "gp_model - класс используемого эмулятора"
  "gp_likelihood - класс используемого лайклихуда"
  "Anzats - класс используемого анзатса"
  "bound_list - torch.tensor состоящий из torch.tensor границ в каждом из которых два элемента"
  "simulator - используемый симулятор"
  "start_train_per_dim - Количество стартовых точек/размерность"
  "valid_per_dim валидационных точек/размерность"
  "bound_percent - целевая максимальная ошибка эмулятора на целевом наборе"
  "max_iters - максимальное возможное количество итераций обучения"
  "emulator_path - путь(а точнее просто название файла) где лежит конфиг эмулятора"
  def __init__(
      self, 
      Anzats, 
      bounds_list, 
      gp_model=MesonGP, 
      gp_likelihood=gpytorch.likelihoods.MultitaskGaussianLikelihood, 
      k_sw_impact=1.0, 
      L=1.0, 
      simulator=simulator, 
      anzats_constants=None, 
      start_train_per_dim=10, 
      valid_per_dim=5*10, 
      meson_masses_number=12, 
      bound_percent_max=None,
      bound_percent_mean = 0.5,
      max_iters=50000, 
      show_metrics=False, 
      validation_path="validation_data.csv", 
      train_path="train_data.csv",
      train_initialised = False,
      validation_initialised = False,
      emulator_path="emulator_state.pth"
    ):
    self.validation_path = validation_path
    self.validation_initialised = validation_initialised

    self.train_path = train_path
    self.train_initialised = train_initialised

    self.emulator_path = emulator_path
    self.show_metrics = show_metrics

    self.gp_model = gp_model
    self.gp_likelihood_class = gp_likelihood
    self.k_sw_impact = k_sw_impact
    self.L = L
    self.Anzats = Anzats
    self.meson_masses_number = meson_masses_number

    lower_bounds = []
    upper_bounds = []

    self.anzats_parametrs_size = len(bounds_list)
    self.input_dimension = len(bounds_list)

    for bound in bounds_list:
      lower_bounds.append(bound[0])
      upper_bounds.append(bound[1])

    self.lower_bounds = torch.tensor(lower_bounds, dtype=torch.double)
    self.upper_bounds = torch.tensor(upper_bounds, dtype=torch.double)

    self.simulator = simulator
    self.anzats_constants = anzats_constants

    self.validation_points_number = valid_per_dim * len(bounds_list)
    self.init_training_set_number = start_train_per_dim * len(bounds_list)

    self.emulator = None
    self.likelihood = None

    self.validation_x = None
    self.validation_y = None
    self.validation_x_unit = None
    self.validation_y_unit = None

    self.y_mu = None
    self.y_std = None

    self.msll_errors = []
    self.percent_error_max_arr = []
    self.percent_error_mean_arr = []
    self.rmse_errors = []

    self.bound_percent_max = bound_percent_max
    self.bound_percent_mean = bound_percent_mean
  
    self.max_iters = max_iters

    self.train_x = None
    self.train_y = None

    self.train_x_real = None
    self.train_y_real = None

    self.trained = False


  def Train(self):
    "Выполняет весь цикл работы, при использовании нужно вызывать только эту функцию"
    self.initialise_validation_set()
    self.initialise_emulator()
    self.train_emulator()

  def train_on_generated_data(self, start_points=20):
    "Метод тренировки с максимизацией msll"
    if not self.validation_initialised or not self.train_initialised:
      raise RuntimeError("Исследовать динамику обучения можно только на сгенерированных данных")
    self.initialise_validation_set()
    self.initialise_emulator()
    indexies = self.generate_start_pool(start_points)
    temp_x = self.train_x[indexies]
    temp_y = self.train_y[indexies]
    temp_x_unit = to_unit_cube(temp_x, self.lower_bounds, self.upper_bounds)  
    temp_y_unit = standardise_y(temp_y, self.y_mu, self.y_std)

    likelihood = self.gp_likelihood_class(num_tasks=self.meson_masses_number // 4, batch_shape=torch.Size([4])).double()
    emulator = self.gp_model(temp_x_unit, format_y_to_batch(temp_y_unit), likelihood).double()
    states = [{"em_state" : emulator.state_dict(), "lik_state" : likelihood.state_dict(), "x" : emulator.train_inputs[0], "y" : emulator.train_targets}]
    trainBFGS(emulator, likelihood)
    prediction = make_prediction(emulator, likelihood, self.validation_x_unit)

    self.msll_errors.append(MSLL(prediction, self.validation_y_unit).item())
    self.rmse_errors.append(RMSE(prediction, self.validation_y_unit).item())
    max_percent_error = percent_error_max(prediction, self.validation_y_unit, self.y_mu, self.y_std)
    mean_percent_error = percent_error_mean(prediction, self.validation_y_unit, self.y_mu, self.y_std)
    self.percent_error_max_arr.append(max_percent_error)
    self.percent_error_mean_arr.append(mean_percent_error)
  
    while(len(temp_x_unit) < len(self.train_x)):
      next_id = adapt_mse_batch(emulator, likelihood, to_unit_cube(self.train_x, self.lower_bounds, self.upper_bounds))
      next_x = self.train_x[next_id]
      next_y = self.train_y[next_id]

      next_x = to_unit_cube(next_x, self.lower_bounds, self.upper_bounds)
      next_y = standardise_y(next_y, self.y_mu, self.y_std)
      
      temp_x_unit = torch.cat([emulator.train_inputs[0], next_x.view(1, -1)])
      temp_y_unit = torch.cat([format_y_from_batch(emulator.train_targets), next_y.view(1, -1)])

      emulator.set_train_data(inputs=temp_x_unit, targets=format_y_to_batch(temp_y_unit), strict=False)
      trainBFGS(emulator, likelihood)
      prediction = make_prediction(emulator, likelihood, self.validation_x_unit)

      self.msll_errors.append(MSLL(prediction, self.validation_y_unit).item())
      self.rmse_errors.append(RMSE(prediction, self.validation_y_unit).item())
      max_percent_error = percent_error_max(prediction, self.validation_y_unit, self.y_mu, self.y_std)
      mean_percent_error = percent_error_mean(prediction, self.validation_y_unit, self.y_mu, self.y_std)
      self.percent_error_max_arr.append(max_percent_error)
      self.percent_error_mean_arr.append(mean_percent_error)

      states.append({"em_state" : emulator.state_dict(), "lik_state" : likelihood.state_dict(), "x" : emulator.train_inputs[0], "y" : emulator.train_targets})

    self.plot_tensor_values(torch.tensor(self.msll_errors, dtype=torch.double), "Mean Standardized Log Loss")
    self.plot_tensor_values(torch.tensor(self.percent_error_max_arr, dtype=torch.double), "Max error, %")
    self.plot_tensor_values(torch.tensor(self.percent_error_mean_arr, dtype=torch.double), "Mean error, %") 
    while True:
      choice = int(input("\nВыберите состояние: "))
      if (isinstance(choice, int)):
        print(f"Сохраняем состояние с количеством точек {choice + start_points}")
        stat = {
        'model_state_dict' : states[choice]["em_state"],
        'likelihood_state_dict' : states[choice]["lik_state"],

        'train_x' : states[choice]["x"],
        'train_y' : states[choice]["y"],

        'y_mu': self.y_mu,
        'y_std': self.y_std
        }
        torch.save(stat, self.emulator_path)
        print(f"Состояние эмулятора сохранено в {self.emulator_path}")
        return
  

  
  def generate_start_pool(self, start_points):
    x_all = to_unit_cube(self.train_x, self.lower_bounds, self.upper_bounds)

    center = x_all.mean(dim=0, keepdim=True)
    first_idx = torch.argmin(torch.norm(x_all - center, dim=1)).item()

    selected_indices = [first_idx]
    min_distances = torch.norm(x_all - x_all[first_idx], dim=1)

    for _ in range(1, start_points):
        next_idx = torch.argmax(min_distances).item()
        selected_indices.append(next_idx)

        dist_to_new = torch.norm(x_all - x_all[next_idx], dim=1)
        min_distances = torch.minimum(min_distances, dist_to_new)

    return selected_indices


  def generate_data_set(self, number_of_points):
    "Генерирует дата сеты входные точки на основе латинского гиперкуба, выходы отдает симулятор"
    model_input = base_latinian_sample(self.input_dimension, number_of_points, self.lower_bounds, self.upper_bounds)
    model_output = []
    for m_input in tqdm(model_input, desc="Генерация данных", leave=False):
      metric_deformation = self.Anzats(m_input)
      output = self.simulator(self.L, self.k_sw_impact, metric_deformation)
      model_output.append(output)
    model_output = torch.stack([t.detach().cpu() for t in model_output]).to(dtype=torch.double)
    return model_input, model_output


  def initialise_emulator(self):
    "Собирает эмулятор для обучения"
    if (self.emulator is not None):
      print("Эмулятор уже инициализирован")
      return

    if not self.train_initialised:
      self.train_x, self.train_y = self.generate_data_set(self.init_training_set_number)
      save_data_to_csv(self.train_x, self.train_y, self.train_path)
      print(f"Начальные обучающие данные сохранены в {self.train_path}")
    else:
      print("Читаем обучающие данные из файла")
      self.train_x, self.train_y = initialise_data_from_file(self.train_path)

    print("Стартовые данные сгенерированы")

    train_x_unit = to_unit_cube(self.train_x, self.lower_bounds, self.upper_bounds)
    train_y_unit = standardise_y(self.train_y, self.y_mu, self.y_std)

    # self.likelihood = self.gp_likelihood_class(num_tasks=self.meson_masses_number // 4, batch_shape=torch.Size([4])).double()
    # self.emulator = self.gp_model(train_x_unit, format_y_to_batch(train_y_unit), self.likelihood).double()
    # print("Эмулятор создан, стартовый подгон гиперпарметров")
    # trainBFGS(self.emulator, self.likelihood)

  def initialise_validation_set(self):
    "Генерирует валидационные данные"
    if self.validation_x is not None and self.validation_y is not None:
      print("Валидационный сет инициализирован")
      return
    print("Начинаем иниициализацию валидационного сета")

    if not self.validation_initialised:
      self.validation_x, self.validation_y = self.generate_data_set(self.validation_points_number)
      save_data_to_csv(self.validation_x, self.validation_y, self.validation_path)
      print(f"Валидационные данные сохранены в {self.validation_path}")
    else:
      print("Читаем валидационные данные из файла")
      self.validation_x, self.validation_y = initialise_data_from_file(self.validation_path)
    self.validation_x_unit = to_unit_cube(self.validation_x, self.lower_bounds, self.upper_bounds)
    self.validation_y_unit, self.y_mu, self.y_std = standardise_y_first(self.validation_y)


  def make_one_step(self, train_need):
    "Добавляет одну точку в обучающий набор, и собирает метрики обученности"
    next_x = self.adaptive_learning_using_mse()
    metric_deformation = self.Anzats(next_x)
    next_y = self.simulator(self.L, self.k_sw_impact, metric_deformation)

    next_x = to_unit_cube(next_x, self.lower_bounds, self.upper_bounds)
    next_y = standardise_y(next_y, self.y_mu, self.y_std)

    self.train_x = torch.cat([self.emulator.train_inputs[0], next_x.view(1, -1)])
    self.train_y = torch.cat([format_y_from_batch(self.emulator.train_targets), next_y.view(1, -1)])

    self.train_x_real = from_unit_cube(self.train_x, self.lower_bounds, self.upper_bounds)
    self.train_y_real = destandardise_y(self.train_y, self.y_mu, self.y_std)

    self.emulator.set_train_data(inputs=self.train_x, targets=format_y_to_batch(self.train_y), strict=False)
    if (train_need):
      trainBFGS(self.emulator, self.likelihood)
    prediction = make_prediction(self.emulator, self.likelihood, self.validation_x_unit)

    self.msll_errors.append(MSLL(prediction, self.validation_y_unit).item())
    self.rmse_errors.append(RMSE(prediction, self.validation_y_unit).item())
    max_percent_error = percent_error_max(prediction, self.validation_y_unit, self.y_mu, self.y_std)
    mean_percent_error = percent_error_mean(prediction, self.validation_y_unit, self.y_mu, self.y_std)
    self.percent_error_max_arr.append(max_percent_error)
    self.percent_error_mean_arr.append(mean_percent_error)

    if (self.bound_percent_max is None):
      return mean_percent_error
    return max_percent_error


  def train_emulator(self):
    "Запускает адаптивное обучение эмулятора,"
    "Работает пока не превысит лимит итераций или не получит необходимый предел ошибки"
    if (self.bound_percent_max is None):
      bound_percent = self.bound_percent_mean
    else:
      bound_percent = self.bound_percent_max

    percent_error = 1e10
    iterations = 0
    with tqdm(total=self.max_iters, desc="Обучение эмулятора") as pbar:
      while(percent_error > bound_percent and iterations < self.max_iters):
        # if (iterations % 5 == 0):
        #   percent_error = self.make_one_step(True)
        # else:
        #   percent_error= self.make_one_step(False)
        percent_error = self.make_one_step(True)
        iterations += 1
        pbar.update(1)
        pbar.set_postfix({
          "Max Err %": f"{percent_error:.2f}%", 
          "Target": f"{bound_percent}%"
        })
    self.trained = True

    self.plot_tensor_values(torch.tensor(self.msll_errors, dtype=torch.double), "Mean Standardized Log Loss")
    self.plot_tensor_values(torch.tensor(self.percent_error_max_arr, dtype=torch.double), "Max error, %")
    self.plot_tensor_values(torch.tensor(self.percent_error_mean_arr, dtype=torch.double), "Mean error, %") 
    
    self.check_ebalo()
    self.save_emulator_state()

  def check_ebalo(self):
    "Проверяет насколько хорош итоговый эмулятор, требует верификацию обученности"
    msll, percent_max, percent_mean = self.test_metrics()
    prediction = make_prediction(self.emulator, self.likelihood, self.validation_x_unit)
    percent_error_diagnostic(prediction, self.validation_y_unit, self.y_mu, self.y_std, self.validation_x)

    while True:
      print("Обчение завершено")
      print(f"Mean Standardized Log Loss: {msll}")
      if (msll > 0):
        print("ВНИМАНИЕ ЭМУЛЯТОР АППРОКСИМИРУЕТ ТОЧКИ НЕ ЛУЧШЕ БЕЛОГО ШУМА")

      if (self.bound_percent_max is not None):
        print(f"Максимальное отклонение от валидационного набора {percent_max}%, целевое: {self.bound_percent_max}%")
        if (percent_max > self.bound_percent_max):
          print("ВНИМАНИЕ, МАКСИМАЛЬНАЯ ОШИБКА БОЛЬШЕ ЦЕЛЕВОЙ")

      print(f"Среднее отклонение от валидационного набора {percent_mean}")

      choice = input("\nПродолжить работу? (Y/N): ").strip().upper()
      if choice == 'Y':
        return True
      elif choice == 'N':
        print("Остановка работы.")
        sys.exit(0)
      print("Ошибка: введите 'Y' для продолжения или 'N' для выхода.")

  def test_metrics(self):
    "Генерирует msll и максимальную ошибку в процентах для валидации"
    prediction = make_prediction(self.emulator, self.likelihood, self.validation_x_unit)
    msll = MSLL(prediction, self.validation_y_unit)
    max_percent_error = percent_error_max(prediction, self.validation_y_unit, self.y_mu, self.y_std)
    mean_percent_error = percent_error_mean(prediction, self.validation_y_unit, self.y_mu, self.y_std)
    return msll, max_percent_error, mean_percent_error

  def save_emulator_state(self):
    "Сохраняет состояние эмулятора"
    state = {
      'model_state_dict' : self.emulator.state_dict(),
      'likelihood_state_dict' : self.likelihood.state_dict(),

      'train_x' : self.emulator.train_inputs[0],
      'train_y' : self.emulator.train_targets,

      'y_mu': self.y_mu,
      'y_std': self.y_std
    }
    torch.save(state, self.emulator_path)
    print(f"Состояние эмулятора сохранено в {self.emulator_path}")


  #Генерирует следующую точку
  def adaptive_learning_using_mse(self, tries=15, iterations=50):
    "Генерирует следующую точку(максимальная дисперсия эмулятора в границах)"
    self.emulator.eval()
    self.likelihood.eval()

    candidates_unit = []
    cand_variances = []

    dim = self.lower_bounds.shape[0]

    for _ in range(tries):
      x_unit = torch.rand(dim, dtype=self.lower_bounds.dtype, requires_grad=True)
      optimizer = torch.optim.Adam([x_unit], lr=0.05)

      for _ in range(iterations):
        optimizer.zero_grad()

        prediction = make_prediction(self.emulator, self.likelihood, x_unit.unsqueeze(0))
        variance = format_y_from_batch(prediction.variance.squeeze(0)).mean()

        loss = -variance
        loss.backward()
        optimizer.step()

        with torch.no_grad():
          x_unit.data.clamp_(0.0, 1.0)

      with torch.no_grad():
        candidates_unit.append(x_unit.detach().clone())
        cand_variances.append(variance.item())

    cand_variances = torch.tensor(cand_variances, dtype=torch.double)
    best_id = torch.argmax(cand_variances)
    best_unit_point = candidates_unit[best_id]

    return from_unit_cube(best_unit_point, self.lower_bounds, self.upper_bounds)

  def plot_tensor_values(self, tensor, name, color="blue"):

    if tensor.dim() > 1:
        tensor = tensor.flatten()

    y_values = tensor.detach().cpu().numpy()

    x_values = range(len(y_values))

    plt.figure(figsize=(10, 5))
    plt.plot(x_values, y_values, marker='o', linestyle='-', color=color, markersize=4)
    
    plt.xlabel("Итерация обучения")
    plt.ylabel(f"Значение {name}")
    plt.title(f"Зависимость {name} от итерации обучения")
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()





def save_data_to_csv(inputs: torch.Tensor, outputs: torch.Tensor, filename: str):
  print("Сохранение данных...")

  if inputs.ndim == 1:
      inputs = inputs.unsqueeze(1)
  if outputs.ndim == 1:
      outputs = outputs.unsqueeze(1)

  input_size = inputs.shape[1]
  output_size = outputs.shape[1]

  in_cols = [f"in[{i}]" for i in range(input_size)]
  out_cols = [f"out[{i}]" for i in range(output_size)]

  combined_data = (
      torch.cat([inputs, outputs], dim=1)
      .detach()
      .cpu()
      .numpy()
  )

  df = pd.DataFrame(combined_data, columns=in_cols + out_cols)
  df.to_csv(filename, index=False)
  print(f"Данные успешно сохранены в {filename}")


def initialise_data_from_file(filename):
  df = pd.read_csv(filename)
  in_cols = [col for col in df.columns if col.startswith("in")]
  out_cols = [col for col in df.columns if col.startswith("out")]

  inputs = torch.tensor(df[in_cols].values, dtype=torch.double)
  outputs = torch.tensor(df[out_cols].values, dtype=torch.double)

  return inputs, outputs
#Идет оптимизация дисперсии в точке, то есть, по сути нужно отдать prediction vatience и оптимизировать

#Мы хотим передавать 