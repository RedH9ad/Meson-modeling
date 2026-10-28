# Meson Mass Emulator

Суррогатный эмулятор на основе **гауссовских процессов (Gaussian Processes)** для ускоренного расчёта спектра масс мезонов, вычисления относительной среднеквадратичной ошибки ($\text{MSE}$) и её аналитических градиентов по параметрам анзаца.

Эмулятор заменяет прямое численное решение дифференциальных уравнений (задачи Штурма–Лиувилля / уравнения Шрёдингера), позволяя проводить высокоскоростную градиентную оптимизацию и подгонку параметров модели под экспериментальные данные.

---

## Основные возможности

- **Суррогатное моделирование на базе GP:** многозадачный гауссовский процесс (`gpytorch.likelihoods.MultitaskGaussianLikelihood`, `MesonGP`) аппроксимирует зависимость масс состояний от параметров потенциала.
- **Автоматическое масштабирование:** внутреннее преобразование входных параметров в единичный гиперкуб $[0, 1]^d$ (`to_unit_cube`) и дестандартизация предсказанных масс (`destandardise_y`).
- **Градиенты без численной аппроксимации:** вычисление точных градиентов функции потерь $\nabla_\theta \text{MSE}$ через PyTorch Autograd за один вызов.
- **Поддержка двух конфигураций анзаца:**
  - **Chen–Huan** ($d = 6$ параметров);
  - **Parametric** ($d = 7$ параметров).

---

## Диапазоны параметров и чекпоинты

Воркер принимает диапазоны допустимых значений параметров `[lower, upper]` для нормализации и путь к обученному состоянию:

| Анзац | Параметров ($d$) | Файл состояния (`.pth`) | Границы (`bounds_list`) |
| :--- | :---: | :--- | :--- |
| **Chen–Huan** | 6 | `trained_emulator_states/chenhuan_emulator_state_final.pth` | `[[1e-4, 3.5], [1e-4, 3.5], [-3.5, 3.5], [-3.5, 0], [-3.5, 3.5], [1, 15]]` |
| **Parametric** | 7 | `trained_emulator_states/parametric_emulator_final.pth` | `[[-5, 0], [-5, 0], [-5, 0], [1e-4, 5], [1e-6, 5], [1e-6, 5], [1, 15]]` |

---

## Быстрый старт

Класс `Emulator_Worker` используется после того, как эмулятор был обучен тренером (`trainer.Train()`) и сохранён в `.pth` файл.

```python
import torch
from emulator.emulator_worker import Emulator_Worker

# 1. Задание границ параметров и чекпоинта (пример для Parametric анзаца)
bound_list_parametric = [
    [-5, 0], [-5, 0], [-5, 0],
    [1e-4, 5], [1e-6, 5], [1e-6, 5],
    [1, 15]
]
parametric_state_path = "trained_emulator_states/parametric_emulator_final.pth"

# 2. Инициализация воркера (автоматически загружает веса и переводит модель в eval-режим)
emulator = Emulator_Worker(
    bounds_list=bound_list_parametric,
    emulator_path=parametric_state_path
)

# 3. Вектор параметров анзаца
parameters = torch.tensor([1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 6.0], dtype=torch.double)

# 4. Получение предсказанных масс мезонов (в ГэВ)
masses = emulator.return_mass_values(parameters)
print("Вычисленные массы:", masses)

# 5. Расчет MSE и градиентов по параметрам для оптимизатора
mse, grad = emulator.return_mse_and_grad(parameters)
print(f"MSE: {mse}")
print(f"Градиент d(MSE)/d(params): {grad}")
```

### Пример инициализации для анзаца Chen–Huan:

```python
bounds_list_huan = [
    [1e-4, 3.5], [1e-4, 3.5],
    [-3.5, 3.5], [-3.5, 0], [-3.5, 3.5],
    [1, 15]
]
huan_state_path = "trained_emulator_states/chenhuan_emulator_state_final.pth"

emulator_huan = Emulator_Worker(
    bounds_list=bounds_list_huan,
    emulator_path=huan_state_path
)
```

---

## Архитектура и описание `Emulator_Worker`

### Конструктор `__init__`

```python
Emulator_Worker(
    bounds_list, 
    gp_model=MesonGP,
    gp_likelihood=gpytorch.likelihoods.MultitaskGaussianLikelihood,
    target_masses=torch.tensor([3.140, 3.6569, 4.0558], dtype=torch.double),
    target_masses_errors=None,
    bound_percent=0.5, 
    emulator_path="emulator_state.pth"
)
```

#### Параметры:
- **`bounds_list`** (*list[list[float]]*): список диапазонов `[lower, upper]` для каждого параметра анзаца. Используется для переноса входного пространства в единичный гиперкуб $[0, 1]^d$.
- **`gp_model`** (*type*): класс модели гауссовского процесса (по умолчанию `MesonGP`).
- **`gp_likelihood`** (*type*): класс правдоподобия (по умолчанию `MultitaskGaussianLikelihood`).
- **`target_masses`** (*torch.Tensor*, опционально): экспериментальные целевые массы мезонов (по умолчанию чармоний: $J/\psi(1S) = 3.140$, $\psi(2S) = 3.6569$, $\psi(3S) = 4.0558$ ГэВ).
- **`target_masses_errors`** (*torch.Tensor*, опционально): экспериментальные погрешности целевых масс.
- **`bound_percent`** (*float*): целевая погрешность эмулятора (по умолчанию `0.5%`).
- **`emulator_path`** (*str*): путь к сохранённому словарю состояния `.pth`.

При создании объекта автоматически вызывается метод `Initialise_emulator()`.

---

### Методы класса

#### 1. `Initialise_emulator()`
Загружает сохранённое состояние эмулятора из файла `emulator_path`.
- Извлекает `train_x`, `train_y`, `y_mu`, `y_std`, а также веса модели и правдоподобия (`model_state_dict`, `likelihood_state_dict`).
- Восстанавливает структуру `gp_likelihood` и `gp_model`.
- Переводит модель и правдоподобие в режим инференса (`eval()`).

#### 2. `return_mass_values(parametrs) -> torch.Tensor`
Вычисляет физические массы мезонных состояний:
- Принимает вектор параметров (тензор или список).
- Масштабирует параметры в единичный куб: $\theta_{\text{unit}} = \frac{\theta - \theta_{\min}}{\theta_{\max} - \theta_{\min}}$.
- Вычисляет среднее апостериорного распределения GP и дестандартизирует его:
  $$m = \mu_y + \hat{y} \cdot \sigma_y$$
- **Возвращает:** одномерный тензор `torch.Tensor` с физическими массами (в единицах ГэВ).

#### 3. `return_mse_and_grad(parametrs) -> tuple[float, torch.Tensor]`
Вычисляет относительную среднеквадратичную ошибку по отношению к целевым массам и её точный градиент:
- Функция потерь определяется как:
  $$\text{MSE} = \frac{1}{N} \sum_{i=1}^N \left( \frac{m_i(\theta) - m_i^{\text{target}}}{m_i^{\text{target}}} \right)^2$$
- Автоматически вычисляет вектор производных по входным параметрам анзаца:
  $$\nabla_\theta \text{MSE} = \left( \frac{\partial \text{MSE}}{\partial \theta_1}, \dots, \frac{\partial \text{MSE}}{\partial \theta_d} \right)$$
- **Возвращает кортеж `(mse, grad)`:**
  - `mse` (*float*): скалярное значение функции потерь;
  - `grad` (*torch.Tensor*): одномерный тензор градиентов размерности $d$.