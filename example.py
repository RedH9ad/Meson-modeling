from emulator.emulator_worker import Emulator_Worker
import torch
#Список границ в формате [lower, upper] для параметров анзаца Chen-Huan
bounds_list_huan = [[1e-4, 3.5], [1e-4, 3.5], [-3.5, 3.5], [-3.5, 0], [-3.5, 3.5], [1, 15]]

#Список границ в формате [lower, upper] для параметров анзаца параметрик
bound_list_parametric = [[-5, 0], [-5, 0], [-5, 0], [1e-4, 5], [1e-6, 5], [1e-6, 5], [1, 15]]

#Путь до состояния обученного эмулятора Huan анзаца
huan_state_path = "trained_emulator_states/chenhuan_emulator_state_final.pth"

#Путь до состояния обученного эмулятора Parametric анзаца
parametric_state_path = "trained_emulator_states/parametric_emulator_final.pth"

#Далее нужно инициализировать воркер следующим образом
emulator = Emulator_Worker(bound_list_parametric, emulator_path=parametric_state_path)

#Пример для параметрического:
parametrs = torch.tensor([1, 1, 1, -1, 1, 1, 6])

#Возвращает массы:
print(emulator.return_mass_values(parametrs))

#Возвращает метрику(в стоке mse) и ее градиент по параметрам 
res = emulator.return_mse_and_grad(parametrs)
print(f"MSE: {res[0]} градиент: {res[1:]}")