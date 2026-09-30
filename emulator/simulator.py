import meson_lib
import torch

# Задаем деформацию метрики A(z)
C_WARP = 1.16
def simulator(L, k_sw_impact, metric_deformation):
    "Функиция принимающая k, L, и анзац и возвращающая массы тяжелых мезонов"
    masses = meson_lib.heavy_meson_mass_counter(L, k_sw_impact, metric_deformation)
    return torch.tensor(masses, dtype=torch.double)


def metric_deformation(z: float) -> float:
    return -(C_WARP / 8.0) * (z**2)

# if __name__ == "__main__":
#     print("Начинаю исполнение программы")
#     masses = meson_lib.heavy_meson_mass_counter(1.0, 0.8, metric_deformation)

#     print("Masses shape:", masses.shape)
#     print("Meson masses:", masses)


