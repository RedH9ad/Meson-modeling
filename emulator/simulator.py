import meson_lib
import torch


name = "param"
# name = "chen"
def simulator(L, k_sw_impact, metric_deformation):
    "Функиция принимающая k, L, и анзац и возвращающая массы тяжелых мезонов"
    masses = meson_lib.charm_hw(metric_deformation.theta_temp.numpy()[-1], L, k_sw_impact, name, metric_deformation.theta_temp.numpy()[:-1])
    return torch.tensor(masses, dtype=torch.double)



