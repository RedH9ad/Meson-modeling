import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("build/meson_masses_empty.csv")
df_1 = pd.read_csv("build/meson_masses_log.csv")
df_2 = pd.read_csv("build/meson_masses_case2.csv")

index = np.array(df["index"])
mass_a_empty = np.array(df["mass_autodiff"])
mass_n_empty = np.array(df["mass_numerical"])

mass_a_log = np.array(df_1["mass_autodiff"])
mass_n_log = np.array(df_1["mass_numerical"])

mass_a_case2 = np.array(df_2["mass_autodiff"])
mass_n_case2 = np.array(df_2["mass_numerical"])

# mass_a = mass_a[0 : -10]
# mass_n = mass_n[0 : -10]

index = index[0 : -980]
mass_a_empty = mass_a_empty[0 : -980]
mass_a_log = mass_a_log[0 : -980]
mass_a_case2 = mass_a_case2[0 : -980]

plt.plot(index, mass_a_empty ** 2, "o", color="red", label="Зависимость квадрата массы от индекса A(z) = 0")
plt.plot(index, mass_a_log ** 2, "o", color="blue", label="Зависимость квадрата массы от индекса A(z) = ln(1 + kz)")
plt.plot(index, mass_a_case2 ** 2, "o", color="green", label="Зависимость квадрата массы от индекса A(z) = функция из case 2")
plt.xlabel("Индекс")
plt.ylabel("Масса мезона в, ГэВ ** 2")
plt.legend()
plt.show()