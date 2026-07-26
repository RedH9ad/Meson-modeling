import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("build/meson_masses.csv")
index = np.array(df["index"])
mass_a = np.array(df["mass_autodiff"])
mass_n = np.array(df["mass_numerical"])

mass_a = mass_a[100 : -10]
mass_n = mass_n[100 : -10]
index = index[100 : -10]

plt.plot(index, mass_a, "o", color="red", label="Зависимость квадрата массы от индекса автодиф")
plt.plot(index, mass_n, "o", color="blue", label="Зависимость квадрата массы от индекса численно")
plt.xlabel("Индекс")
plt.ylabel("Масса мезона, ГэВ")
plt.legend()
plt.show()