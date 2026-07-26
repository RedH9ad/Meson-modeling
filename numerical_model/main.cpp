#include "autodiff/forward/dual/dual.hpp"
#include "numerical_model.hpp"
#include "variable_functions.hpp"
#include "variable_functions.hpp"
#include <cmath>
#include <iostream>
#include <fstream>

int main() {
  //Для второго тест кейса
  std::vector<double> parametrs {0.473, 1.7};
  auto deform = Metric_Deformation<autodiff::dual2nd>(parametrs);
  auto scale_factor = Metric_Scale_Factor<autodiff::dual2nd>();
  auto model = Mesons_Model<autodiff::dual2nd>(1000, 0.473, deform, scale_factor);
  auto res = model.get_meson_mass_a();
  for (size_t i = 0; i < 10; i++) {
    std::cout << std::sqrt(res(i)) << "\n";
  }

  std::cout << "----------------------------" << "\n";
  std::cout << "Далее расчитаные численно" << "\n";
  std::cout << "----------------------------" << "\n";

  std::vector<double> parametrs_n {0.473, 1.7};
  auto deform_num = Metric_Deformation<double>(parametrs_n);
  auto scale_factor_num = Metric_Scale_Factor<double>();
  auto model_num = Mesons_Model<double>(1000,0.473, deform_num, scale_factor_num);
  auto res_num = model_num.get_meson_mass_num();
  for (size_t i = 0; i < 10; i++) {
    std::cout << std::sqrt(res_num(i)) << "\n";
  }

  std::ofstream csv_file("meson_masses_case2.csv");
    if (!csv_file.is_open()) {
      std::cerr << "Ошибка: не удалось открыть файл для записи!" << std::endl;
       return 1;
    }
  csv_file << "index,mass_autodiff,mass_numerical\n";
  for (size_t i = 0; i < res.size(); i++) {
    csv_file << i << "," << res(i) << "," << res_num(i) << '\n';
  }

  csv_file.close();
  std::cout << "\n[INFO] Результаты успешно записаны в файл meson_masses.csv\n";
}