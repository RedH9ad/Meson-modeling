#include "autodiff/forward/dual/dual.hpp"
#include "numerical_model.hpp"
#include "variable_functions.hpp"
#include "variable_functions.hpp"
#include <cmath>
#include <iostream>

int main() {
  auto deform = Metric_Deformation<autodiff::dual2nd>();
  auto scale_factor = Metric_Scale_Factor<autodiff::dual2nd>();
  auto model = Mesons_Model<autodiff::dual2nd>(1000, 0.380, deform, scale_factor);
  auto res = model.get_meson_mass_a();
  for (size_t i = 0; i < 10; i++) {
    std::cout << std::sqrt(res(i)) << "\n";
  }

  std::cout << "----------------------------" << "\n";
  std::cout << "Далее расчитаные численно" << "\n";
  std::cout << "----------------------------" << "\n";

  auto deform_num = Metric_Deformation<double>();
  auto scale_factor_num = Metric_Scale_Factor<double>();
  auto model_num = Mesons_Model<double>(1000, 0.380, deform_num, scale_factor_num);
  auto res_num = model_num.get_meson_mass_num();
  for (size_t i = 0; i < 10; i++) {
    std::cout << std::sqrt(res_num(i)) << "\n";
  }
}