#include "autodiff/forward/dual/dual.hpp"
#include "autodiff/forward/real/real.hpp"
#include "numerical_model.hpp"
#include "variable_functions.hpp"
#include "variable_functions.hpp"
#include <cmath>
#include <iostream>
#include <fstream>
#include <ostream>
#include <vector>

int main() {
  auto deform = Metric_Deformation<autodiff::real3rd>();
  double acc = 0.0;
  Eigen::VectorXd goal {{0.759, 1.202, 1.519, 1.779, 2.005, 2.207, 2.393}};

  std::vector<double> error;
  std::vector<double> test_ks;

  size_t counter = 1;
  for (double k = 0.000; k < 0.100; k += 0.005) {
    auto model = Mesons_Model<autodiff::real3rd>(1000, 15, k, deform);
    auto res_dual = model.get_rho_meson_mass_a();
    auto res_raw = res_dual.unaryExpr([](const autodiff::dual2nd& x) {
            return autodiff::val(x); 
        });

    auto res = res_raw.head(7);
    auto percent_error = (goal.array() - res.array()) / goal.array() * 100;

    auto mean_error = percent_error.array().abs().mean();
    error.push_back(mean_error);
    test_ks.push_back(k);
    std::cout << "Цикл" << counter << "из" << int(1.51 / 0.01) << std::endl;
    counter += 1;
  }

  std::ofstream csv_file("ks and errors.csv");
  if (!csv_file.is_open()) {
    std::cerr << "Ошибка: не удалось открыть файл для записи!" << std::endl;
      return 1;
  }

  csv_file << "k value,error\n";
  for (size_t i = 0; i < error.size(); i++) {
    csv_file << test_ks[i] << "," << error[i] << '\n';
  }

  csv_file.close();
  std::cout << "\n[INFO] Результаты успешно записаны в файл meson_masses.csv\n";
}