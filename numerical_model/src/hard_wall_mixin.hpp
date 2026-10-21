#pragma once
#include <cmath>
#include <csignal>
#include <vector>
#include <stdexcept>
#include <autodiff/forward/real.hpp>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

template <typename Model>
class HardWallMixin {
  public:
  HardWallMixin() = default;
  Model& model();
  std::pair<Eigen::VectorXd, Eigen::VectorXd> build_threediag_matrix_vectors();
  Eigen::MatrixXd build_weight_matrix();
  //Матрица K из инструкции left обозвана
  Eigen::MatrixXd build_left_matrix();
  Eigen::MatrixXd build_mass_addition_matrix();
  Eigen::VectorXd count_eigenvals_sturm();

  void initialise_flux();
  void initialise_weights();
  void initialise_mass_addition();
  void initialise_system();

  void initialise_flux_art();
  void initialise_weights_art();
  void initialise_mass_addition_art();
  void initialise_system_art();

 private:
 //В качестве полей здесь будут выступать векторы t = (main_coef(z_{i+0.5} / net_step)) 
 //Вектор weights - w из инструкции
 //mass_additions - q из инструкции
 std::vector<double> flux_coeffs {};
 std::vector<double> weights {};
 std::vector<double> mass_additions {};

};



#include "hard_wall_mixin_impl.hpp"