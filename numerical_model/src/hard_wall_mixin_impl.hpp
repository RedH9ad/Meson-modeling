#pragma once
#include "hard_wall_mixin.hpp"
#include "numerical_model.hpp"
#include <cmath>
#include <cstddef>


template <typename Model>
Model& HardWallMixin<Model>::model() {
  return static_cast<Model&>(*this);
}

template <typename Model>
void HardWallMixin<Model>::initialise_flux() {
  auto net_step = model().net_step;
  const auto& net = model().net;
  auto total_net_nodes = model().total_net_nodes;
  for (size_t i = 0; i < total_net_nodes - 1; i++) {
    auto num = std::exp(static_cast<double>(model().metric_deformation(static_cast<double>(net[i].z) + 0.5 * net_step))) * kinetic_function_sturm(static_cast<double>(net[i].z) + 0.5 * net_step);
    auto den = net_step * (static_cast<double>(net[i].z) + 0.5 * net_step);
    flux_coeffs.push_back(num / den);
  }
}

template <typename Model>
void HardWallMixin<Model>::initialise_weights() {
  auto net_step = model().net_step;
  const auto& net = model().net;
  auto total_net_nodes = model().total_net_nodes;
  for (size_t i = 0; i < total_net_nodes - 1; i++) {
    auto num = std::exp(static_cast<double>(model().metric_deformation(static_cast<double>(net[i + 1].z)))) * kinetic_function_sturm(static_cast<double>(net[i + 1].z));
    auto den = static_cast<double>(net[i + 1].z);
    weights.push_back(num / den);
  }
}

template <typename Model>
void HardWallMixin<Model>::initialise_mass_addition() {
  auto net_step = model().net_step;
  const auto& net = model().net;
  auto total_net_nodes = model().total_net_nodes;
  auto meson = model().heavy_meson.value();
  for (size_t i = 0; i < total_net_nodes - 1; i++) {
    auto num = std::exp(3 * static_cast<double>(model().metric_deformation(static_cast<double>(net[i + 1].z)))) * kinetic_function_sturm(static_cast<double>(net[i + 1].z));
    auto den = std::pow(static_cast<double>(net[i + 1].z), 3);
    auto addition = meson.potential_addition();
    mass_additions.push_back((num * addition / den));
  }
}

template <typename Model>
void HardWallMixin<Model>::initialise_system() {
  flux_coeffs.clear();
  weights.clear();
  mass_additions.clear();
  model().inintialise_net();
  initialise_ext_dilaton();
  initialise_flux();
  initialise_weights();
  initialise_mass_addition();
}

template <typename Model>
Eigen::MatrixXd HardWallMixin<Model>::build_weight_matrix() {
  std::vector<double> temp;
  for (size_t i = 0; i < weights.size() - 1; i++) {
    temp.push_back(model().net_step * weights[i]);
  }
  temp.push_back(model().net_step / 2 * weights.back());
  Eigen::VectorXd temp_eigen = Eigen::Map<const Eigen::VectorXd>(temp.data(), temp.size());
  return temp_eigen.asDiagonal();
} 

template <typename Model>
Eigen::MatrixXd HardWallMixin<Model>::build_mass_addition_matrix() {
  std::vector<double> temp;
  for (size_t i = 0; i < mass_additions.size() - 1; i++) {
    temp.push_back(model().net_step * mass_additions[i]);
  }
  temp.push_back(model().net_step / 2 * mass_additions.back());
  Eigen::VectorXd temp_eigen = Eigen::Map<const Eigen::VectorXd>(temp.data(), temp.size());
  return temp_eigen.asDiagonal();
}

template <typename Model>
Eigen::MatrixXd HardWallMixin<Model>::build_left_matrix() {
  Eigen::MatrixXd left = Eigen::MatrixXd::Zero(flux_coeffs.size(), flux_coeffs.size());
  for (size_t i = 0; i < flux_coeffs.size() - 1; i++) {
    left(i, i + 1) = -flux_coeffs[i + 1];
    left(i + 1, i) = -flux_coeffs[i + 1];
    left(i, i) = flux_coeffs[i] + flux_coeffs[i + 1];
  }
  left(flux_coeffs.size() - 1, flux_coeffs.size() - 1) = flux_coeffs.back();
  return left;
}

template <typename Model>
std::pair<Eigen::VectorXd, Eigen::VectorXd> HardWallMixin<Model>::build_threediag_matrix_vectors() {
  Eigen::VectorXd diagonal(model().net.size() - 1);
  Eigen::VectorXd subdiagonal(model().net.size() - 2);
  auto left_m = build_left_matrix();
  auto weight_m = build_weight_matrix();
  auto addition_m = build_mass_addition_matrix();
  auto full_left = left_m + addition_m;

  for (size_t i = 0; i < diagonal.size() - 1; i++) {
    double w_i = std::max(weight_m(i, i), 1e-300);
    double w_next = std::max(weight_m(i + 1, i + 1), 1e-300);

    diagonal(i) = full_left(i, i) / w_i;
    subdiagonal(i) = full_left(i, i + 1) / (std::sqrt(w_i) * std::sqrt(w_next));
  }
  size_t last = diagonal.size() - 1;
  double w_last = std::max(weight_m(last, last), 1e-300);
  diagonal(last) = full_left(last, last) / w_last;

  return {diagonal, subdiagonal};
}

template <typename Model>
Eigen::VectorXd HardWallMixin<Model>::count_eigenvals_sturm() {
  auto [diag, subdiag] = build_threediag_matrix_vectors();
  
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver;
  solver.computeFromTridiagonal(diag, subdiag, Eigen::EigenvaluesOnly);

  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("HardWallMixin: Failed to compute eigenvalues (solver did not converge).");
  }
  
  return solver.eigenvalues();
}


template <typename Model>
void HardWallMixin<Model>::initialise_flux_art() {
  auto net_step = model().net_step;
  const auto& net = model().net;
  auto total_net_nodes = model().total_net_nodes;
  for (size_t i = 0; i < total_net_nodes - 1; i++) {
    auto num = std::exp(static_cast<double>(model().metric_deformation(static_cast<double>(net[i].z) + 0.5 * net_step))) * model().kinetic_function_art(static_cast<double>(net[i].z) + 0.5 * net_step);
    auto den = net_step * (static_cast<double>(net[i].z) + 0.5 * net_step);
    flux_coeffs.push_back(num / den);
  }
}

template <typename Model>
void HardWallMixin<Model>::initialise_weights_art() {
  auto net_step = model().net_step;
  const auto& net = model().net;
  auto total_net_nodes = model().total_net_nodes;
  for (size_t i = 0; i < total_net_nodes - 1; i++) {
    auto num = std::exp(static_cast<double>(model().metric_deformation(static_cast<double>(net[i + 1].z)))) * model().kinetic_function_art(static_cast<double>(net[i + 1].z));
    auto den = static_cast<double>(net[i + 1].z);
    weights.push_back(num / den);
  }
}

template <typename Model>
void HardWallMixin<Model>::initialise_mass_addition_art() {
  auto net_step = model().net_step;
  const auto& net = model().net;
  auto total_net_nodes = model().total_net_nodes;
  auto meson = model().heavy_meson.value();
  for (size_t i = 0; i < total_net_nodes - 1; i++) {
    auto num = std::exp(3 * static_cast<double>(model().metric_deformation(static_cast<double>(net[i + 1].z)))) * model().kinetic_function_art(static_cast<double>(net[i + 1].z));
    auto den = std::pow(static_cast<double>(net[i + 1].z), 3);
    auto addition = meson.potential_addition();
    mass_additions.push_back((num * addition / den));
  }
}

template <typename Model>
void HardWallMixin<Model>::initialise_system_art() {
  flux_coeffs.clear();
  weights.clear();
  mass_additions.clear();
  model().inintialise_net();
  model().dilaton_setter();
  initialise_flux_art();
  initialise_weights_art();
  initialise_mass_addition_art();
}


template <typename Model>
void HardWallMixin<Model>::initialise_ext_dilaton() {
  if (dilaton_setted_sturm) {
    return;
  }
  auto epsilon = model().epsilon;
  using ValueType = typename Model::value_type;
  using std::sqrt;
  std::vector<ValueType> ext_net;
  for (size_t i = 0; i < 2 * model().total_net_nodes - 1; i++) {
    double current_z = epsilon + i * model().net_step / 2;
    ValueType z_val = static_cast<ValueType>((current_z));
    ext_net.emplace_back(z_val);
  }

  std::vector<ValueType> dilaton_ders;
  for (auto& z : ext_net) {
    auto derivs = autodiff::derivatives(model().metric_deformation,  autodiff::along(1.0), autodiff::at(z));

    double md_fd = derivs[1];
    double md_sd = derivs[2];
    double md_td = derivs[3];

    auto dilaton_first_der_raw = 6 * (md_fd * md_fd - md_sd - 2 * md_fd / z);
    if (dilaton_first_der_raw < 0) {
      throw std::runtime_error("Подкоренное выражение производной дилатона < 0, данный анзац не подходит");
    }
    auto dilaton_first_der = sqrt(dilaton_first_der_raw);
    auto dilaton_second_der = 3 * (2 * md_fd * md_sd - md_td - 2 * md_sd / z + 2 * md_fd / (z * z)) / dilaton_first_der;

    dilaton_ders.push_back(dilaton_first_der); 
  }

  dilaton_ext_vals.push_back(0);
  for (size_t i = 0; i < dilaton_ders.size() - 1; i++) {
    ValueType new_val = dilaton_ext_vals.back() + dilaton_ders[i] * model().net_step / 2;
    dilaton_ext_vals.push_back(static_cast<double>(new_val));
  }
  std::for_each(dilaton_ext_vals.begin(), dilaton_ext_vals.end(), [this](double& z) {z *= model().k_sw_impact;});
  dilaton_setted_sturm = true;
}

template <typename Model>
double HardWallMixin<Model>::auxularity_function_sturm(double z) {
  auto epsilon = model().epsilon;
  double id = static_cast<size_t>(std::llround((z - epsilon) / (model().net_step / 2)));
  auto heavy_meson = model().heavy_meson;
  using std::tanh;
  double dilat = static_cast<double>(dilaton_ext_vals[id]);
  double correction = heavy_meson.value().M * z + tanh(1.0 / (heavy_meson.value().M * z) - heavy_meson.value().dilaton_k / heavy_meson.value().sqrt_gamma);
  return dilat + correction;
}

template <typename Model>
double HardWallMixin<Model>::kinetic_function_sturm(double z) {
  using std::exp;
  return exp(-(static_cast<double>(model().metric_deformation(z)) + static_cast<double>(auxularity_function_sturm(z))));
}