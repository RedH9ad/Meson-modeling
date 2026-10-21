#pragma once
#include "hard_wall_mixin.hpp"
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
    auto num = std::exp(static_cast<double>(model().metric_deformation(static_cast<double>(net[i].z) + 0.5 * net_step))) * model().kinetic_function(static_cast<double>(net[i].z) + 0.5 * net_step);
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
    auto num = std::exp(static_cast<double>(model().metric_deformation(static_cast<double>(net[i + 1].z)))) * model().kinetic_function(static_cast<double>(net[i + 1].z));
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
    auto num = std::exp(3 * static_cast<double>(model().metric_deformation(static_cast<double>(net[i + 1].z)))) * model().kinetic_function(static_cast<double>(net[i + 1].z));
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
  model().dilaton_setter();
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
    diagonal(i) = full_left(i, i) / weight_m(i, i);
    subdiagonal(i) = full_left(i, i + 1) / std::sqrt(weight_m(i, i) * weight_m(i + 1, i + 1));
  }
  size_t last = diagonal.size() - 1;
  diagonal(last) = full_left(last, last) / weight_m(last, last);

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
