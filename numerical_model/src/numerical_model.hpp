#include "net_node.hpp"
#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <functional>
#include <vector>
#include <stdexcept>
#include <autodiff/forward/real.hpp>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#pragma once

constexpr double epsilon = 1e-10;

template<class T>
class Mesons_Model {
 private:
  double net_step = 0.0;
  double z_max = 0.0;
  double k_sw_impact = 0.0;
  double L = 1.0;

  size_t total_net_nodes = 0;
  std::vector<NetNode<T>> net = {};

  std::function<T(T)> metric_deformation;

 public:
  Mesons_Model() = default;
  Mesons_Model(size_t, double, double, std::function<T(T)>);
  Mesons_Model(size_t, double, double, double, std::function<T(T)>);

  Mesons_Model(size_t, double, std::function<T(T)>);

  void dilaton_setter();
  void dilaton_setter_num();

  T dilaton(T) const;
  void inintialise_net();
  T warp_factor_conv(T) const;

  void make_net_rho();
  void make_net_rho_num();

  void make_net_atwo();
  void make_net_atwo_num();

  Eigen::MatrixXd build_matrix();
  //C индексом l просто для удобства всяких проверок
  Eigen::MatrixXd build_matrix_l();

  Eigen::VectorXd get_rho_meson_mass_a();
  Eigen::VectorXd get_atwo_meson_mass_a();

  Eigen::VectorXd get_rho_meson_mass_num();
  Eigen::VectorXd get_atwo_meson_mass_num();

  static std::pair<std::vector<T>, std::vector<T>> first_and_second_derivative_num_counter(const std::vector<T>&, double);

};

template<class T>
std::pair<std::vector<T>, std::vector<T>> Mesons_Model<T>::first_and_second_derivative_num_counter(const std::vector<T>& scale, double net_step) {
  std::vector<T> first_derivatives(scale.size());
  std::vector<T> second_derivatives(scale.size());
  size_t net_size = scale.size();

  for (size_t i = 0; i < scale.size(); i++) {
    T first_der = 0;
    T second_der = 0;
    //Краевой случай первой ячейки
    if (i == 0) {
      first_der = (-3 * scale[0] + 4 * scale[1] - scale[2]) / (2 * net_step);
      second_der = (2 * scale[0] - 5 * scale[1] + 4 * scale[2] - scale[3]) / (net_step * net_step);
    }
    //Краевой случай последней ячейки
    else if (i == net_size - 1) {
      first_der = (3 * scale[net_size - 1] - 4 * scale[net_size - 2] + scale[net_size - 3]) / (2 * net_step);
      second_der = (2 * scale[net_size - 1] - 5 * scale[net_size - 2] + 4 * scale[net_size - 3] - scale[net_size - 4]) / (net_step * net_step);
    }
    //Обший случай
    else {
      first_der = (scale[i + 1] - scale[i - 1]) / (2 * net_step);
      second_der = (scale[i + 1] - 2 * scale[i] + scale[i - 1]) / (net_step * net_step);
    }

    first_derivatives[i] = first_der;
    second_derivatives[i] = second_der;    
  }
  return {first_derivatives, second_derivatives};
}

template <class T>
Mesons_Model<T>::Mesons_Model(size_t nn, double zm, double k, std::function<T(T)> md) : total_net_nodes(nn), z_max(zm), k_sw_impact(k), metric_deformation(md) {
  net = {};
  net_step = z_max / total_net_nodes;
  L = 1.0;
}

template <class T>
Mesons_Model<T>::Mesons_Model(size_t nn, double zm, double k, double L, std::function<T(T)> md) : total_net_nodes(nn), z_max(zm), k_sw_impact(k), L(L), metric_deformation(md) {
  net = {};
  net_step = z_max / total_net_nodes;
}

template <class T>
Mesons_Model<T>::Mesons_Model(size_t nn, double k, std::function<T(T)> md) : total_net_nodes(nn), k_sw_impact(k), metric_deformation(md) {
  z_max = 10 / k;
  net = {};
  net_step = z_max / total_net_nodes;
  L = 1.0;
}

//Восстановление вектора дилатонов из его производной, в новой нотации Ф - c домножением на кожффициент
template <class T>
void Mesons_Model<T>::dilaton_setter() {
  std::vector<T> dilaton_ders;

  //Рассчет производной дилатона
  for (NetNode<T>& node : net) {
    auto derivs = autodiff::derivatives(metric_deformation, autodiff::along(1.0), autodiff::at(node.z));

    double md_fd = derivs[1];
    double md_sd = derivs[2];
    double md_td = derivs[3];

    auto dilaton_first_der_raw = 6 * (md_fd * md_fd - md_sd - 2 * md_fd / node.z);
    if (dilaton_first_der_raw < 0) {
      throw std::runtime_error("Подкоренное выражение производной дилатона < 0, данный анзац не подходит");
    }
    auto dilaton_first_der = sqrt(dilaton_first_der_raw);
    auto dilaton_second_der = 3 * (2 * md_fd * md_sd - md_td - 2 * md_sd / node.z + 2 * md_fd / (node.z * node.z)) / dilaton_first_der;

    node.dilaton_first_der = k_sw_impact * dilaton_first_der.val();
    node.dilaton_secod_der = k_sw_impact * dilaton_second_der.val();

    dilaton_ders.push_back(dilaton_first_der); 
  }

  net[0].dilaton = 0;
  for (size_t i = 0; i < dilaton_ders.size() - 1; i++) {
    T new_val = net[i].dilaton + dilaton_ders[i] * net_step;

    net[i + 1].dilaton = new_val.val();
  }
  std::for_each(net.begin(), net.end(), [this](NetNode<T>& node) {node.dilaton *= k_sw_impact;});
}

template <class T> 
T Mesons_Model<T>::warp_factor_conv(T z) const {
  using std::log;
  return metric_deformation(z) - log(z) + std::log(L);
}


//Все в численном расчете производных реализуется через центральные разности или их краевые частные случаи
template<class T>
void Mesons_Model<T>::dilaton_setter_num() {
  std::vector<T> metric_deformation_vals(net.size());
  std::vector<T> dilaton_ders;

  for (NetNode<T>& node : net) {
    auto md_value = metric_deformation(node.z);
    metric_deformation_vals[node.node_id] = md_value;
  }

  auto [md_first_ders, md_second_ders] = first_and_second_derivative_num_counter(metric_deformation_vals, net_step);
  for (NetNode<T>& node : net) {
    using std::sqrt;
    auto dilaton_first_der = sqrt(6 * (md_first_ders[node.node_id] * md_first_ders[node.node_id] - md_second_ders[node.node_id] - 2 * md_first_ders[node.node_id] / node.z));
    dilaton_ders.push_back(dilaton_first_der);
    node.dilaton_first_der = k_sw_impact * dilaton_first_der;
  }
 
  net[0].dilaton = 0;
  for (size_t i = 0; i < dilaton_ders.size() - 1; i++) {
    T new_val = net[i].dilaton + dilaton_ders[i] * net_step;

    if (new_val < 0) {
      throw std::runtime_error("Данный анзац не подходит");
    }
    net[i + 1].dilaton = new_val;
  }
  std::for_each(net.begin(), net.end(), [this](NetNode<T>& node) {node.dilaton *= k_sw_impact;});
  
  for (NetNode<T>& node : net) {
    if (node.node_id == 0) {
      node.dilaton_secod_der = (-3 * net[0].dilaton_first_der + 4 * net[1].dilaton_first_der - net[2].dilaton_first_der) / (2 * net_step);
    }
    else if (node.node_id == net.size() - 1) {
      node.dilaton_secod_der = (3 * net[net.size() - 1].dilaton_first_der - 4 * net[net.size() - 2].dilaton_first_der + net[net.size() - 3].dilaton_first_der) / (2 * net_step);
    }
    else {
      node.dilaton_secod_der = (net[node.node_id + 1].dilaton_first_der - net[node.node_id - 1].dilaton_first_der) / (2 * net_step);
    }
  }
}

//Инициализация сетки с костылем для того чтобы избегать занулений
template <class T>
void Mesons_Model<T>::inintialise_net() {
  net.clear();
  for (size_t i = 0; i < total_net_nodes; i++) {
    double current_z = epsilon + i * net_step;
    T z_val = static_cast<T>((current_z));
    net.emplace_back(z_val, i);
  }
}

//Сетка через автодиф для ро мезонов
template <class T>
void Mesons_Model<T>::make_net_rho() {
  inintialise_net();
  dilaton_setter();

  for (NetNode<T>& node : net) {
    if (node.node_id == 0 || node.node_id == total_net_nodes - 1) {
      continue;
    }
    auto A_func = [this](T z_arg) { return warp_factor_conv(z_arg); };
    auto derivs = autodiff::derivatives(A_func, autodiff::along(1.0), autodiff::at(node.z));

    double A_fd  = derivs[1];
    double A_sd  = derivs[2];

    double B_rho_fd = node.dilaton_first_der - A_fd;
    double B_rho_sd = node.dilaton_secod_der - A_sd;

    node.potential = 0.25 * B_rho_fd * B_rho_fd - 0.5 * B_rho_sd;
  }
}

template <class T>
void Mesons_Model<T>::make_net_rho_num() {
  inintialise_net();
  dilaton_setter_num();
  std::vector<T> warp(net.size());
  for (NetNode<T>& node : net) {
    warp[node.node_id] = warp_factor_conv(node.z);
  }

  auto [warp_first_ders, warp_second_ders] = first_and_second_derivative_num_counter(warp, net_step);

  for (NetNode<T>& node : net) {
    if (node.node_id == 0 || node.node_id == total_net_nodes - 1) {
      continue;
    }

    double A_fd  = warp_first_ders[node.node_id];
    double A_sd  = warp_second_ders[node.node_id];

    double B_rho_fd = node.dilaton_first_der - A_fd;
    double B_rho_sd = node.dilaton_secod_der - A_sd;

    node.potential = 0.25 * B_rho_fd * B_rho_fd - 0.5 * B_rho_sd;
  }
}

//Сетка через автодиф для а2 мезонов
template <class T>
void Mesons_Model<T>::make_net_atwo() {
  inintialise_net();
  dilaton_setter();

  for (NetNode<T>& node : net) {
    if (node.node_id == 0 || node.node_id == total_net_nodes - 1) {
      continue;
    }
    auto A_func = [this](T z_arg) { return warp_factor_conv(z_arg); };
    auto derivs = autodiff::derivatives(A_func, autodiff::along(1.0), autodiff::at(node.z));

    double A_fd  = derivs[1];
    double A_sd  = derivs[2];

    double B_atwo_fd = node.dilaton_first_der - 3 * A_fd;
    double B_atwo_sd = node.dilaton_secod_der - 3 * A_sd;

    node.potential = 0.25 * B_atwo_fd * B_atwo_fd - 0.5 * B_atwo_sd;
  }
}

template <class T>
void Mesons_Model<T>::make_net_atwo_num() {
  inintialise_net();
  dilaton_setter_num();
  std::vector<T> warp(net.size());
  for (NetNode<T>& node : net) {
    warp[node.node_id] = warp_factor_conv(node.z);
  }

  auto [warp_first_ders, warp_second_ders] = first_and_second_derivative_num_counter(warp, net_step);

  for (NetNode<T>& node : net) {
    if (node.node_id == 0 || node.node_id == total_net_nodes - 1) {
      continue;
    }

    double A_fd  = warp_first_ders[node.node_id];
    double A_sd  = warp_second_ders[node.node_id];

    double B_rho_fd = node.dilaton_first_der - 3 * A_fd;
    double B_rho_sd = node.dilaton_secod_der - 3 * A_sd;

    node.potential = 0.25 * B_rho_fd * B_rho_fd - 0.5 * B_rho_sd;
  }
}

//Функция собирающая матрицу оператора
template <class T>
Eigen::MatrixXd Mesons_Model<T>::build_matrix_l() {
  Eigen::MatrixXd Ham = Eigen::MatrixXd::Zero(total_net_nodes, total_net_nodes);
  double squared_step = net_step * net_step;

  Ham(0, 0) = 1;
  Ham(total_net_nodes - 1, total_net_nodes - 1) = 1;

  for (int64_t i = 1; i < total_net_nodes - 1; i++) {
    Ham(i, i) = 2 / squared_step + net[i].potential;
    if (i > 0) {
      Ham(i, i - 1) = -1 / squared_step;
    }
    if (i < total_net_nodes - 1) {
      Ham(i, i + 1) = -1 / squared_step;
    }
  }
  return Ham;
}

template <class T>
Eigen::MatrixXd Mesons_Model<T>::build_matrix() {
  Eigen::MatrixXd Ham = Eigen::MatrixXd::Zero(total_net_nodes - 2, total_net_nodes - 2);
  double squared_step = net_step * net_step;

  for (int64_t i = 0; i < total_net_nodes - 2; i++) {
    Ham(i, i) = 2 / squared_step + net[i + 1].potential;
    if (i > 0) {
      Ham(i, i - 1) = -1 / squared_step;
    }
    if (i < total_net_nodes - 3) {
      Ham(i, i + 1) = -1 / squared_step;
    }
  }
  return Ham;
}

template <class T>
Eigen::VectorXd Mesons_Model<T>::get_rho_meson_mass_a() {
  make_net_rho();
  auto Ham = build_matrix();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Ham);
    
  if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Eigen не смог диагонализировать матрицу Гамильтониана!");
  }
  Eigen::VectorXd eigenvalues = solver.eigenvalues();
  return  eigenvalues;
}

template <class T>
Eigen::VectorXd Mesons_Model<T>::get_rho_meson_mass_num() {
  make_net_rho_num();
  auto Ham = build_matrix();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Ham);
    
  if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Eigen не смог диагонализировать матрицу Гамильтониана!");
  }
  Eigen::VectorXd eigenvalues = solver.eigenvalues();
  return  eigenvalues;
}

template <class T>
Eigen::VectorXd Mesons_Model<T>::get_atwo_meson_mass_num() {
  make_net_atwo_num();
  auto Ham = build_matrix();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Ham);
    
  if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Eigen не смог диагонализировать матрицу Гамильтониана!");
  }
  Eigen::VectorXd eigenvalues = solver.eigenvalues();
  return  eigenvalues;
}