#include "net_node.hpp"
#include <cmath>
#include <csignal>
#include <cstddef>
#include <functional>
#include <vector>
#include <stdexcept>
#include <autodiff/forward/dual.hpp>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#pragma once

template<class T>
class Mesons_Model {
 private:
  double net_step = 0.0;
  double z_max = 0.0;
  double k_sw_impact = 0.0;
  size_t total_net_nodes = 0;
  std::vector<NetNode<T>> net = {};

  std::function<T(T)> metric_deformation;
  std::function<T(T)> metric_scale_factor;

 public:
  Mesons_Model() = default;
  Mesons_Model(size_t, double, double, std::function<T(T)>, std::function<T(T)>);
  Mesons_Model(size_t, double, std::function<T(T)>, std::function<T(T)>);

  T Dilaton(T) const;
  void inintialise_net();

  T count_w(T) const;
  T w_first_derivative_num(NetNode<T>&) const;
  T w_second_derivative_num(NetNode<T>&) const;

  void set_potential_num(NetNode<T>&);

  void make_net_num();
  void make_net_a();

  Eigen::MatrixXd build_matrix();
  //C индексом l просто для удобства всяких проверок
  Eigen::MatrixXd build_matrix_l();

  Eigen::VectorXd get_meson_mass_a();
  Eigen::VectorXd get_meson_mass_num();
};

template <class T>
Mesons_Model<T>::Mesons_Model(size_t nn, double zm, double k, std::function<T(T)> md, std::function<T(T)> msf) : total_net_nodes(nn), z_max(zm), k_sw_impact(k), metric_deformation(md), metric_scale_factor(msf) {
  net = {};
  net_step = z_max / total_net_nodes;
}

template <class T>
Mesons_Model<T>::Mesons_Model(size_t nn, double k, std::function<T(T)> md, std::function<T(T)> msf) : total_net_nodes(nn), k_sw_impact(k), metric_deformation(md), metric_scale_factor(msf) {
  z_max = 10 / k;
  net = {};
  net_step = z_max / total_net_nodes;
}

//Рассчет Ф(z) = k^2*z^2 + A(z)
template <class T>
T Mesons_Model<T>::Dilaton(T z) const {
  return k_sw_impact * k_sw_impact * z * z + metric_deformation(z);
}

//Инициализация сетки с костылем для того чтобы избегать занулений
template <class T>
void Mesons_Model<T>::inintialise_net() {
  net.clear();
  double current_z = 0;
  double epsilon = 1e-7;
  current_z = epsilon;
  for (size_t i = 0; i < total_net_nodes; i++) {
    T w = count_w(current_z);
    net.emplace_back(current_z, i, w);
    current_z += net_step;
  }
}

//Рассчет w(z) = Ф(z) - ln(a(z))
template <class T>
T Mesons_Model<T>::count_w(T z) const {
  return Dilaton(z) - log(metric_scale_factor(z));
}

//Численный рассчет первой и второй производных соответственно.
//Не делается для первого и последнего узлов сетки, так как их инициализируют граничные условия
template <class T>
T Mesons_Model<T>::w_first_derivative_num(NetNode<T>& node) const {
  if (node.node_id == 0 || node.node_id == total_net_nodes - 1) {
    throw std::runtime_error("Для первой и последней ячеек не считаем первую");
  }
  auto next_w = net[node.node_id + 1].w;
  auto prev_w = net[node.node_id - 1].w;
  return (next_w - prev_w) / (2 * net_step);
}

template <class T>
T Mesons_Model<T>::w_second_derivative_num(NetNode<T>& node) const {
  if (node.node_id == 0 || node.node_id == total_net_nodes - 1) {
    throw std::runtime_error("Для первой и последней ячеек не считаем первую");
  }
  auto next_w = net[node.node_id + 1].w;
  auto prev_w = net[node.node_id - 1].w;
  return (next_w - 2 * node.w + prev_w) / (net_step * net_step);
}

//Функция рассчета потенциала U(z)
template <class T>
void Mesons_Model<T>::set_potential_num(NetNode<T>& node) {
  double first_dev = 0.0;
  double second_dev = 0.0;

  if constexpr (std::is_same_v<T, double>) {
      first_dev = w_first_derivative_num(node);
      second_dev = w_second_derivative_num(node);

  } else {
    first_dev = autodiff::val(autodiff::val(w_first_derivative_num(node)));
    second_dev = autodiff::val(autodiff::val(w_second_derivative_num(node)));
  }

  node.potential = 0.25 * first_dev * first_dev - 0.5 * second_dev;
}

//Функция выстраивающая сеть(инициализация + рассчет потенциалов) для даблов 
template <class T>
void Mesons_Model<T>::make_net_num() {
  inintialise_net();
  for (NetNode<T>& node : net) {
    if (node.node_id == 0 || node.node_id == total_net_nodes - 1) {
      continue;
    }
    set_potential_num(node);
  }
}

//Сетка через автодиф
template <class T>
void Mesons_Model<T>::make_net_a() {
  inintialise_net();
  for (NetNode<T>& node : net) {
    if (node.node_id == 0 || node.node_id == total_net_nodes - 1) {
      continue;
    }
    auto w_func = [this](T z_arg) { return count_w(z_arg); };
    auto derivs = autodiff::derivatives(w_func, autodiff::wrt(node.z), autodiff::at(node.z));

    double w_val = derivs[0];
    double w_fd  = derivs[1];
    double w_sd  = derivs[2];

    node.w = static_cast<T>(w_val);

    node.potential = 0.25 * w_fd * w_fd - 0.5 * w_sd;
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
    if (i - 1 > 0) {
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
    if (i - 1 > 0) {
      Ham(i, i - 1) = -1 / squared_step;
    }
    if (i < total_net_nodes - 3) {
      Ham(i, i + 1) = -1 / squared_step;
    }
  }
  return Ham;
}

template <class T>
Eigen::VectorXd Mesons_Model<T>::get_meson_mass_a() {
  make_net_a();
  auto Ham = build_matrix();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Ham);
    
  if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Eigen не смог диагонализировать матрицу Гамильтониана!");
  }
  Eigen::VectorXd eigenvalues = solver.eigenvalues();
  return  eigenvalues;
}

template <class T>
Eigen::VectorXd Mesons_Model<T>::get_meson_mass_num() {
  make_net_num();
  auto Ham = build_matrix();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Ham);
    
  if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Eigen не смог диагонализировать матрицу Гамильтониана!");
  }
  Eigen::VectorXd eigenvalues = solver.eigenvalues();
  return  eigenvalues;
}