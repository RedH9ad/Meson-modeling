#include "net_node.hpp"
#include "base_heavy_mesons_config.hpp"
#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <functional>
#include <optional>
#include <vector>
#include <stdexcept>
#include <autodiff/forward/real.hpp>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>


#pragma once
//В статье L принято единицей

constexpr double epsilon = 1e-10;

//Параметры для расчета альфа и k для параметризации дилатона из статьи
namespace Parametrs_For_Dilaton_Counter {
  constexpr double A_ALPHA = 0.054;
  constexpr double B_ALPHA = 0.8484;
  constexpr double C_ALPHA = 0.4233;

  constexpr double A_K = 15.2085;
  constexpr double B_K = 14.8082;
  constexpr double C_K = 0.0524;
}

template<class T>
class Mesons_Model {
 public:
  double net_step = 0.0;
  double z_max = 0.0;
  double k_sw_impact = 0.0;
  double L = 1.0;

  //constants
  double alpha;
  double beta;
  double c;
  double k_1;
  double k_2;
  double gfive_squared = 4 * M_PI * M_PI;

  std::optional<HeavyMesonConfig> heavy_meson;

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

  double dilaton(T) const;
  void inintialise_net();
  T warp_factor_conv(T) const;

  void initialize_kiral_sector();
  

  void make_net_rho();
  void make_net_rho_num();

  void make_net_atwo();
  void make_net_atwo_num();

  void make_net_aone();
  void make_net_aone_num();

  void make_net_fzero();
  void make_net_fzero_num();

  Eigen::MatrixXd build_matrix();

  Eigen::MatrixXd count_eigenvals();
  //C индексом l просто для удобства всяких проверок
  Eigen::MatrixXd build_matrix_l();

  Eigen::VectorXd get_rho_meson_mass_a();
  Eigen::VectorXd get_atwo_meson_mass_a();
  Eigen::VectorXd get_aone_meson_mass_a();
  Eigen::VectorXd get_fzero_meson_mass_a();

  Eigen::VectorXd get_rho_meson_mass_num();
  Eigen::VectorXd get_atwo_meson_mass_num();
  Eigen::VectorXd get_aone_meson_mass_num();
  Eigen::VectorXd get_fzero_meson_mass_num();

  static std::pair<std::vector<T>, std::vector<T>> first_and_second_derivative_num_counter(const std::vector<T>&, double);

  //Функции для рассчета всего по статье по каким-то причинам не работают, хотя значения альфа и к из самой статьи вполне выполняют пайплайн
  // static double dilaton_alpha(double);
  // static double dilaton_k(double);

  T Dilaton_art(T);
  T Gaude_kinetic_function(T);
  T Auxiliary_function(T);
  T article_potential_brick(T);

  T Auxiliary_function_prod(T);
  T article_potential_brick_prod(T);

  void make_net_exotic();
  void make_net_exotic_num();
  Eigen::VectorXd get_exotic_meson_mass_a();
  Eigen::VectorXd get_exotic_meson_mass_num();

  void make_net_exotic_prod();
  Eigen::VectorXd get_exotic_meson_mass_prod();
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

//Инициализация кирального сектора происходит после задания дилатона
//В дабл кастится вся хуйня потому что киральный сектор позиционируется как дабл
template <class T>
void Mesons_Model<T>::initialize_kiral_sector() {
  std::vector<double> first_factor(net.size());
  std::vector<double> second_factor(net.size());

  for (NetNode<T>& node : net) {
    using std::exp, std::pow;
    first_factor[node.node_id] = static_cast<double>(exp(3 * warp_factor_conv(node.z) - node.dilaton));
    second_factor[node.node_id] = static_cast<double>(exp(5 * warp_factor_conv(node.z) - node.dilaton));

    //Для начала инициализируем v(z) в соответствие с ассимптотическим предположением
    node.kiral_sector = static_cast<double>(alpha * node.z + beta * pow(node.z, 3));
  }

  auto squared_step = net_step * net_step;
  Eigen::VectorXd deviation_func = Eigen::VectorXd::Zero(total_net_nodes);
  Eigen::MatrixXd jacobian(total_net_nodes, total_net_nodes);
  for (size_t iters = 0; iters < 20; iters++) {
    jacobian.setZero();
    jacobian(0, 0) = 1.0;

    deviation_func(0) = net[0].kiral_sector.value() - static_cast<double>((alpha * net[0].z + beta * pow(net[0].z, 3)));

    for (size_t i = 1; i < total_net_nodes - 1; i++) {
      double first_factor_plus = (first_factor[i] + first_factor[i + 1]) / 2.0;
      double first_factor_minus = (first_factor[i - 1] + first_factor[i]) / 2.0;
      deviation_func(i) = ((first_factor_plus * (net[i + 1].kiral_sector.value() - net[i].kiral_sector.value()) - first_factor_minus * (net[i].kiral_sector.value() - net[i - 1].kiral_sector.value())) / squared_step
                                  - second_factor[i] * (-3 * net[i].kiral_sector.value() - k_1 * net[i].kiral_sector.value() * net[i].kiral_sector.value() - k_2 * pow(net[i].kiral_sector.value(), 3)));
      jacobian(i, i - 1) = first_factor_minus / squared_step;
      jacobian(i, i + 1) = first_factor_plus / squared_step;
      jacobian(i, i) = -(first_factor_plus + first_factor_minus) / squared_step - second_factor[i] * (-3 - 2 * k_1 * net[i].kiral_sector.value() - 3 * k_2 * net[i].kiral_sector.value() * net[i].kiral_sector.value());
    }
    size_t last = total_net_nodes - 1;
    deviation_func(last) = 3 * net[last].kiral_sector.value() - 4 * net[last - 1].kiral_sector.value() + net[last - 2].kiral_sector.value();
    jacobian(last, last) = 3.0;
    jacobian(last, last-1) = -4.0;
    jacobian(last, last-2) = 1.0;

    // Решаем плотную систему уравнений
    Eigen::VectorXd delta = jacobian.partialPivLu().solve(-deviation_func);
    for (size_t i = 0; i < total_net_nodes; i++) {
      net[i].kiral_sector = delta(i) + net[i].kiral_sector.value();
    }

    if (delta.norm() < 1e-8) break;
  }
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

template<class T>
double Mesons_Model<T>::dilaton(T z) const {
  double id = static_cast<double>((z - epsilon) / net_step);
  return static_cast<double> (net[id].dilaton);  
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

template <class T>
void Mesons_Model<T>::make_net_aone() {
  make_net_rho();
  initialize_kiral_sector();
  for (NetNode<T>& node : net) {
    using std::exp;
    node.potential += gfive_squared * node.kiral_sector.value() * node.kiral_sector.value() * exp(2 * warp_factor_conv(node.z));
  }
}

template <class T>
void Mesons_Model<T>::make_net_aone_num() {
  make_net_rho_num();
  initialize_kiral_sector();
  for (NetNode<T>& node : net) {
    using std::exp;
    node.potential += (gfive_squared * node.kiral_sector.value() * node.kiral_sector.value() * exp(2 * warp_factor_conv(node.z)));
  }
}

template <class T>
void Mesons_Model<T>::make_net_fzero() {
  make_net_atwo();
  initialize_kiral_sector();
  for (NetNode<T>& node : net) {
    using std::exp;
    node.potential += (exp(2 * warp_factor_conv(node.z)) * (-3 - 2 * k_1 * node.kiral_sector.value() - 3 * k_2 * node.kiral_sector.value() * node.kiral_sector.value()));
  }
}

template <class T>
void Mesons_Model<T>::make_net_fzero_num() {
  make_net_atwo_num();
  initialize_kiral_sector();
  for (NetNode<T>& node : net) {
    using std::exp;
    node.potential += (exp(2 * warp_factor_conv(node.z)) * (-3 - 2 * k_1 * node.kiral_sector.value() - 3 * k_2 * node.kiral_sector.value() * node.kiral_sector.value()));
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
Eigen::MatrixXd Mesons_Model<T>::count_eigenvals() {
  auto Ham = build_matrix();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Ham);
  if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Eigen не смог диагонализировать матрицу Гамильтониана!");
  }
  Eigen::VectorXd eigenvalues = solver.eigenvalues();
  return  eigenvalues;
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
Eigen::VectorXd Mesons_Model<T>::get_atwo_meson_mass_a() {
  make_net_atwo();
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

template <class T>
Eigen::VectorXd Mesons_Model<T>::get_aone_meson_mass_a() {
  make_net_aone();
  auto Ham = build_matrix();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Ham);
    
  if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Eigen не смог диагонализировать матрицу Гамильтониана!");
  }
  Eigen::VectorXd eigenvalues = solver.eigenvalues();
  return  eigenvalues;
}

template <class T>
Eigen::VectorXd Mesons_Model<T>::get_aone_meson_mass_num() {
  make_net_aone_num();
  auto Ham = build_matrix();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Ham);
    
  if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Eigen не смог диагонализировать матрицу Гамильтониана!");
  }
  Eigen::VectorXd eigenvalues = solver.eigenvalues();
  return  eigenvalues;
}

template <class T>
Eigen::VectorXd Mesons_Model<T>::get_fzero_meson_mass_a() {
  make_net_fzero();
  auto Ham = build_matrix();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Ham);
    
  if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Eigen не смог диагонализировать матрицу Гамильтониана!");
  }
  Eigen::VectorXd eigenvalues = solver.eigenvalues();
  return  eigenvalues;
}

template <class T>
Eigen::VectorXd Mesons_Model<T>::get_fzero_meson_mass_num() {
  make_net_fzero_num();
  auto Ham = build_matrix();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Ham);
    
  if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Eigen не смог диагонализировать матрицу Гамильтониана!");
  }
  Eigen::VectorXd eigenvalues = solver.eigenvalues();
  return  eigenvalues;
}

//Все функции дальнейшего блока нужны для рассчета тяжелых и экзотических мезонов
//Там используется неквадратичный дилатон, введенный в статье и функция f(z) введенная там же
// template <class T>
// double Mesons_Model<T>::dilaton_alpha(double mass) {
//   using namespace Parametrs_For_Dilaton_Counter;
//   return A_ALPHA - B_ALPHA * std::exp(-C_ALPHA * mass * mass);
//   // return 0.54;
// }

// template <class T> 
// double Mesons_Model<T>::dilaton_k(double mass) {
//   using namespace Parametrs_For_Dilaton_Counter;
//   return A_K - B_K * std::exp(-C_K * mass * mass);
//   // return 1.8;
// }

template <class T>
T Mesons_Model<T>::Dilaton_art(T z) {
  if (!heavy_meson) {
    throw std::runtime_error("Масса тяжелого или экзотического мезона с которым должен работать класс не инициализирована");
  }
  using std::pow;
  T base = heavy_meson.value().dilaton_k * z;
  double exponent = 2.0 - heavy_meson.value().dilaton_alpha;
  return pow(base, exponent);
}

//B(z) из статьи
template <class T>
T Mesons_Model<T>::Auxiliary_function(T z) {
  using std::tanh;
  T dilaton = Dilaton_art(z);
  T correction = heavy_meson.value().M * z + tanh(1.0 / (heavy_meson.value().M * z) - heavy_meson.value().dilaton_k / heavy_meson.value().sqrt_gamma);
  return dilaton + correction;
}

template <class T>
T Mesons_Model<T>::Auxiliary_function_prod(T z) {
  using std::tanh;
  double dilat = dilaton(z);
  T correction = heavy_meson.value().M * z + tanh(1.0 / (heavy_meson.value().M * z) - heavy_meson.value().dilaton_k / heavy_meson.value().sqrt_gamma);
  return dilat + correction;
}

//f(z) из статьи
template <class T>
T Mesons_Model<T>::Gaude_kinetic_function(T z) {
  using std::exp;
  return exp(-(metric_deformation(z) + Auxiliary_function(z)));
}

template <class T>
T Mesons_Model<T>::article_potential_brick(T z) {
  using std::log;
  return static_cast<T>(0.5) * (log(static_cast<T>(L)) - Auxiliary_function(z) - log(z));
}

template <class T>
T Mesons_Model<T>::article_potential_brick_prod(T z) {
  using std::log;
  return static_cast<T>(0.5) * (log(static_cast<T>(L)) - Auxiliary_function_prod(z) - log(z));
}

template <class T>
void Mesons_Model<T>::make_net_exotic() {
  inintialise_net();
  for (NetNode<T>& node : net) {
    if (node.node_id == 0 || node.node_id == total_net_nodes - 1) {
      continue;
    }
    auto brick_func = [this](T z_arg) { return article_potential_brick(z_arg); };
    auto derivs = autodiff::derivatives(brick_func, autodiff::along(1.0), autodiff::at(node.z));
    auto brick_fd = derivs[1];
    auto brick_sd = derivs[2];
    node.potential = brick_fd * brick_fd + brick_sd + static_cast<double>(heavy_meson.value().potential_addition() / (node.z * node.z));
  }
}

template <class T>
void Mesons_Model<T>::make_net_exotic_num() {
  inintialise_net();
  std::vector<T> brick(net.size());
  for (NetNode<T>& node : net) {
    brick[node.node_id] = article_potential_brick(node.z);
  }

  auto [brick_first_ders, brick_second_ders] = first_and_second_derivative_num_counter(brick, net_step);

  for (NetNode<T>& node : net) {
    if (node.node_id == 0 || node.node_id == total_net_nodes - 1) {
      continue;
    }

    double brick_fd  = brick_first_ders[node.node_id];
    double brick_sd  = brick_second_ders[node.node_id];

    node.potential = brick_fd * brick_fd + brick_sd + static_cast<double>(heavy_meson.value().potential_addition() / (node.z * node.z));
  }
}

template <class T>
void Mesons_Model<T>::make_net_exotic_prod() {
  inintialise_net();
  dilaton_setter_num();
  std::vector<T> brick(net.size());
  for (NetNode<T>& node : net) {
    brick[node.node_id] = article_potential_brick_prod(node.z);
  }

  auto [brick_first_ders, brick_second_ders] = first_and_second_derivative_num_counter(brick, net_step);

  for (NetNode<T>& node : net) {
    if (node.node_id == 0 || node.node_id == total_net_nodes - 1) {
      continue;
    }

    double brick_fd  = brick_first_ders[node.node_id];
    double brick_sd  = brick_second_ders[node.node_id];

    node.potential = brick_fd * brick_fd + brick_sd + static_cast<double>(heavy_meson.value().potential_addition() / (node.z * node.z));
  }
}

template <class T>
Eigen::VectorXd Mesons_Model<T>::get_exotic_meson_mass_a() {
  make_net_exotic();
  auto Ham = build_matrix();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Ham);
    
  if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Eigen не смог диагонализировать матрицу Гамильтониана!");
  }
  Eigen::VectorXd eigenvalues = solver.eigenvalues();
  return  eigenvalues;
}

template <class T>
Eigen::VectorXd Mesons_Model<T>::get_exotic_meson_mass_num() {
  make_net_exotic_num();
  auto Ham = build_matrix();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Ham);
    
  if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Eigen не смог диагонализировать матрицу Гамильтониана!");
  }
  Eigen::VectorXd eigenvalues = solver.eigenvalues();
  return  eigenvalues;
}

template <class T>
Eigen::VectorXd Mesons_Model<T>::get_exotic_meson_mass_prod() {
  make_net_exotic_prod();
  auto Ham = build_matrix();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Ham);
    
  if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Eigen не смог диагонализировать матрицу Гамильтониана!");
  }
  Eigen::VectorXd eigenvalues = solver.eigenvalues();
  return  eigenvalues;
}