#pragma once
#include "base_heavy_mesons_config.hpp"
#include "heavy_meson_processor.hpp"
#include "numerical_model.hpp"
#include <chrono>
#include <functional>
#include <stdexcept>
#include <vector>
#include "chrono"
#include <thread>
#include <Spectra/SymEigsSolver.h>

template <class T>
Heavy_Mesons_Processor<T>::Heavy_Mesons_Processor(Mesons_Model<T>& model, const std::vector<HeavyMesonConfig>& mesons) : model(model), mesons(mesons) {
  if (mesons.empty()) {
    throw std::runtime_error("Вектор тяжелых мезонов с которыми предполагается работать пустой");
  }
  meson_count = mesons.size();
}

template <class T> 
Eigen::VectorXd Heavy_Mesons_Processor<T>::get_heavy_meson_mass_num() {
  auto matrixies = make_matrixies_ready_num();
  Eigen::VectorXd result(3 * meson_count);
  std::vector<std::thread> threads;
  for (size_t id = 0; id < meson_count; id++) {
    threads.emplace_back(&Heavy_Mesons_Processor<T>::process_eigenvals, this, id, std::cref(matrixies[id]), std::ref(result));
  }

  for (auto& thread : threads) {
    thread.join();
  }
  return result;
}

template <class T> 
Eigen::VectorXd Heavy_Mesons_Processor<T>::get_heavy_meson_mass_anz() {
  auto matrixies = make_matrixies_ready_anz();
  Eigen::VectorXd result(3 * meson_count);
  std::vector<std::thread> threads;
  for (size_t id = 0; id < meson_count; id++) {
    threads.emplace_back(&Heavy_Mesons_Processor<T>::process_eigenvals, this, id, std::cref(matrixies[id]), std::ref(result));
  }

  for (auto& thread : threads) {
    thread.join();
  }
  return result;
}

template <class T> 
Eigen::VectorXd Heavy_Mesons_Processor<T>::get_heavy_meson_mass_num_spectra() {
  auto matrixies = make_matrixies_ready_num();
  Eigen::VectorXd result(3 * meson_count);
  std::vector<std::thread> threads;
  for (size_t id = 0; id < meson_count; id++) {
    threads.emplace_back(&Heavy_Mesons_Processor<T>::process_eigenvals_spectra, this, id, std::cref(matrixies[id]), std::ref(result));
  }

  for (auto& thread : threads) {
    thread.join();
  }
  return result;
}

template <class T>
void Heavy_Mesons_Processor<T>::process_eigenvals(size_t id, const Eigen::MatrixXd& Ham, Eigen::VectorXd& result) {
  auto masses = get_eigenvals(Ham).array().sqrt();
  for (size_t i = 0; i < 3; i++) {
    result(3 * id + i) = masses(i);
  }
}

template <class T>
void Heavy_Mesons_Processor<T>::process_eigenvals_spectra(size_t id, const Eigen::MatrixXd& Ham, Eigen::VectorXd& result) {
  auto masses = get_eigenvals_spectra(Ham, 3).array().sqrt();
  for (size_t i = 0; i < 3; i++) {
    result(3 * id + i) = masses(i);
  }
}

template <class T>
Eigen::VectorXd Heavy_Mesons_Processor<T>::get_eigenvals(const Eigen::MatrixXd& Ham) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Ham);
  if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Eigen не смог диагонализировать матрицу Гамильтониана!");
  }
  Eigen::VectorXd eigenvalues = solver.eigenvalues();
  return  eigenvalues;
}

template <class T>
Eigen::VectorXd Heavy_Mesons_Processor<T>::get_eigenvals_spectra(const Eigen::MatrixXd& Ham, size_t count_masses) {
  size_t ncv = 5 * count_masses;
  if (Ham.rows() < 5 * count_masses) {
    ncv = Ham.rows();
  }
  Spectra::DenseSymMatProd<double> spectra_ham(Ham);
  Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> solver(spectra_ham, count_masses , ncv);

  solver.init();
  auto number_of_reched = solver.compute(Spectra::SortRule::SmallestAlge);
    
  Eigen::VectorXd evalues;
    if (solver.info() == Spectra::CompInfo::Successful) {
      evalues = solver.eigenvalues();
    } else {
      throw std::runtime_error("Spectra не смогла выделить собственные значения");
    }
  return evalues;
}

template <class T> 
std::vector<Eigen::MatrixXd> Heavy_Mesons_Processor<T>::make_matrixies_ready_num() {
  std::vector<Eigen::MatrixXd> matrixies_vector;
  model.inintialise_net();
  model.dilaton_setter_num();
  for (auto& meson : mesons) {
    model.heavy_meson = meson;
    model.make_net_exotic_prod();
    matrixies_vector.emplace_back(model.build_matrix());
  }
  return matrixies_vector;
}

template <class T> 
std::vector<Eigen::MatrixXd> Heavy_Mesons_Processor<T>::make_matrixies_ready_anz() {
  std::vector<Eigen::MatrixXd> matrixies_vector;
  model.inintialise_net();
  model.dilaton_setter();
  for (auto& meson : mesons) {
    model.heavy_meson = meson;
    model.make_net_exotic_prod_anz();
    matrixies_vector.emplace_back(model.build_matrix());
  }
  return matrixies_vector;
}

template <class T>
void Heavy_Mesons_Processor<T>::check_process_times(const HeavyMesonConfig& meson) {
  model.heavy_meson = meson;
  std::cout << "Размер сетки: " << model.total_net_nodes << '\n';
  model.inintialise_net();
  auto start = std::chrono::high_resolution_clock::now();
  model.dilaton_setter_num();
  auto set_dilaton_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(set_dilaton_time - start);
  std::cout << "Инициализация дилатона заняла:" << duration.count() << "microseconds\n";

  auto start_net_making = std::chrono::high_resolution_clock::now();
  model.make_net_exotic_prod();
  auto end_net_making = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end_net_making - start_net_making);
  std::cout << "Инициализация потенциала заняла:" << duration.count() << "microseconds\n";

  auto start_matrix_building = std::chrono::high_resolution_clock::now();
  auto matrix = model.build_matrix();
  auto end_matrix_building = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end_matrix_building - start_matrix_building);
  std::cout << "Построение матрицы заняло:" << duration.count() << "microseconds\n";
  std::cout << matrix(0, 0);
}