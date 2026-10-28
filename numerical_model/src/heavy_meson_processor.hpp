#pragma once
#include "numerical_model.hpp"
#include "base_heavy_mesons_config.hpp"
#include <cstddef>
#include <vector>

template <class T>
class Heavy_Mesons_Processor {
  public:
  std::vector<HeavyMesonConfig> mesons;
  Mesons_Model<T> model;
  size_t meson_count;

  Heavy_Mesons_Processor() = default;
  Heavy_Mesons_Processor(Mesons_Model<T>&, const std::vector<HeavyMesonConfig>&);

  Eigen::VectorXd get_heavy_meson_mass_num();
  Eigen::VectorXd get_heavy_meson_mass_num_spectra();

  Eigen::VectorXd get_heavy_meson_mass_anz();
  Eigen::VectorXd get_heavy_meson_mass_anz_sturm();

  Eigen::VectorXd get_eigenvals(const Eigen::MatrixXd&);
  Eigen::VectorXd get_eigenvals_sturm(const std::pair<Eigen::VectorXd, Eigen::VectorXd>&);
  Eigen::VectorXd get_eigenvals_spectra(const Eigen::MatrixXd&, size_t);

  void process_eigenvals(size_t, const Eigen::MatrixXd&, Eigen::VectorXd&);
  void process_eigenvals_sturm(size_t, const std::pair<Eigen::VectorXd, Eigen::VectorXd>&, Eigen::VectorXd&);
  void process_eigenvals_spectra(size_t, const Eigen::MatrixXd&, Eigen::VectorXd&);

  std::vector<Eigen::MatrixXd> make_matrixies_ready_num();
  std::vector<Eigen::MatrixXd> make_matrixies_ready_anz();
  std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> make_matrixies_ready_anz_sturm();


  void check_process_times(const HeavyMesonConfig&);
};

#include "heavy_meson_processor_impl.hpp"