#pragma once
#include <cstddef>
#include <cmath>
#include <vector>
#include "autodiff/forward/real/real.hpp"
#include "numerical_model.hpp"
#include "heavy_meson_processor.hpp"
#include "ansatz_fabric.hpp"

namespace Net_Parametrs {
  constexpr size_t net_nodes = 5000;
  constexpr double z_max = 9.0;
}

// Eigen::VectorXd heavy_meson_mass_counter(double L, double k_sw_impact, std::function<double(double)> metric_deformation) {
//   std::vector<HeavyMesonConfig> targets{Heavy_Mesons::Charmonium, Heavy_Mesons::Bottomonium, Heavy_Mesons::Tetraquark, Heavy_Mesons::Hybrid_Meson};
//   Eigen::VectorXd meson_masses(12);
//   size_t counter = 0;
//   for (HeavyMesonConfig& heavy_meson : targets) {
//     Mesons_Model<double> simulator(Net_Parametrs::net_nodes, Net_Parametrs::z_max, k_sw_impact, L, metric_deformation);
//     simulator.heavy_meson = heavy_meson;
//     auto heavy_mesons = simulator.get_exotic_meson_mass_prod();
//     for (size_t i = 0; i < 3; i++) {
//       meson_masses(counter * 3 + i) = heavy_mesons(i);
//     }
//     counter++;
//   }
//   return meson_masses.array().sqrt();
// }

// Eigen::VectorXd heavy_meson_mass_counter(double L, double k_sw_impact, std::function<double(double)> metric_deformation) {
//   Mesons_Model<double> simulator(Net_Parametrs::net_nodes, Net_Parametrs::z_max, k_sw_impact, L, metric_deformation);
//   std::vector<HeavyMesonConfig> targets{Heavy_Mesons::Charmonium, Heavy_Mesons::Bottomonium, Heavy_Mesons::Tetraquark, Heavy_Mesons::Hybrid_Meson};
//   Heavy_Mesons_Processor<double> worker(simulator, targets);
//   return worker.get_heavy_meson_mass_num();
// }

Eigen::VectorXd heavy_meson_mass_counter(double L, double k_sw_impact, const std::string& ansatz_name, const std::vector<double>& params) {
  std::shared_ptr<Ansatz_Abstract<autodiff::real3rd>> ansatz = create_ansatz<autodiff::real3rd>(ansatz_name, params);
  std::function<autodiff::real3rd(autodiff::real3rd)> func = [ansatz](autodiff::real3rd z) {
      return (*ansatz)(z);
  };
  Mesons_Model<autodiff::real3rd> simulator(Net_Parametrs::net_nodes, Net_Parametrs::z_max, k_sw_impact, L, func);
  std::vector<HeavyMesonConfig> targets{Heavy_Mesons::Charmonium, Heavy_Mesons::Bottomonium, Heavy_Mesons::Tetraquark, Heavy_Mesons::Hybrid_Meson};
  Heavy_Mesons_Processor<autodiff::real3rd> worker(simulator, targets);
  return worker.get_heavy_meson_mass_anz();
}

Eigen::VectorXd heavy_meson_mass_counter_hard_wall(double zet, double L, double k_sw_impact, const std::string& ansatz_name, const std::vector<double>& params) {
  std::shared_ptr<Ansatz_Abstract<autodiff::real3rd>> ansatz = create_ansatz<autodiff::real3rd>(ansatz_name, params);
  std::function<autodiff::real3rd(autodiff::real3rd)> func = [ansatz](autodiff::real3rd z) {
      return (*ansatz)(z);
  };
  Mesons_Model<autodiff::real3rd> simulator(Net_Parametrs::net_nodes, zet, k_sw_impact, L, func);
  std::vector<HeavyMesonConfig> targets{Heavy_Mesons::Charmonium, Heavy_Mesons::Bottomonium, Heavy_Mesons::Tetraquark, Heavy_Mesons::Hybrid_Meson};
  Heavy_Mesons_Processor<autodiff::real3rd> worker(simulator, targets);
  return worker.get_heavy_meson_mass_anz_sturm();
}

Eigen::VectorXd charm_and_tetra_hw(double zet, double L, double k_sw_impact, const std::string& ansatz_name, const std::vector<double>& params) {
  std::shared_ptr<Ansatz_Abstract<autodiff::real3rd>> ansatz = create_ansatz<autodiff::real3rd>(ansatz_name, params);
  std::function<autodiff::real3rd(autodiff::real3rd)> func = [ansatz](autodiff::real3rd z) {
      return (*ansatz)(z);
  };
  Mesons_Model<autodiff::real3rd> simulator(Net_Parametrs::net_nodes, zet, k_sw_impact, L, func);
  std::vector<HeavyMesonConfig> targets{Heavy_Mesons::Charmonium, Heavy_Mesons::Tetraquark};
  Heavy_Mesons_Processor<autodiff::real3rd> worker(simulator, targets);
  return worker.get_heavy_meson_mass_anz_sturm();
}

Eigen::VectorXd charm_hw(double zet, double L, double k_sw_impact, const std::string& ansatz_name, const std::vector<double>& params) {
  std::shared_ptr<Ansatz_Abstract<autodiff::real3rd>> ansatz = create_ansatz<autodiff::real3rd>(ansatz_name, params);
  std::function<autodiff::real3rd(autodiff::real3rd)> func = [ansatz](autodiff::real3rd z) {
      return (*ansatz)(z);
  };
  Mesons_Model<autodiff::real3rd> simulator(Net_Parametrs::net_nodes, zet, k_sw_impact, L, func);
  std::vector<HeavyMesonConfig> targets{Heavy_Mesons::Charmonium};
  Heavy_Mesons_Processor<autodiff::real3rd> worker(simulator, targets);
  return worker.get_heavy_meson_mass_anz_sturm();
}