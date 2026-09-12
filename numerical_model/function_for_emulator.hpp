#include <cstddef>
#include <cmath>
#include <vector>
#include "numerical_model.hpp"
#pragma once

namespace Net_Parametrs {
  constexpr size_t net_nodes = 1000;
  constexpr double z_max = 10.0;
}

Eigen::VectorXd heavy_meson_mass_counter(double L, double k_sw_impact, std::function<double(double)> metric_deformation) {
  std::vector<HeavyMesonConfig> targets{Heavy_Mesons::Charmonium, Heavy_Mesons::Bottomonium, Heavy_Mesons::Tetraquark, Heavy_Mesons::Hybrid_Meson};
  Eigen::VectorXd meson_masses(12);
  auto x = new int[32];
  size_t counter = 0;
  for (HeavyMesonConfig& heavy_meson : targets) {
    Mesons_Model<double> simulator(Net_Parametrs::net_nodes, Net_Parametrs::z_max, k_sw_impact, L, metric_deformation);
    simulator.heavy_meson = heavy_meson;
    auto heavy_mesons = simulator.get_exotic_meson_mass_prod();
    for (size_t i = 0; i < 3; i++) {
      meson_masses(counter * 3 + i) = heavy_mesons(i);
    }
    counter++;
  }
  return meson_masses.array().sqrt();
}
