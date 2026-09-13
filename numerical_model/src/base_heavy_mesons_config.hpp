#include <iostream>
#include <vector>
#pragma once

//average_constitutet quark mass(из статьи)
namespace Quark_Masses {
  constexpr double U = 0.366;
  constexpr double D = 0.340;
  constexpr double S = 0.486;
  constexpr double C = 1.550;
  constexpr double B = 4.730;
  constexpr double G = 0.700;

}

struct HeavyMesonConfig {
  std::string name;
  double m_bar;             
  double M;
  double sqrt_gamma;
  double spin;
  double scale_dimension;
  std::vector<double> paper_masses_gev;

  double dilaton_alpha;
  double dilaton_k;

  double potential_addition() {
    return (scale_dimension - spin) * (scale_dimension + spin - 4);
  }
};

namespace Heavy_Mesons {
  HeavyMesonConfig Charmonium {
    "Charmonium J/psi",
    Quark_Masses::C,
    1.7,
    0.53,
    1,
    3,
    {3.140, 3.6569, 4.0558},
    0.54,
    1.8
  };

  HeavyMesonConfig Bottomonium {
    "Bottomonium Upsilon",
    Quark_Masses::B,
    2.74,
    1.92,
    1,
    3,
    {9.5065, 9.829, 10.2272},
    0.863,
    9.9
  };

  HeavyMesonConfig Tetraquark {
    "Tetraquark Z_c",
    0.283 * Quark_Masses::C + 0.717 * (Quark_Masses::U + Quark_Masses::D) / 2,
    1.44,
    0.3,
    1,
    6,
    {3.8901, 4.1895, 4.4446},
    0.539,
    1.75
  };

  HeavyMesonConfig Hybrid_Meson {
    "Hybrid meson pi one",
    0.497 * Quark_Masses::U + 0.497 * Quark_Masses::D + 0.006 * Quark_Masses::G,
    0.20,
    0.12,
    1,
    5,
    {1.3547, 1.6181, 1.8499},
    0.034,
    0.468
  };
}