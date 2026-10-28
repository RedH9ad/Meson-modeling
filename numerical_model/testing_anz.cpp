#include "autodiff/forward/real/real.hpp"
#include "heavy_meson_processor.hpp"
#include <iostream>

constexpr double C_WARP = 1.16;
int main() {
  auto deform_num = [](autodiff::real3rd z) {
    return -(C_WARP / 8.0) * z * z;
  };

  auto deform = [](double z) {
    return -(C_WARP / 8.0) * z * z;
  };
  
  const size_t net_nodes = 3000;
  const double z_max_val = 20.0;
  const double dummy_k   = 1.0; 
  std::vector<HeavyMesonConfig> targets{
    Heavy_Mesons::Charmonium, 
    Heavy_Mesons::Bottomonium, 
    Heavy_Mesons::Tetraquark, 
    Heavy_Mesons::Hybrid_Meson
  };
  auto model_num = Mesons_Model<autodiff::real3rd>(net_nodes, z_max_val, dummy_k, deform_num);
  auto processor = Heavy_Mesons_Processor(model_num, targets);
  // processor.check_process_times(targets[0]);
  auto res = processor.get_heavy_meson_mass_anz_sturm();
  std::cout << res;
}