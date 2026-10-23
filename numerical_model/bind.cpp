#include "function_for_emulator.hpp"
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <vector>

// Eigen::VectorXd heavy_meson_mass_counter(double, double, std::function<double(double)> metric_deformation);
Eigen::VectorXd heavy_meson_mass_counter(double, double, const std::string&, const std::vector<double>&);
Eigen::VectorXd heavy_meson_mass_counter_hard_wall(double z_max, double L, double k_sw_impact, const std::string& ansatz_name, const std::vector<double>& params);
Eigen::VectorXd charm_and_tetra_hw(double zet, double L, double k_sw_impact, const std::string& ansatz_name, const std::vector<double>& params);
Eigen::VectorXd charm_hw(double zet, double L, double k_sw_impact, const std::string& ansatz_name, const std::vector<double>& params);
namespace py = pybind11;

// PYBIND11_MODULE(meson_lib, m) {
//     m.doc() = "Heavy mesons simulator Python bindings";

//     m.def("heavy_meson_mass_counter", 
//           &heavy_meson_mass_counter,
//           py::arg("L"),
//           py::arg("k_sw_impact"),
//           py::arg("metric_deformation"),
//           "Calculates heavy meson masses given a metric deformation function.");
// }

PYBIND11_MODULE(meson_lib, m) {
    m.doc() = "Heavy mesons simulator Python bindings";

    m.def("heavy_meson_mass_counter", 
          &heavy_meson_mass_counter,
          py::arg("L"),
          py::arg("k_sw_impact"),
          py::arg("ansatz_name"),
          py::arg("params"),
          "Calculates heavy meson masses given a metric deformation function.");
    
    m.doc() = "Heavy mesons simulator Python bindings";

    m.def("heavy_meson_mass_counter_hardwall", 
          &heavy_meson_mass_counter_hard_wall,
          py::arg("z_max"),
          py::arg("L"),
          py::arg("k_sw_impact"),
          py::arg("ansatz_name"),
          py::arg("params"),
          "Calculates heavy meson masses given a metric deformation function using hard wall model.");

    m.doc() = "Heavy mesons simulator Python bindings";

    m.def("charm_and_thetra_hw", 
          &charm_and_tetra_hw,
          py::arg("z_max"),
          py::arg("L"),
          py::arg("k_sw_impact"),
          py::arg("ansatz_name"),
          py::arg("params"),
          "Calculates heavy meson masses given a metric deformation function using hard wall model.");

    m.doc() = "Heavy mesons simulator Python bindings";

    m.def("charm_hw", 
          &charm_hw,
          py::arg("z_max"),
          py::arg("L"),
          py::arg("k_sw_impact"),
          py::arg("ansatz_name"),
          py::arg("params"),
          "Calculates Charmonium meson masses given a metric deformation function using hard wall model.");
}

