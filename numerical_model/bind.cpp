#include "function_for_emulator.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

Eigen::VectorXd heavy_meson_mass_counter(double, double, std::function<double(double)> metric_deformation);

namespace py = pybind11;

PYBIND11_MODULE(meson_lib, m) {
    m.doc() = "Heavy mesons simulator Python bindings";

    m.def("heavy_meson_mass_counter", 
          &heavy_meson_mass_counter,
          py::arg("L"),
          py::arg("k_sw_impact"),
          py::arg("metric_deformation"),
          "Calculates heavy meson masses given a metric deformation function.");
}