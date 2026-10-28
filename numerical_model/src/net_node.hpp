#include <cstddef>
#include <optional>
#pragma once
template <class T>
struct NetNode {
  T z = 0.0;
  size_t node_id = 0.0;

  double dilaton = 0.0;
  double dilaton_first_der = 0.0;
  double dilaton_secod_der = 0.0;
  std::optional<double> kiral_sector;

  double potential = 0;

  NetNode() = default;
  NetNode(T zc, size_t id) : z(zc), node_id(id) {
    potential = 0.0;
    double dilaton_first_der = 0.0;
    double dilaton_secod_der = 0.0;
  }
};