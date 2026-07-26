#pragma once
#include <vector>
#include <autodiff/forward/dual.hpp>

template<class T>
class Metric_Deformation {
 private:
  std::vector<double> parametrs;
 public:
  Metric_Deformation() = default;

  explicit Metric_Deformation(std::vector<double> pars) {
  parametrs = std::move(pars);
  }

  T operator()(T z) const {
    if (parametrs.empty()) {
      return T(0.0);
    }
    //TODO разные кейсы что по вайбу, пока заглушка
    return T(0.0);
  }
};

template <class T>
class Metric_Scale_Factor {
 private:
  std::vector<double> parametrs;
 public:
  Metric_Scale_Factor() = default;

  explicit Metric_Scale_Factor(std::vector<double> pars) {
  parametrs = std::move(pars);
  }

  T operator()(T z) const {
    if (parametrs.empty()) {
    return z < T(1e-10) ? T(std::numeric_limits<double>::infinity()) : 1 / z;
  }
  //TODO разные кейсы что по вайбу, пока заглушка
  return T(0.0);
  }
};