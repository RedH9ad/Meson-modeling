#pragma once
#include <cmath>
#include <vector>
#include <autodiff/forward/dual.hpp>
#include <iostream>

template<class T>
class Metric_Deformation {
 private:
  std::vector<double> parametrs;
 public:
  Metric_Deformation() = default;

  explicit Metric_Deformation(std::vector<double> pars) {
  parametrs = std::move(pars);
  std::cout <<"Использован конструктор с параметрами" << '\n';
  }

  T operator()(T z) const {
    if (parametrs.empty()) {
      return T(0.0);
    }
    //Для второго тест кейса первый параметр k(aka u_g) второй лямбда_g
    if (parametrs.size() == 2) {
      auto k = parametrs[0];
      auto h = parametrs[1];

      using std::pow;
      return -((pow(h, 4) * pow(k, 4) * pow(z, 4)) / pow((1 + pow(k, 2) * pow(z, 2)), 3));
    }

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