#pragma once
#include "parametric_ansatz.hpp"
#include <cmath>
#include <stdexcept>

template <class T>
Parametric_ansatz<T>::Parametric_ansatz(const std::vector<double>& parms) {
  if (parms.size() != 2 * exponents.size()) {
    throw std::runtime_error("Количество полученных парамтерическим анзацем параметров не соответствует ожидаемому");
  }
  coef_1 = parms[0];
  coef_2 = parms[1];
  coef_3 = parms[2];

  scale_1 = parms[3];
  scale_2 = parms[4];
  scale_3 = parms[5];
}

template <class T>
T Parametric_ansatz<T>::operator()(T z) const {
  using std::pow, std::log;
  T first_term = coef_1 * log(1 + scale_1 * pow(z, exponents[0]));
  T second_term = coef_2 * log(1 + scale_2 * pow(z, exponents[1]));
  T third_term = coef_3 * log(1 + scale_3 * pow(z, exponents[2]));
  return first_term + second_term + third_term;  
}