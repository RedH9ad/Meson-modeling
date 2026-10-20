#pragma once
#include "ChenHuan.hpp"
#include <cmath>
#include <stdexcept>

template <class T>
ChenHuan<T>::ChenHuan(const std::vector<double>& params) {
  if (params.size() != 5) {
    throw std::runtime_error("Полученное анзацем ChenHuang количество параметров не соответствует ожидаемому");
  }
  scale_1 = params[0];
  scale_2 = params[1];
  free_c = params[2];
  coef = params[3];
  free_k = params[4];
}

template <class T>
T ChenHuan<T>::operator()(T z) const {
  using std::pow, std::log;
  T first_term = coef * log(1 + scale_1 * pow(z, 2));
  T second_term = coef * log(1 + scale_2 * pow(z, 4));
  return first_term + second_term;
}