#pragma once
#include "Ansatz_abstract.hpp"
#include "ChenHuan.hpp"
#include "parametric_ansatz.hpp"
#include <memory>
#include <stdexcept>
#include <string>

template <class T>
std::unique_ptr<Ansatz_Abstract<T>> create_ansatz(const std::string& key, const std::vector<double>& pars) {
  std::string lower_name = key;
  std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
  if (lower_name == "param" || lower_name == "parametric") {
    return std::make_unique<Parametric_ansatz<T>>( (pars));
  }
  else if (lower_name == "chen" || lower_name=="chen-huang") {
    return std::make_unique<ChenHuan<T>>(pars);
  }
  throw std::runtime_error("Неизвестный тип анзаца:" + key);
}