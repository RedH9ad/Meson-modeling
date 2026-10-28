#pragma once
#include <vector>
#include <autodiff/forward/dual.hpp>
#include <iostream>

template<class T>
class Ansatz_Abstract {
 public:
  Ansatz_Abstract() = default;

  virtual T operator()(T z) const = 0;

  virtual ~Ansatz_Abstract() = default;
};

#include "Ansatz_abstract_impl.hpp"