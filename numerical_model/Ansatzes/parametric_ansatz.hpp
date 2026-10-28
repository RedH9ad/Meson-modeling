#pragma once
#include "Ansatz_abstract.hpp"
#include "vector"
#include <vector>

template <class T>
class Parametric_ansatz : public Ansatz_Abstract<T> {
 private:
  std::vector<double> exponents {2.0, 4.0, 6.0};
  double coef_1 = 0;
  double coef_2 = 0;
  double coef_3 = 0;

  double scale_1 = 0;
  double scale_2 = 0;
  double scale_3 = 0;

 public:
 explicit Parametric_ansatz(const std::vector<double>&);
 virtual T operator()(T) const override final;  
};



#include "parametric_ansatz_impl.hpp"