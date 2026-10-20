#pragma once
#include "Ansatz_abstract.hpp"

template <class T>
class ChenHuan : public Ansatz_Abstract<T> {
  private:
  double scale_1 = 0;
  double scale_2 = 0;
  double free_c = 0;
  double coef = 0;
  double free_k = 0;
  double G5 = 0.885;

 public: 
  explicit ChenHuan(const std::vector<double>&);
  virtual T operator()(T) const override final;  
};

#include "ChenHuan_impl.hpp"