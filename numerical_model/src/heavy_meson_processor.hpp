#pragma once
#include "numerical_model.hpp"
#include "base_heavy_mesons_config.hpp"
#include <vector>

//В классе родителе build matrix просто выстраивает матрицу по потенциалу и возвращает ее
//Значит по сути нам достаточно интегрировать перерассчет потенциалов
//По стуи теперь не обязательно вообще наследоваться, можно просто сдеалть обертку
//Принимающую класс симулятора и просто процессирующий матрицу 
//
template <class T>
class Heavy_Mesons_Processor {
  public:
  std::vector<HeavyMesonConfig> mesons;
  Mesons_Model<T> model;
  size_t meson_count;

  Heavy_Mesons_Processor() = default;
  Heavy_Mesons_Processor(Mesons_Model<T>&, const std::vector<HeavyMesonConfig>&);

  Eigen::VectorXd get_heavy_meson_mass_num();

  Eigen::VectorXd get_eigenvals(const Eigen::MatrixXd&);

  void process_eigenvals(size_t, const Eigen::MatrixXd&, Eigen::VectorXd&);

  std::vector<Eigen::MatrixXd> make_matrixies_ready_num();


  void check_process_times(const HeavyMesonConfig&);
};

#include "heavy_meson_processor_impl.hpp"