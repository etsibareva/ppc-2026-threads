#include "tsibareva_e_integral_calculate_trapezoid_method/seq/include/ops_seq.hpp"

#include <cmath>
#include <vector>

#include "tsibareva_e_integral_calculate_trapezoid_method/common/include/common.hpp"

namespace tsibareva_e_integral_calculate_trapezoid_method {

TsibarevaEIntegralCalculateTrapezoidMethodSEQ::TsibarevaEIntegralCalculateTrapezoidMethodSEQ(const IntegralInput &in)
    : ppc::task::Task<IntegralInput, double>() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool TsibarevaEIntegralCalculateTrapezoidMethodSEQ::ValidationImpl() {
  return true;
}

bool TsibarevaEIntegralCalculateTrapezoidMethodSEQ::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool TsibarevaEIntegralCalculateTrapezoidMethodSEQ::RunImpl() {
  const auto &input = GetInput();
  int dim = input.dimension;

  // Вычисляем шаги по каждому измерению
  std::vector<double> h(dim);
  for (int idx = 0; idx < dim; ++idx) {  // Fix: rename loop variable
    h[idx] = (input.upper_bounds[idx] - input.lower_bounds[idx]) / input.num_steps[idx];
  }

  // Индексы текущей точки
  std::vector<int> indices(dim, 0);
  double sum = 0.0;

  // Перебор всех узлов сетки
  while (true) {
    // Формируем точку
    std::vector<double> point(dim);
    for (int idx = 0; idx < dim; ++idx) {                            // Fix: rename loop variable
      point[idx] = input.lower_bounds[idx] + indices[idx] * h[idx];  // Fix: add parentheses
    }

    // Вычисляем вес для метода трапеций
    int boundary_count = 0;
    for (int idx = 0; idx < dim; ++idx) {  // Fix: rename loop variable
      if (indices[idx] == 0 || indices[idx] == input.num_steps[idx]) {
        ++boundary_count;
      }
    }

    // Вес = 0.5 для каждой граничной координаты
    double weight = (boundary_count == 0) ? 1.0 : std::pow(0.5, boundary_count);
    sum += weight * input.function(point);

    // Переход к следующей точке
    int idx = dim - 1;
    while (idx >= 0) {
      if (++indices[idx] <= input.num_steps[idx]) {
        break;
      }
      indices[idx] = 0;
      --idx;
    }
    if (idx < 0) {
      break;
    }
  }

  // Умножаем на произведение шагов
  double product_of_steps = 1.0;
  for (int idx = 0; idx < dim; ++idx) {  // Fix: rename loop variable
    product_of_steps *= h[idx];
  }

  GetOutput() = sum * product_of_steps;

  return true;
}

bool TsibarevaEIntegralCalculateTrapezoidMethodSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace tsibareva_e_integral_calculate_trapezoid_method
