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

  std::vector<double> h(dim);
  for (int i = 0; i < dim; ++i) {
    h[i] = (input.upper_bounds[i] - input.lower_bounds[i]) / input.num_steps[i];
  }

  std::vector<int> indices(dim, 0);
  double sum = 0.0;

  while (true) {
    std::vector<double> point(dim);
    for (int i = 0; i < dim; ++i) {
      point[i] = input.lower_bounds[i] + (indices[i] * h[i]);
    }

    int boundary_count = 0;
    for (int i = 0; i < dim; ++i) {
      if (indices[i] == 0 || indices[i] == input.num_steps[i]) {
        ++boundary_count;
      }
    }

    double weight = (boundary_count == 0) ? 1.0 : std::pow(0.5, boundary_count);
    sum += weight * input.function(point);

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

  double product_of_steps = 1.0;
  for (int i = 0; i < dim; ++i) {
    product_of_steps *= h[i];
  }

  GetOutput() = sum * product_of_steps;

  return true;
}

bool TsibarevaEIntegralCalculateTrapezoidMethodSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace tsibareva_e_integral_calculate_trapezoid_method
