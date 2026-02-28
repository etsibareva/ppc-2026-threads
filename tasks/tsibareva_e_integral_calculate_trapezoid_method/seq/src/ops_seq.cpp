#include "tsibareva_e_integral_calculate_trapezoid_method/seq/include/ops_seq.hpp"

#include <cmath>
#include <vector>

#include "task/include/task.hpp"
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

std::vector<double> TsibarevaEIntegralCalculateTrapezoidMethodSEQ::ComputePoint(const std::vector<int> &indices,
                                                                                const std::vector<double> &h,
                                                                                int dim) const {
  const auto &input = GetInput();
  std::vector<double> point(dim);
  for (int i = 0; i < dim; ++i) {
    point[i] = input.lower_bounds[i] + (indices[i] * h[i]);
  }
  return point;
}

int TsibarevaEIntegralCalculateTrapezoidMethodSEQ::ComputeBoundaryCount(const std::vector<int> &indices,
                                                                        int dim) const {
  const auto &input = GetInput();
  int count = 0;
  for (int i = 0; i < dim; ++i) {
    if (indices[i] == 0 || indices[i] == input.num_steps[i]) {
      ++count;
    }
  }
  return count;
}

bool TsibarevaEIntegralCalculateTrapezoidMethodSEQ::AdvanceIndices(std::vector<int> &indices, int dim) const {
  const auto &input = GetInput();
  int position = dim - 1;
  while (position >= 0) {
    if (++indices[position] <= input.num_steps[position]) {
      return true;
    }
    indices[position] = 0;
    --position;
  }
  return false;
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
    std::vector<double> point = ComputePoint(indices, h, dim);
    int boundary_count = ComputeBoundaryCount(indices, dim);

    double weight = (boundary_count == 0) ? 1.0 : std::pow(0.5, boundary_count);
    sum += weight * input.function(point);

    if (!AdvanceIndices(indices, dim)) {
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
