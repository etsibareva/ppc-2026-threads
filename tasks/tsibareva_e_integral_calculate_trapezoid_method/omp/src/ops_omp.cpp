#include "tsibareva_e_integral_calculate_trapezoid_method/omp/include/ops_omp.hpp"

#include <cmath>
#include <vector>

#include "tsibareva_e_integral_calculate_trapezoid_method/common/include/common.hpp"

namespace tsibareva_e_integral_calculate_trapezoid_method {

TsibarevaEIntegralCalculateTrapezoidMethodOMP::TsibarevaEIntegralCalculateTrapezoidMethodOMP(const InType &in)
    : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool TsibarevaEIntegralCalculateTrapezoidMethodOMP::ValidationImpl() {
  return true;
}

bool TsibarevaEIntegralCalculateTrapezoidMethodOMP::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool TsibarevaEIntegralCalculateTrapezoidMethodOMP::RunImpl() {
  int dim = GetInput().dim;

  std::vector<double> h(dim);
  for (int i = 0; i < dim; ++i) {
    h[i] = (GetInput().hi[i] - GetInput().lo[i]) / GetInput().steps[i];
  }

  int total_nodes = 1;
  std::vector<int> sizes(dim);
  for (int i = 0; i < dim; ++i) {
    sizes[i] = GetInput().steps[i] + 1;
    total_nodes *= sizes[i];
  }

  double global_sum = 0.0;

#pragma omp parallel
  {
    double local_sum = 0.0;

#pragma omp for
    for (int node = 0; node < total_nodes; ++node) {
      std::vector<int> indexes(dim);
      int remainder = node;
      for (int i = dim - 1; i >= 0; --i) {
        indexes[i] = remainder % sizes[i];
        remainder /= sizes[i];
      }

      std::vector<double> point(dim);
      for (int i = 0; i < dim; ++i) {
        
        point[i] = GetInput().lo[i] + (indexes[i] * h[i]);
      }

      int boundary_count = 0;
      for (int i = 0; i < dim; ++i) {
        if (indexes[i] == 0 || indexes[i] == GetInput().steps[i]) {
          boundary_count++;
        }
      }

      double weight = (boundary_count == 0) ? 1.0 : std::pow(0.5, boundary_count);
      local_sum += weight * GetInput().f(point);
    }

#pragma omp atomic
    global_sum += local_sum;
  }

  double res_h = 1.0;
  for (int i = 0; i < dim; ++i) {
    res_h *= h[i];
  }

  GetOutput() = global_sum * res_h;

  return true;
}

bool TsibarevaEIntegralCalculateTrapezoidMethodOMP::PostProcessingImpl() {
  return true;
}

}  // namespace tsibareva_e_integral_calculate_trapezoid_method
