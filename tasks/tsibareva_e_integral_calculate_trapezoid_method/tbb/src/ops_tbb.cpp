// ops_tbb.cpp
#include "tsibareva_e_integral_calculate_trapezoid_method/tbb/include/ops_tbb.hpp"

#include <cmath>
#include <vector>

#include "tsibareva_e_integral_calculate_trapezoid_method/common/include/common.hpp"
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

namespace tsibareva_e_integral_calculate_trapezoid_method {

TsibarevaEIntegralCalculateTrapezoidMethodTBB::TsibarevaEIntegralCalculateTrapezoidMethodTBB(const InType &in)
    : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool TsibarevaEIntegralCalculateTrapezoidMethodTBB::ValidationImpl() {
  return true;
}

bool TsibarevaEIntegralCalculateTrapezoidMethodTBB::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool TsibarevaEIntegralCalculateTrapezoidMethodTBB::RunImpl() {
  const Integral& input = GetInput();
  int dim = input.dim;

  std::vector<double> h(dim);
  std::vector<int> sizes(dim);
  int total_nodes = 1;
  for (int i = 0; i < dim; ++i) {
    h[i] = (input.hi[i] - input.lo[i]) / input.steps[i];
    sizes[i] = input.steps[i] + 1;
    total_nodes *= sizes[i];
  }

  double total_sum = tbb::parallel_reduce(
      tbb::blocked_range<int>(0, total_nodes),
      0.0,
      [&](const tbb::blocked_range<int>& r, double local_sum) {
        return local_sum + ComputeRangeSum(r, 0.0, input, h, sizes);
      },
      std::plus<double>()
  );

  double res_h = 1.0;
  for (int i = 0; i < dim; ++i) {
    res_h *= h[i];
  }
  GetOutput() = total_sum * res_h;
  return true;
}

bool TsibarevaEIntegralCalculateTrapezoidMethodTBB::PostProcessingImpl() {
  return true;
}

double TsibarevaEIntegralCalculateTrapezoidMethodTBB::ComputeRangeSum(
    const tbb::blocked_range<int>& range,
    double init,
    const Integral& input,
    const std::vector<double>& h,
    const std::vector<int>& sizes) {
  double sum = init;
  int dim = input.dim;

  for (int node = range.begin(); node != range.end(); ++node) {
    int remainder = node;
    double node_weight = 1.0;
    std::vector<double> point(dim);

    for (int i = dim - 1; i >= 0; --i) {
      int idx = remainder % sizes[i];
      remainder /= sizes[i];

      if (idx == 0 || idx == input.steps[i]) {
        node_weight *= 0.5;
      }

      point[i] = input.lo[i] + (idx * h[i]);
    }

    sum += node_weight * input.f(point);
  }

  return sum;
}

}  // namespace tsibareva_e_integral_calculate_trapezoid_method