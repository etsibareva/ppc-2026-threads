#include "tsibareva_e_integral_calculate_trapezoid_method/all/include/ops_all.hpp"

#include <mpi.h>

#include <cmath>
#include <functional>
#include <vector>

#include "tsibareva_e_integral_calculate_trapezoid_method/common/include/common.hpp"

namespace tsibareva_e_integral_calculate_trapezoid_method {

TsibarevaEIntegralCalculateTrapezoidMethodALL::TsibarevaEIntegralCalculateTrapezoidMethodALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool TsibarevaEIntegralCalculateTrapezoidMethodALL::ValidationImpl() {
  return true;
}

bool TsibarevaEIntegralCalculateTrapezoidMethodALL::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

double TsibarevaEIntegralCalculateTrapezoidMethodALL::ComputePartialSum(
    int begin, int finish, const std::vector<double> &lo, const std::vector<double> &h, const std::vector<int> &sizes,
    const std::vector<int> &steps, int dim, const std::function<double(const std::vector<double> &)> &f) {
  double partial = 0.0;
#pragma omp parallel default(none) shared(begin, finish, dim, h, sizes, lo, steps, f) reduction(+ : partial)
  {
    std::vector<double> point(dim);
#pragma omp for
    for (int node = begin; node < finish; ++node) {
      int remainder_idx = node;
      double node_weight = 1.0;
      for (int i = dim - 1; i >= 0; --i) {
        int idx = remainder_idx % sizes[i];
        remainder_idx /= sizes[i];
        if (idx == 0 || idx == steps[i]) {
          node_weight *= 0.5;
        }
        point[i] = lo[i] + (idx * h[i]);
      }
      partial += node_weight * f(point);
    }
  }
  return partial;
}

bool TsibarevaEIntegralCalculateTrapezoidMethodALL::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &lo = GetInput().lo;
  const auto &hi = GetInput().hi;
  const auto &steps = GetInput().steps;
  const auto &f = GetInput().f;
  int dim = GetInput().dim;

  std::vector<double> h(dim);
  std::vector<int> sizes(dim);
  int total_nodes = 1;
  for (int i = 0; i < dim; ++i) {
    h[i] = (hi[i] - lo[i]) / static_cast<double>(steps[i]);
    sizes[i] = steps[i] + 1;
    total_nodes *= sizes[i];
  }

  int nodes_per_proc = total_nodes / size;
  int remainder = total_nodes % size;
  int start = (rank * nodes_per_proc) + (rank < remainder ? rank : remainder);
  int end = start + nodes_per_proc + (rank < remainder ? 1 : 0);

  double local_sum = ComputePartialSum(start, end, lo, h, sizes, steps, dim, f);

  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    double res_h = 1.0;
    for (int i = 0; i < dim; ++i) {
      res_h *= h[i];
    }
    GetOutput() = global_sum * res_h;
  }

  MPI_Bcast(&GetOutput(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool TsibarevaEIntegralCalculateTrapezoidMethodALL::PostProcessingImpl() {
  return true;
}

}  // namespace tsibareva_e_integral_calculate_trapezoid_method
