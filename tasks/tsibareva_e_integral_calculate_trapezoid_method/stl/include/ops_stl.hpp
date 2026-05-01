#pragma once

#include <functional>
#include <vector>

#include "task/include/task.hpp"
#include "tsibareva_e_integral_calculate_trapezoid_method/common/include/common.hpp"

namespace tsibareva_e_integral_calculate_trapezoid_method {

class TsibarevaEIntegralCalculateTrapezoidMethodSTL : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSTL;
  }
  explicit TsibarevaEIntegralCalculateTrapezoidMethodSTL(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static void MWork(int thread_id, int start, int end, const std::vector<int> &sizes, const std::vector<double> &h,
                    std::vector<double> &partial_sums, int dim, const std::vector<int> &steps,
                    const std::vector<double> &lo, const std::function<double(const std::vector<double> &)> &f);
};

}  // namespace tsibareva_e_integral_calculate_trapezoid_method
