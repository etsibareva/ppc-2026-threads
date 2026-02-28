#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace tsibareva_e_integral_calculate_trapezoid_method {

enum class IntegralTestType : std::uint8_t {
  kSuccessSimple2D,
  kSuccessConstant2D,
  kSuccessSimple3D,
  kSuccessConstant3D,
  kInvalidLowerBoundEqual,
  kInvalidStepsNegative,
  kInvalidEmptyBounds,
};

struct IntegralInput {
  std::vector<double> lower_bounds;
  std::vector<double> upper_bounds;
  std::vector<int> num_steps;
  std::function<double(const std::vector<double> &)> function;
  int dimension{0};
};

using InType = IntegralInput;
using OutType = double;
using TestType = std::tuple<IntegralTestType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

inline IntegralInput GenerateIntegralInput(IntegralTestType type) {
  IntegralInput input;

  switch (type) {
    case IntegralTestType::kSuccessSimple2D: {
      input.dimension = 2;
      input.lower_bounds = {0.0, 0.0};
      input.upper_bounds = {1.0, 1.0};
      input.num_steps = {100, 100};
      input.function = [](const std::vector<double> &x) { return (x[0] * x[0]) + (x[1] * x[1]); };
      break;
    }
    case IntegralTestType::kSuccessConstant2D: {
      input.dimension = 2;
      input.lower_bounds = {0.0, 0.0};
      input.upper_bounds = {2.0, 3.0};
      input.num_steps = {50, 50};
      input.function = [](const std::vector<double> &) { return 5.0; };
      break;
    }
    case IntegralTestType::kSuccessSimple3D: {
      input.dimension = 3;
      input.lower_bounds = {0.0, 0.0, 0.0};
      input.upper_bounds = {1.0, 1.0, 1.0};
      input.num_steps = {50, 50, 50};
      input.function = [](const std::vector<double> &x) { return x[0] + x[1] + x[2]; };
      break;
    }
    case IntegralTestType::kSuccessConstant3D: {
      input.dimension = 3;
      input.lower_bounds = {0.0, 0.0, 0.0};
      input.upper_bounds = {2.0, 2.0, 2.0};
      input.num_steps = {40, 40, 40};
      input.function = [](const std::vector<double> &) { return 3.0; };
      break;
    }
    case IntegralTestType::kInvalidLowerBoundEqual: {
      input.dimension = 2;
      input.lower_bounds = {1.0, 0.0};
      input.upper_bounds = {1.0, 1.0};
      input.num_steps = {10, 10};
      input.function = [](const std::vector<double> &x) { return x[0]; };
      break;
    }
    case IntegralTestType::kInvalidStepsNegative: {
      input.dimension = 2;
      input.lower_bounds = {0.0, 0.0};
      input.upper_bounds = {1.0, 1.0};
      input.num_steps = {-5, 10};
      input.function = [](const std::vector<double> &x) { return x[0]; };
      break;
    }
    case IntegralTestType::kInvalidEmptyBounds: {
      input.dimension = 0;
      input.lower_bounds = {};
      input.upper_bounds = {};
      input.num_steps = {};
      input.function = [](const std::vector<double> &) { return 0.0; };
      break;
    }
  }

  return input;
}

inline double GenerateExpectedOutput(IntegralTestType type) {
  switch (type) {
    case IntegralTestType::kSuccessSimple2D:
      return 2.0 / 3.0;
    case IntegralTestType::kSuccessConstant2D:
      return 30.0;
    case IntegralTestType::kSuccessSimple3D:
      return 1.5;
    case IntegralTestType::kSuccessConstant3D:
      return 24.0;
    case IntegralTestType::kInvalidLowerBoundEqual:
    case IntegralTestType::kInvalidStepsNegative:
    case IntegralTestType::kInvalidEmptyBounds:
      return 0.0;
  }
  return 0.0;
}

}  // namespace tsibareva_e_integral_calculate_trapezoid_method
