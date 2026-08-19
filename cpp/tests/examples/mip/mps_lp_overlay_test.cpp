/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "mps_lp_overlay.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::examples {
namespace {

structure_model_t make_maximization_model()
{
  constexpr double objective_scale  = 2.0;
  constexpr double objective_offset = 1.0;
  const double infinity             = std::numeric_limits<double>::infinity();
  const std::vector<double> matrix_values{1.0, 1.0, 1.0, 1.0};
  const std::vector<structure_index_t> matrix_indices{0, 1, 2, 0};
  const std::vector<structure_index_t> matrix_offsets{0, 3, 4};
  const std::vector<double> objective{10.0, 6.0, 4.0};
  const std::vector<char> variable_types{'B', 'B', 'B'};
  const std::vector<double> variable_lower{0.0, 0.0, 0.0};
  const std::vector<double> variable_upper{1.0, 1.0, 1.0};
  const std::vector<double> row_lower{1.0, -infinity};
  const std::vector<double> row_upper{1.0, 0.5};

  structure_model_t model;
  model.set_csr_constraint_matrix(matrix_values, matrix_indices, matrix_offsets);
  model.set_objective_coefficients(objective);
  model.set_variable_types(variable_types);
  model.set_variable_lower_bounds(variable_lower);
  model.set_variable_upper_bounds(variable_upper);
  model.set_constraint_lower_bounds(row_lower);
  model.set_constraint_upper_bounds(row_upper);
  model.set_variable_names({"x", "y", "z"});
  model.set_row_names({"choose_one", "x_capacity"});
  model.set_problem_name("max-scaled-root-lp");
  model.set_objective_name("profit");
  model.set_maximize(true);
  model.set_objective_scaling_factor(objective_scale);
  model.set_objective_offset(objective_offset);
  return model;
}

model_analysis_t make_analysis()
{
  model_analysis_t analysis;
  analysis.scope               = "original";
  analysis.original_column_ids = {0, 1, 2};

  row_group_t exact_one;
  exact_one.id        = 0;
  exact_one.row       = 0;
  exact_one.stable_id = "original:choose_one";
  exact_one.family    = "exact_one";
  exact_one.members   = {{0, true}, {1, true}, {2, true}};
  analysis.exact_one_groups.push_back(std::move(exact_one));
  return analysis;
}

TEST(MpsLpOverlay, NormalizesScaledMaximizationAndSourceDualSigns)
{
  const auto model    = make_maximization_model();
  const auto analysis = make_analysis();

  lp_overlay_options_t options;
  options.time_limit_seconds = 5.0;
  options.iteration_limit    = 10'000;
  options.threads            = 1;
  const auto summary         = analyze_root_lp_overlay(model, analysis, options);

  ASSERT_EQ(summary.status, lp_overlay_status_t::optimal) << summary.detail;
  ASSERT_TRUE(summary.has_optimal_solution);
  EXPECT_FALSE(summary.cuts_added);
  EXPECT_NE(summary.formulation.find("LP relaxation"), std::string::npos);
  EXPECT_NE(summary.formulation.find("no cuts"), std::string::npos);

  ASSERT_TRUE(summary.relaxation_objective);
  ASSERT_TRUE(summary.source_objective_at_solution);
  EXPECT_NEAR(*summary.relaxation_objective, 18.0, 1e-8);
  EXPECT_NEAR(*summary.source_objective_at_solution, 18.0, 1e-8);

  ASSERT_EQ(summary.primal_values.size(), 3);
  EXPECT_NEAR(summary.primal_values[0], 0.5, 1e-8);
  EXPECT_NEAR(summary.primal_values[1], 0.5, 1e-8);
  EXPECT_NEAR(summary.primal_values[2], 0.0, 1e-8);
  EXPECT_EQ(summary.integer_variables, 3);
  EXPECT_EQ(summary.fractional_integer_variables, 2);

  ASSERT_EQ(summary.row_duals.size(), 2);
  EXPECT_NEAR(summary.row_duals[0], 12.0, 1e-5);
  EXPECT_NEAR(summary.row_duals[1], 8.0, 1e-5);
  ASSERT_EQ(summary.reduced_costs.size(), 3);
  EXPECT_NEAR(summary.reduced_costs[0], 0.0, 1e-5);
  EXPECT_NEAR(summary.reduced_costs[1], 0.0, 1e-5);
  EXPECT_NEAR(summary.reduced_costs[2], -4.0, 1e-5);

  ASSERT_EQ(summary.exact_one_groups.size(), 1);
  EXPECT_FALSE(summary.exact_one_groups[0].integral);
  EXPECT_EQ(summary.exact_one_groups[0].fractional_members, 2);
  EXPECT_NEAR(summary.exact_one_groups[0].entropy, std::log(2.0), 1e-8);
  EXPECT_NEAR(summary.exact_one_groups[0].effective_support, 2.0, 1e-8);
  EXPECT_NEAR(summary.exact_one_groups[0].largest_value, 0.5, 1e-8);
  EXPECT_NEAR(summary.exact_one_groups[0].second_largest_value, 0.5, 1e-8);
  EXPECT_NEAR(summary.exact_one_groups[0].margin, 0.0, 1e-8);
}

}  // namespace
}  // namespace cuopt::mathematical_optimization::examples
