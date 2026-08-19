/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "mps_structure_analysis.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace cuopt::mathematical_optimization::examples {

enum class lp_objective_mode_t { source, erased };

enum class lp_overlay_status_t {
  not_run,
  optimal,
  infeasible,
  unbounded,
  unbounded_or_infeasible,
  iteration_limit,
  time_limit,
  numerical_issues,
  cutoff,
  concurrent_limit,
  work_limit,
  unsupported,
  invalid_options,
  error
};

struct lp_overlay_options_t {
  lp_objective_mode_t objective_mode{lp_objective_mode_t::source};
  double time_limit_seconds{30.0};
  structure_index_t iteration_limit{2'000'000};
  structure_index_t threads{1};
  double integrality_tolerance{1e-6};
  double near_integrality_tolerance{1e-4};
  double activity_tolerance{1e-7};
  double zero_tolerance{1e-9};
  bool solver_log{false};
};

struct lp_histogram_bin_t {
  double lower_bound{};
  double upper_bound{};
  bool lower_bound_inclusive{};
  bool upper_bound_inclusive{true};
  std::size_t count{};
};

struct lp_histogram_t {
  std::string quantity;
  std::vector<lp_histogram_bin_t> bins;
};

struct lp_scalar_distribution_t {
  std::size_t count{};
  quantile_summary_t quantiles;
};

struct lp_variable_record_t {
  structure_index_t column{};
  structure_index_t original_column{};
  std::string name;
  char source_type{'C'};
  std::string family;
  double value{};
  double source_objective_coefficient{};
  double reduced_cost{};
  std::optional<double> fractional_part;
  std::optional<double> distance_to_integrality;
  bool fractional{};
};

struct lp_row_record_t {
  structure_index_t row{};
  std::string stable_id;
  std::string name;
  std::string domain;
  std::vector<std::string> families;
  double activity{};
  std::optional<double> lower_bound;
  std::optional<double> upper_bound;
  std::optional<double> lower_slack;
  std::optional<double> upper_slack;
  std::optional<double> nearest_bound_distance;
  double violation{};
  double dual{};
  bool active{};
};

struct lp_row_summary_t {
  std::size_t rows{};
  std::size_t active_rows{};
  std::size_t violated_rows{};
  std::size_t nonzero_duals{};
  lp_scalar_distribution_t nearest_bound_distance;
  lp_scalar_distribution_t signed_dual;
  lp_scalar_distribution_t absolute_dual;
};

struct lp_row_family_summary_t {
  std::string family;
  lp_row_summary_t summary;
};

struct lp_reduced_cost_family_summary_t {
  std::string family;
  std::size_t variables{};
  std::size_t negative_reduced_costs{};
  std::size_t positive_reduced_costs{};
  std::size_t zero_reduced_costs{};
  lp_scalar_distribution_t signed_reduced_cost;
  lp_scalar_distribution_t absolute_reduced_cost;
};

struct lp_exact_one_group_record_t {
  structure_index_t group{};
  structure_index_t row{};
  std::string stable_id;
  std::vector<literal_t> members;
  std::vector<double> literal_values;
  double literal_sum{};
  double entropy{};
  double effective_support{};
  double largest_value{};
  double second_largest_value{};
  double margin{};
  std::optional<literal_t> leading_literal;
  std::optional<literal_t> second_literal;
  std::size_t fractional_members{};
  bool integral{};
  bool nearly_integral{};
};

struct lp_exact_one_summary_t {
  std::size_t groups{};
  std::size_t integral_groups{};
  std::size_t nearly_integral_groups{};
  std::size_t fractional_groups{};
  lp_scalar_distribution_t entropy;
  lp_scalar_distribution_t effective_support;
  lp_scalar_distribution_t largest_value;
  lp_scalar_distribution_t second_largest_value;
  lp_scalar_distribution_t margin;
};

struct lp_overlay_summary_t {
  bool attempted{};
  bool has_optimal_solution{};
  bool cuts_added{};
  lp_objective_mode_t objective_mode{lp_objective_mode_t::source};
  lp_overlay_status_t status{lp_overlay_status_t::not_run};
  std::string formulation;
  std::string detail;
  double time_limit_seconds{};
  structure_index_t iteration_limit{};
  structure_index_t threads{};
  double solve_seconds{};
  structure_index_t iterations{};
  std::optional<double> relaxation_objective;
  std::optional<double> source_objective_at_solution;
  std::optional<double> l2_primal_residual;
  std::optional<double> l2_dual_residual;

  std::size_t integer_variables{};
  std::size_t binary_variables{};
  std::size_t fractional_integer_variables{};
  std::size_t fractional_binary_variables{};
  lp_histogram_t fractional_part_histogram;
  lp_histogram_t distance_to_integrality_histogram;

  lp_row_summary_t rows;
  std::vector<lp_row_family_summary_t> row_family_summaries;
  std::vector<lp_reduced_cost_family_summary_t> reduced_cost_family_summaries;
  lp_exact_one_summary_t exact_one;

  // Complete source-indexed vectors and records are retained for JSON sidecars and follow-up work.
  std::vector<double> primal_values;
  std::vector<double> row_duals;
  std::vector<double> reduced_costs;
  std::vector<lp_variable_record_t> variables;
  std::vector<lp_row_record_t> row_records;
  std::vector<lp_exact_one_group_record_t> exact_one_groups;
};

lp_overlay_summary_t analyze_root_lp_overlay(const structure_model_t& model,
                                             const model_analysis_t& analysis,
                                             const lp_overlay_options_t& options = {});

void print_root_lp_overlay_summary(const lp_overlay_summary_t& summary);

std::string_view lp_objective_mode_name(lp_objective_mode_t mode);
std::string_view lp_overlay_status_name(lp_overlay_status_t status);

}  // namespace cuopt::mathematical_optimization::examples
