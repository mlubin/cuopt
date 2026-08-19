/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/mathematical_optimization/cpu_optimization_problem.hpp>
#include <cuopt/mathematical_optimization/optimization_problem_utils.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_solution.hpp>

#include <dual_simplex/solve.hpp>
#include <dual_simplex/user_problem.hpp>
#include <math_optimization/tic_toc.hpp>

#include "mps_lp_overlay.hpp"

namespace cuopt::mathematical_optimization::mip {
template <typename i_t, typename f_t>
class problem_t;
}  // namespace cuopt::mathematical_optimization::mip

#include <pdlp/translate.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace cuopt::mathematical_optimization::examples {
namespace {

using simplex_status_t = simplex::lp_status_t;

constexpr double infinity = std::numeric_limits<double>::infinity();

bool is_integer_type(char type) { return type == 'I' || type == 'B'; }

char source_type(const structure_model_t& model, structure_index_t column)
{
  const auto& types = model.get_variable_types();
  return column >= 0 && static_cast<std::size_t>(column) < types.size()
           ? types[static_cast<std::size_t>(column)]
           : 'C';
}

bool is_binary_column(const structure_model_t& model, structure_index_t column)
{
  const auto offset = static_cast<std::size_t>(column);
  const auto& lower = model.get_variable_lower_bounds();
  const auto& upper = model.get_variable_upper_bounds();
  if (column < 0 || offset >= lower.size() || offset >= upper.size() ||
      !is_integer_type(source_type(model, column))) {
    return false;
  }
  const auto lower_is_binary = lower[offset] == 0.0 || lower[offset] == 1.0;
  const auto upper_is_binary = upper[offset] == 0.0 || upper[offset] == 1.0;
  return lower_is_binary && upper_is_binary && lower[offset] <= upper[offset];
}

std::string variable_name(const structure_model_t& model, structure_index_t column)
{
  const auto& names = model.get_variable_names();
  return column >= 0 && static_cast<std::size_t>(column) < names.size() &&
             !names[static_cast<std::size_t>(column)].empty()
           ? names[static_cast<std::size_t>(column)]
           : "x[" + std::to_string(column) + "]";
}

std::string row_name(const structure_model_t& model, structure_index_t row)
{
  const auto& names = model.get_row_names();
  return row >= 0 && static_cast<std::size_t>(row) < names.size() &&
             !names[static_cast<std::size_t>(row)].empty()
           ? names[static_cast<std::size_t>(row)]
           : "row[" + std::to_string(row) + "]";
}

double interpolate_quantile(const std::vector<double>& sorted, double probability)
{
  if (sorted.empty()) { return 0.0; }
  const double position = probability * static_cast<double>(sorted.size() - 1);
  const auto lower      = static_cast<std::size_t>(std::floor(position));
  const auto upper      = static_cast<std::size_t>(std::ceil(position));
  const double weight   = position - static_cast<double>(lower);
  return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

lp_scalar_distribution_t summarize_values(std::vector<double> values)
{
  values.erase(std::remove_if(
                 values.begin(), values.end(), [](double value) { return !std::isfinite(value); }),
               values.end());
  lp_scalar_distribution_t summary;
  summary.count = values.size();
  if (values.empty()) { return summary; }
  std::sort(values.begin(), values.end());
  summary.quantiles.minimum       = values.front();
  summary.quantiles.quartile_1    = interpolate_quantile(values, 0.25);
  summary.quantiles.median        = interpolate_quantile(values, 0.50);
  summary.quantiles.quartile_3    = interpolate_quantile(values, 0.75);
  summary.quantiles.percentile_90 = interpolate_quantile(values, 0.90);
  summary.quantiles.percentile_99 = interpolate_quantile(values, 0.99);
  summary.quantiles.maximum       = values.back();
  return summary;
}

lp_histogram_t make_histogram(std::string quantity,
                              const std::vector<double>& values,
                              const std::vector<double>& upper_bounds)
{
  lp_histogram_t histogram;
  histogram.quantity = std::move(quantity);
  double lower       = 0.0;
  for (std::size_t bin = 0; bin < upper_bounds.size(); ++bin) {
    histogram.bins.push_back(lp_histogram_bin_t{lower, upper_bounds[bin], bin == 0, true, 0});
    lower = upper_bounds[bin];
  }
  for (const double value : values) {
    if (!std::isfinite(value)) { continue; }
    for (auto& bin : histogram.bins) {
      if (value <= bin.upper_bound &&
          (bin.lower_bound_inclusive ? value >= bin.lower_bound : value > bin.lower_bound)) {
        ++bin.count;
        break;
      }
    }
  }
  return histogram;
}

lp_overlay_status_t translate_status(simplex_status_t status)
{
  switch (status) {
    case simplex_status_t::OPTIMAL: return lp_overlay_status_t::optimal;
    case simplex_status_t::INFEASIBLE: return lp_overlay_status_t::infeasible;
    case simplex_status_t::UNBOUNDED: return lp_overlay_status_t::unbounded;
    case simplex_status_t::UNBOUNDED_OR_INFEASIBLE:
      return lp_overlay_status_t::unbounded_or_infeasible;
    case simplex_status_t::ITERATION_LIMIT: return lp_overlay_status_t::iteration_limit;
    case simplex_status_t::TIME_LIMIT: return lp_overlay_status_t::time_limit;
    case simplex_status_t::NUMERICAL_ISSUES: return lp_overlay_status_t::numerical_issues;
    case simplex_status_t::CUTOFF: return lp_overlay_status_t::cutoff;
    case simplex_status_t::CONCURRENT_LIMIT: return lp_overlay_status_t::concurrent_limit;
    case simplex_status_t::WORK_LIMIT: return lp_overlay_status_t::work_limit;
    case simplex_status_t::UNSET: return lp_overlay_status_t::error;
  }
  return lp_overlay_status_t::error;
}

std::optional<double> finite_value(double value)
{
  return std::isfinite(value) ? std::optional<double>{value} : std::nullopt;
}

lp_row_summary_t summarize_rows(const std::vector<const lp_row_record_t*>& rows,
                                double activity_tolerance,
                                double zero_tolerance)
{
  lp_row_summary_t summary;
  summary.rows = rows.size();
  std::vector<double> distances;
  std::vector<double> signed_duals;
  std::vector<double> absolute_duals;
  distances.reserve(rows.size());
  signed_duals.reserve(rows.size());
  absolute_duals.reserve(rows.size());
  for (const auto* row : rows) {
    summary.active_rows += static_cast<std::size_t>(row->active);
    summary.violated_rows += static_cast<std::size_t>(row->violation > activity_tolerance);
    summary.nonzero_duals += static_cast<std::size_t>(std::abs(row->dual) > zero_tolerance);
    if (row->nearest_bound_distance) { distances.push_back(*row->nearest_bound_distance); }
    signed_duals.push_back(row->dual);
    absolute_duals.push_back(std::abs(row->dual));
  }
  summary.nearest_bound_distance = summarize_values(std::move(distances));
  summary.signed_dual            = summarize_values(std::move(signed_duals));
  summary.absolute_dual          = summarize_values(std::move(absolute_duals));
  return summary;
}

lp_reduced_cost_family_summary_t summarize_reduced_cost_family(
  std::string family, const std::vector<double>& reduced_costs, double zero_tolerance)
{
  lp_reduced_cost_family_summary_t summary;
  summary.family    = std::move(family);
  summary.variables = reduced_costs.size();
  std::vector<double> absolute;
  absolute.reserve(reduced_costs.size());
  for (const double reduced_cost : reduced_costs) {
    if (reduced_cost < -zero_tolerance) {
      ++summary.negative_reduced_costs;
    } else if (reduced_cost > zero_tolerance) {
      ++summary.positive_reduced_costs;
    } else {
      ++summary.zero_reduced_costs;
    }
    absolute.push_back(std::abs(reduced_cost));
  }
  summary.signed_reduced_cost   = summarize_values(reduced_costs);
  summary.absolute_reduced_cost = summarize_values(std::move(absolute));
  return summary;
}

double compute_source_objective(const structure_model_t& model, const std::vector<double>& x)
{
  const auto& objective = model.get_objective_coefficients();
  const auto count      = std::min(objective.size(), x.size());
  const double linear =
    std::inner_product(objective.begin(), objective.begin() + count, x.begin(), 0.0);
  return model.get_objective_scaling_factor() * (linear + model.get_objective_offset());
}

void collect_variable_records(const structure_model_t& model,
                              const model_analysis_t& analysis,
                              const lp_overlay_options_t& options,
                              lp_overlay_summary_t& summary)
{
  const auto n_columns  = model.get_n_variables();
  const auto& objective = model.get_objective_coefficients();
  std::vector<bool> exact_one_member(static_cast<std::size_t>(n_columns), false);
  for (const auto& group : analysis.exact_one_groups) {
    for (const auto& member : group.members) {
      if (member.column >= 0 && member.column < n_columns) {
        exact_one_member[static_cast<std::size_t>(member.column)] = true;
      }
    }
  }

  std::vector<double> fractional_parts;
  std::vector<double> distances;
  std::map<std::string, std::vector<double>> reduced_costs_by_family;
  summary.variables.reserve(static_cast<std::size_t>(n_columns));
  for (structure_index_t column = 0; column < n_columns; ++column) {
    const auto offset  = static_cast<std::size_t>(column);
    const char type    = source_type(model, column);
    const bool binary  = is_binary_column(model, column);
    const bool integer = is_integer_type(type);

    std::string family;
    if (binary && exact_one_member[offset]) {
      family = "binary_exact_one";
    } else if (binary) {
      family = "binary_other";
    } else if (integer) {
      family = "general_integer";
    } else {
      family = "continuous";
    }

    lp_variable_record_t record;
    record.column = column;
    record.original_column =
      offset < analysis.original_column_ids.size() ? analysis.original_column_ids[offset] : column;
    record.name        = variable_name(model, column);
    record.source_type = type;
    record.family      = family;
    record.value       = summary.primal_values[offset];
    record.source_objective_coefficient =
      model.get_objective_scaling_factor() * (offset < objective.size() ? objective[offset] : 0.0);
    record.reduced_cost = summary.reduced_costs[offset];

    if (integer) {
      const double fractional_part   = record.value - std::floor(record.value);
      const double distance          = std::abs(record.value - std::round(record.value));
      record.fractional_part         = fractional_part;
      record.distance_to_integrality = distance;
      record.fractional              = distance > options.integrality_tolerance;
      fractional_parts.push_back(fractional_part);
      distances.push_back(distance);
      ++summary.integer_variables;
      summary.fractional_integer_variables += static_cast<std::size_t>(record.fractional);
      if (binary) {
        ++summary.binary_variables;
        summary.fractional_binary_variables += static_cast<std::size_t>(record.fractional);
      }
    }
    reduced_costs_by_family[family].push_back(record.reduced_cost);
    summary.variables.push_back(std::move(record));
  }

  summary.fractional_part_histogram =
    make_histogram("fractional_part",
                   fractional_parts,
                   {1e-9, 1e-6, 1e-4, 1e-2, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0});
  summary.distance_to_integrality_histogram = make_histogram(
    "distance_to_nearest_integer", distances, {1e-9, 1e-6, 1e-4, 1e-2, 0.10, 0.25, 0.50});

  for (auto& [family, reduced_costs] : reduced_costs_by_family) {
    summary.reduced_cost_family_summaries.push_back(
      summarize_reduced_cost_family(family, reduced_costs, options.zero_tolerance));
  }
}

void collect_row_records(const structure_model_t& model,
                         const model_analysis_t& analysis,
                         const lp_overlay_options_t& options,
                         lp_overlay_summary_t& summary)
{
  const auto n_rows   = model.get_n_constraints();
  const auto& offsets = model.get_constraint_matrix_offsets();
  const auto& indices = model.get_constraint_matrix_indices();
  const auto& values  = model.get_constraint_matrix_values();
  const auto& lower   = model.get_constraint_lower_bounds();
  const auto& upper   = model.get_constraint_upper_bounds();
  std::vector<double> activities(static_cast<std::size_t>(n_rows), 0.0);
  for (structure_index_t row = 0; row < n_rows; ++row) {
    const auto begin = offsets[static_cast<std::size_t>(row)];
    const auto end   = offsets[static_cast<std::size_t>(row) + 1];
    for (structure_index_t entry = begin; entry < end; ++entry) {
      const auto offset = static_cast<std::size_t>(entry);
      activities[static_cast<std::size_t>(row)] +=
        values[offset] * summary.primal_values[static_cast<std::size_t>(indices[offset])];
    }
  }

  std::vector<const row_record_t*> structural_rows(static_cast<std::size_t>(n_rows), nullptr);
  for (const auto& record : analysis.rows) {
    if (record.row >= 0 && record.row < n_rows) {
      structural_rows[static_cast<std::size_t>(record.row)] = &record;
    }
  }

  summary.row_records.reserve(static_cast<std::size_t>(n_rows));
  for (structure_index_t row = 0; row < n_rows; ++row) {
    const auto offset = static_cast<std::size_t>(row);
    lp_row_record_t record;
    record.row      = row;
    record.name     = row_name(model, row);
    record.activity = activities[offset];
    record.dual     = summary.row_duals[offset];
    if (const auto* structural = structural_rows[offset]) {
      record.stable_id = structural->stable_id;
      record.domain    = structural->domain;
      record.families  = structural->families;
    } else {
      record.stable_id = "row:" + std::to_string(row);
      record.domain    = "unclassified";
    }

    if (offset < lower.size() && std::isfinite(lower[offset])) {
      record.lower_bound = lower[offset];
      record.lower_slack = record.activity - lower[offset];
    }
    if (offset < upper.size() && std::isfinite(upper[offset])) {
      record.upper_bound = upper[offset];
      record.upper_slack = upper[offset] - record.activity;
    }

    double nearest = infinity;
    if (record.lower_slack) {
      nearest          = std::min(nearest, std::abs(*record.lower_slack));
      record.violation = std::max(record.violation, std::max(0.0, -*record.lower_slack));
    }
    if (record.upper_slack) {
      nearest          = std::min(nearest, std::abs(*record.upper_slack));
      record.violation = std::max(record.violation, std::max(0.0, -*record.upper_slack));
    }
    if (std::isfinite(nearest)) {
      record.nearest_bound_distance = nearest;
      record.active                 = nearest <= options.activity_tolerance;
    }
    summary.row_records.push_back(std::move(record));
  }

  std::vector<const lp_row_record_t*> all_rows;
  std::map<std::string, std::vector<const lp_row_record_t*>> rows_by_family;
  all_rows.reserve(summary.row_records.size());
  for (const auto& row : summary.row_records) {
    all_rows.push_back(&row);
    rows_by_family["domain:" + row.domain].push_back(&row);
    for (const auto& family : row.families) {
      rows_by_family[family].push_back(&row);
    }
  }
  summary.rows = summarize_rows(all_rows, options.activity_tolerance, options.zero_tolerance);
  for (const auto& [family, rows] : rows_by_family) {
    summary.row_family_summaries.push_back(lp_row_family_summary_t{
      family, summarize_rows(rows, options.activity_tolerance, options.zero_tolerance)});
  }
}

void collect_exact_one_records(const model_analysis_t& analysis,
                               const lp_overlay_options_t& options,
                               lp_overlay_summary_t& summary)
{
  std::vector<double> entropies;
  std::vector<double> effective_supports;
  std::vector<double> largest_values;
  std::vector<double> second_values;
  std::vector<double> margins;
  summary.exact_one_groups.reserve(analysis.exact_one_groups.size());

  for (const auto& group : analysis.exact_one_groups) {
    lp_exact_one_group_record_t record;
    record.group     = group.id;
    record.row       = group.row;
    record.stable_id = group.stable_id;
    record.members   = group.members;
    record.literal_values.reserve(group.members.size());
    std::vector<std::pair<double, std::size_t>> ordered;
    ordered.reserve(group.members.size());
    bool nearly_integral = true;
    for (std::size_t member_index = 0; member_index < group.members.size(); ++member_index) {
      const auto& member = group.members[member_index];
      if (member.column < 0 ||
          static_cast<std::size_t>(member.column) >= summary.primal_values.size()) {
        record.literal_values.push_back(0.0);
        nearly_integral = false;
        continue;
      }
      const double column_value  = summary.primal_values[static_cast<std::size_t>(member.column)];
      const double literal_value = member.value ? column_value : 1.0 - column_value;
      record.literal_values.push_back(literal_value);
      record.literal_sum += literal_value;
      ordered.emplace_back(literal_value, member_index);
      const double distance = std::min(std::abs(literal_value), std::abs(1.0 - literal_value));
      record.fractional_members +=
        static_cast<std::size_t>(distance > options.integrality_tolerance);
      nearly_integral = nearly_integral && distance <= options.near_integrality_tolerance;
    }

    std::stable_sort(ordered.begin(), ordered.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.first > rhs.first;
    });
    if (!ordered.empty()) {
      record.largest_value   = ordered[0].first;
      record.leading_literal = group.members[ordered[0].second];
    }
    if (ordered.size() > 1) {
      record.second_largest_value = ordered[1].first;
      record.second_literal       = group.members[ordered[1].second];
    }
    record.margin = record.largest_value - record.second_largest_value;

    const double positive_sum =
      std::accumulate(record.literal_values.begin(),
                      record.literal_values.end(),
                      0.0,
                      [](double sum, double value) { return sum + std::max(0.0, value); });
    if (positive_sum > options.zero_tolerance) {
      for (const double value : record.literal_values) {
        const double probability = std::max(0.0, value) / positive_sum;
        if (probability > 0.0) { record.entropy -= probability * std::log(probability); }
      }
      record.effective_support = std::exp(record.entropy);
    }

    const bool sum_is_one = std::abs(record.literal_sum - 1.0) <= options.activity_tolerance;
    record.integral       = record.fractional_members == 0 && sum_is_one;
    record.nearly_integral =
      !record.integral && nearly_integral &&
      std::abs(record.literal_sum - 1.0) <= options.near_integrality_tolerance;

    ++summary.exact_one.groups;
    summary.exact_one.integral_groups += static_cast<std::size_t>(record.integral);
    summary.exact_one.nearly_integral_groups += static_cast<std::size_t>(record.nearly_integral);
    summary.exact_one.fractional_groups += static_cast<std::size_t>(!record.integral);
    entropies.push_back(record.entropy);
    effective_supports.push_back(record.effective_support);
    largest_values.push_back(record.largest_value);
    second_values.push_back(record.second_largest_value);
    margins.push_back(record.margin);
    summary.exact_one_groups.push_back(std::move(record));
  }

  summary.exact_one.entropy              = summarize_values(std::move(entropies));
  summary.exact_one.effective_support    = summarize_values(std::move(effective_supports));
  summary.exact_one.largest_value        = summarize_values(std::move(largest_values));
  summary.exact_one.second_largest_value = summarize_values(std::move(second_values));
  summary.exact_one.margin               = summarize_values(std::move(margins));
}

void print_distribution(std::string_view label, const lp_scalar_distribution_t& distribution)
{
  std::cout << label << "n=" << distribution.count;
  if (distribution.count != 0) {
    std::cout << ", min=" << distribution.quantiles.minimum
              << ", q25=" << distribution.quantiles.quartile_1
              << ", median=" << distribution.quantiles.median
              << ", q90=" << distribution.quantiles.percentile_90
              << ", max=" << distribution.quantiles.maximum;
  }
  std::cout << '\n';
}

void print_histogram(const lp_histogram_t& histogram)
{
  std::cout << "    " << histogram.quantity << " histogram:";
  for (const auto& bin : histogram.bins) {
    std::cout << ' ' << (bin.lower_bound_inclusive ? '[' : '(') << bin.lower_bound << ','
              << bin.upper_bound << (bin.upper_bound_inclusive ? ']' : ')') << '=' << bin.count;
  }
  std::cout << '\n';
}

}  // namespace

std::string_view lp_objective_mode_name(lp_objective_mode_t mode)
{
  switch (mode) {
    case lp_objective_mode_t::source: return "source";
    case lp_objective_mode_t::erased: return "erased";
  }
  return "unknown";
}

std::string_view lp_overlay_status_name(lp_overlay_status_t status)
{
  switch (status) {
    case lp_overlay_status_t::not_run: return "NOT_RUN";
    case lp_overlay_status_t::optimal: return "OPTIMAL";
    case lp_overlay_status_t::infeasible: return "INFEASIBLE";
    case lp_overlay_status_t::unbounded: return "UNBOUNDED";
    case lp_overlay_status_t::unbounded_or_infeasible: return "UNBOUNDED_OR_INFEASIBLE";
    case lp_overlay_status_t::iteration_limit: return "ITERATION_LIMIT";
    case lp_overlay_status_t::time_limit: return "TIME_LIMIT";
    case lp_overlay_status_t::numerical_issues: return "NUMERICAL_ISSUES";
    case lp_overlay_status_t::cutoff: return "CUTOFF";
    case lp_overlay_status_t::concurrent_limit: return "CONCURRENT_LIMIT";
    case lp_overlay_status_t::work_limit: return "WORK_LIMIT";
    case lp_overlay_status_t::unsupported: return "UNSUPPORTED";
    case lp_overlay_status_t::invalid_options: return "INVALID_OPTIONS";
    case lp_overlay_status_t::error: return "ERROR";
  }
  return "UNKNOWN";
}

lp_overlay_summary_t analyze_root_lp_overlay(const structure_model_t& model,
                                             const model_analysis_t& analysis,
                                             const lp_overlay_options_t& options)
{
  // RESEARCH-BREADCRUMB(mps-structure/cut-aware-root-state) [solver-invasive]
  // This function is an LP relaxation with no cuts. A cut-aware mode needs a dedicated root
  // observer and explicit post-cut stop; a tree node limit does not provide that contract.
  lp_overlay_summary_t summary;
  summary.objective_mode     = options.objective_mode;
  summary.time_limit_seconds = options.time_limit_seconds;
  summary.iteration_limit    = options.iteration_limit;
  summary.threads            = options.threads;
  summary.cuts_added         = false;
  summary.formulation        = options.objective_mode == lp_objective_mode_t::source
                                 ? "source-objective LP relaxation (integer types relaxed; no cuts)"
                                 : "objective-erased LP relaxation A/B (integer types relaxed; no cuts)";

  if (!std::isfinite(options.time_limit_seconds) || options.time_limit_seconds <= 0.0 ||
      options.iteration_limit <= 0 || options.threads <= 0 ||
      !std::isfinite(options.integrality_tolerance) || options.integrality_tolerance < 0.0 ||
      !std::isfinite(options.near_integrality_tolerance) ||
      options.near_integrality_tolerance < options.integrality_tolerance ||
      !std::isfinite(options.activity_tolerance) || options.activity_tolerance < 0.0 ||
      !std::isfinite(options.zero_tolerance) || options.zero_tolerance < 0.0) {
    summary.status = lp_overlay_status_t::invalid_options;
    summary.detail = "LP overlay limits and tolerances must be finite and positive/nonnegative";
    return summary;
  }
  if (model.has_quadratic_objective() || model.has_quadratic_constraints()) {
    summary.status = lp_overlay_status_t::unsupported;
    summary.detail = "the dual-simplex overlay supports linear MPS models only";
    return summary;
  }
  if (std::find(model.get_variable_types().begin(), model.get_variable_types().end(), 'S') !=
      model.get_variable_types().end()) {
    summary.status = lp_overlay_status_t::unsupported;
    summary.detail = "semi-continuous domains require a separate convex-hull relaxation";
    return summary;
  }

  summary.attempted = true;
  try {
    cpu_optimization_problem_t<structure_index_t, structure_value_t> cpu_problem;
    populate_from_mps_data_model(&cpu_problem, model);
    auto user_problem =
      cuopt_problem_to_user_problem<structure_index_t, structure_value_t>(nullptr, cpu_problem);

    std::fill(user_problem.var_types.begin(),
              user_problem.var_types.end(),
              simplex::variable_type_t::CONTINUOUS);
    if (options.objective_mode == lp_objective_mode_t::source) {
      const double scale     = model.get_objective_scaling_factor();
      user_problem.obj_scale = model.get_sense() ? -scale : scale;
      user_problem.obj_constant =
        model.get_sense() ? -model.get_objective_offset() : model.get_objective_offset();
      if (model.get_sense()) {
        for (auto& coefficient : user_problem.objective) {
          coefficient *= -1.0;
        }
      }
    } else {
      std::fill(user_problem.objective.begin(), user_problem.objective.end(), 0.0);
      user_problem.obj_scale    = 1.0;
      user_problem.obj_constant = 0.0;
    }

    simplex::simplex_solver_settings_t<structure_index_t, structure_value_t> settings;
    settings.time_limit           = options.time_limit_seconds;
    settings.iteration_limit      = options.iteration_limit;
    settings.num_threads          = options.threads;
    settings.barrier              = false;
    settings.print_presolve_stats = false;
    settings.set_log(options.solver_log);

    simplex::lp_solution_t<structure_index_t, structure_value_t> solution(user_problem.num_rows,
                                                                          user_problem.num_cols);
    const double start = tic();
    const auto simplex_status =
      simplex::solve_linear_program(user_problem, settings, start, solution);
    summary.solve_seconds        = toc(start);
    summary.iterations           = solution.iterations;
    summary.status               = translate_status(simplex_status);
    summary.detail               = std::string{simplex::lp_status_to_string(simplex_status)};
    summary.has_optimal_solution = simplex_status == simplex_status_t::OPTIMAL;
    if (!summary.has_optimal_solution) { return summary; }

    summary.relaxation_objective = finite_value(solution.user_objective);
    summary.l2_primal_residual   = finite_value(solution.l2_primal_residual);
    summary.l2_dual_residual     = finite_value(solution.l2_dual_residual);
    summary.primal_values        = std::move(solution.x);
    summary.row_duals            = std::move(solution.y);
    summary.reduced_costs        = std::move(solution.z);

    if (options.objective_mode == lp_objective_mode_t::source) {
      const double source_dual_scale =
        model.get_objective_scaling_factor() * (model.get_sense() ? -1.0 : 1.0);
      for (auto& dual : summary.row_duals) {
        dual *= source_dual_scale;
      }
      for (auto& reduced_cost : summary.reduced_costs) {
        reduced_cost *= source_dual_scale;
      }
    }
    summary.source_objective_at_solution =
      finite_value(compute_source_objective(model, summary.primal_values));

    collect_variable_records(model, analysis, options, summary);
    collect_row_records(model, analysis, options, summary);
    collect_exact_one_records(analysis, options, summary);
  } catch (const std::exception& error) {
    summary.status               = lp_overlay_status_t::error;
    summary.has_optimal_solution = false;
    summary.detail               = error.what();
  } catch (...) {
    summary.status               = lp_overlay_status_t::error;
    summary.has_optimal_solution = false;
    summary.detail               = "unknown exception while solving the LP relaxation";
  }
  return summary;
}

void print_root_lp_overlay_summary(const lp_overlay_summary_t& summary)
{
  const auto old_flags     = std::cout.flags();
  const auto old_precision = std::cout.precision();
  std::cout << "  root LP structural overlay:\n"
            << "    formulation: " << summary.formulation << '\n'
            << "    status: " << lp_overlay_status_name(summary.status);
  if (!summary.detail.empty()) { std::cout << " (" << summary.detail << ')'; }
  std::cout << '\n';
  if (!summary.attempted) {
    std::cout.flags(old_flags);
    std::cout.precision(old_precision);
    return;
  }
  std::cout << "    limits: " << summary.time_limit_seconds << " seconds, "
            << summary.iteration_limit << " iterations, " << summary.threads << " thread(s)\n"
            << "    solve: " << std::fixed << std::setprecision(3) << summary.solve_seconds
            << " seconds, " << summary.iterations << " iterations\n";
  if (!summary.has_optimal_solution) {
    std::cout << "    structural diagnostics omitted because no certified optimal LP solution was "
                 "returned\n";
    std::cout.flags(old_flags);
    std::cout.precision(old_precision);
    return;
  }

  std::cout << std::scientific << std::setprecision(6);
  if (summary.relaxation_objective) {
    std::cout << "    relaxation objective: " << *summary.relaxation_objective << '\n';
  }
  if (summary.source_objective_at_solution) {
    std::cout << "    source objective at solution: " << *summary.source_objective_at_solution
              << '\n';
  }
  std::cout << "    integer variables: " << summary.integer_variables
            << " (fractional=" << summary.fractional_integer_variables
            << "), binary=" << summary.binary_variables
            << " (fractional=" << summary.fractional_binary_variables << ")\n";
  print_histogram(summary.fractional_part_histogram);
  print_histogram(summary.distance_to_integrality_histogram);

  std::cout << "    exact-one groups: " << summary.exact_one.groups
            << " (integral=" << summary.exact_one.integral_groups
            << ", nearly integral=" << summary.exact_one.nearly_integral_groups
            << ", fractional=" << summary.exact_one.fractional_groups << ")\n";
  print_distribution("    exact-one entropy: ", summary.exact_one.entropy);
  print_distribution("    exact-one effective support: ", summary.exact_one.effective_support);
  print_distribution("    exact-one top-two margin: ", summary.exact_one.margin);

  std::cout << "    rows: active=" << summary.rows.active_rows << '/' << summary.rows.rows
            << ", violated=" << summary.rows.violated_rows
            << ", nonzero dual=" << summary.rows.nonzero_duals << '\n';
  print_distribution("    nearest-bound distance: ", summary.rows.nearest_bound_distance);
  print_distribution("    absolute row dual: ", summary.rows.absolute_dual);

  std::cout << "    reduced costs by variable family:\n";
  for (const auto& family : summary.reduced_cost_family_summaries) {
    std::cout << "      " << family.family << ": n=" << family.variables
              << ", negative=" << family.negative_reduced_costs
              << ", zero=" << family.zero_reduced_costs
              << ", positive=" << family.positive_reduced_costs;
    if (family.absolute_reduced_cost.count != 0) {
      std::cout << ", |rc| median=" << family.absolute_reduced_cost.quantiles.median
                << ", q90=" << family.absolute_reduced_cost.quantiles.percentile_90
                << ", max=" << family.absolute_reduced_cost.quantiles.maximum;
    }
    std::cout << '\n';
  }

  std::vector<const lp_exact_one_group_record_t*> ambiguous_groups;
  for (const auto& group : summary.exact_one_groups) {
    if (!group.integral) { ambiguous_groups.push_back(&group); }
  }
  constexpr std::size_t display_limit = 5;
  const auto count                    = std::min(display_limit, ambiguous_groups.size());
  std::partial_sort(ambiguous_groups.begin(),
                    ambiguous_groups.begin() + static_cast<std::ptrdiff_t>(count),
                    ambiguous_groups.end(),
                    [](const auto* lhs, const auto* rhs) {
                      if (lhs->entropy != rhs->entropy) { return lhs->entropy > rhs->entropy; }
                      return lhs->group < rhs->group;
                    });
  if (count != 0) {
    std::cout << "    highest-entropy fractional exact-one groups:\n";
    for (std::size_t rank = 0; rank < count; ++rank) {
      const auto& group = *ambiguous_groups[rank];
      std::cout << "      " << group.stable_id << ": size=" << group.members.size()
                << ", entropy=" << group.entropy
                << ", effective support=" << group.effective_support
                << ", top=" << group.largest_value << ", second=" << group.second_largest_value
                << ", margin=" << group.margin << '\n';
    }
  }
  std::cout.flags(old_flags);
  std::cout.precision(old_precision);
}

}  // namespace cuopt::mathematical_optimization::examples
