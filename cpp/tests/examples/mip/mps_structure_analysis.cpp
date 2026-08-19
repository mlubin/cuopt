/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "mps_structure_analysis.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::examples {
namespace {

constexpr double absolute_tolerance = 1e-9;
constexpr double relative_tolerance = 1e-9;
constexpr std::size_t display_limit = 5;

class disjoint_set_t {
 public:
  explicit disjoint_set_t(std::size_t size) : parent_(size), rank_(size, 0)
  {
    std::iota(parent_.begin(), parent_.end(), std::size_t{0});
  }

  std::size_t find(std::size_t node)
  {
    if (parent_[node] != node) { parent_[node] = find(parent_[node]); }
    return parent_[node];
  }

  void unite(std::size_t lhs, std::size_t rhs)
  {
    lhs = find(lhs);
    rhs = find(rhs);
    if (lhs == rhs) { return; }
    if (rank_[lhs] < rank_[rhs]) { std::swap(lhs, rhs); }
    parent_[rhs] = lhs;
    if (rank_[lhs] == rank_[rhs]) { ++rank_[lhs]; }
  }

 private:
  std::vector<std::size_t> parent_;
  std::vector<unsigned char> rank_;
};

bool nearly_equal(double lhs, double rhs)
{
  const auto scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
  return std::abs(lhs - rhs) <= absolute_tolerance + relative_tolerance * scale;
}

bool is_integral_type(char type) { return type == 'I' || type == 'B'; }

bool is_binary(const structure_model_t& model, structure_index_t column)
{
  const auto& types = model.get_variable_types();
  const auto& lower = model.get_variable_lower_bounds();
  const auto& upper = model.get_variable_upper_bounds();
  if (column < 0 || column >= static_cast<structure_index_t>(types.size()) ||
      !is_integral_type(types[column])) {
    return false;
  }
  const auto lower_is_binary = nearly_equal(lower[column], 0.0) || nearly_equal(lower[column], 1.0);
  const auto upper_is_binary = nearly_equal(upper[column], 0.0) || nearly_equal(upper[column], 1.0);
  return lower_is_binary && upper_is_binary && lower[column] <= upper[column] + absolute_tolerance;
}

bool is_fixed(const structure_model_t& model, structure_index_t column)
{
  const auto& lower = model.get_variable_lower_bounds();
  const auto& upper = model.get_variable_upper_bounds();
  return column >= 0 && static_cast<std::size_t>(column) < lower.size() &&
         static_cast<std::size_t>(column) < upper.size() && std::isfinite(lower[column]) &&
         std::isfinite(upper[column]) && nearly_equal(lower[column], upper[column]);
}

bool row_is_equality(const structure_model_t& model, structure_index_t row)
{
  const auto lower = model.get_constraint_lower_bounds()[row];
  const auto upper = model.get_constraint_upper_bounds()[row];
  return std::isfinite(lower) && std::isfinite(upper) && nearly_equal(lower, upper);
}

std::string stable_row_id(std::string_view scope, structure_index_t row)
{
  return std::string(scope) + ":r" + std::to_string(row);
}

std::string row_name(const structure_model_t& model, structure_index_t row)
{
  const auto& names = model.get_row_names();
  return row < static_cast<structure_index_t>(names.size()) ? names[row] : std::string{};
}

std::string column_name(const structure_model_t& model, structure_index_t column)
{
  const auto& names = model.get_variable_names();
  if (column < static_cast<structure_index_t>(names.size()) && !names[column].empty()) {
    return names[column];
  }
  return "x[" + std::to_string(column) + "]";
}

std::string canonical_double(double value)
{
  if (std::isnan(value)) { return "nan"; }
  if (value == std::numeric_limits<double>::infinity()) { return "+inf"; }
  if (value == -std::numeric_limits<double>::infinity()) { return "-inf"; }
  if (value == 0.0) { value = 0.0; }
  std::ostringstream out;
  out << std::scientific << std::setprecision(17) << value;
  return out.str();
}

std::string fingerprint(const std::string& key)
{
  std::uint64_t hash = 1469598103934665603ULL;
  for (const auto byte : key) {
    hash ^= static_cast<unsigned char>(byte);
    hash *= 1099511628211ULL;
  }
  std::ostringstream out;
  out << std::hex << std::setfill('0') << std::setw(16) << hash;
  return out.str();
}

std::string join_key(const std::vector<std::string>& parts)
{
  std::ostringstream out;
  for (const auto& part : parts) {
    out << part.size() << ':' << part << ';';
  }
  return out.str();
}

void add_family(model_analysis_t& analysis, row_record_t& row, std::string family)
{
  if (std::find(row.families.begin(), row.families.end(), family) != row.families.end()) { return; }
  ++analysis.row_family_counts[family];
  row.families.push_back(std::move(family));
}

row_group_t make_group(structure_index_t id,
                       structure_index_t row,
                       std::string stable_id,
                       std::string family,
                       const std::vector<literal_t>& members,
                       const std::vector<double>& coefficients,
                       double normalized_lower,
                       double normalized_upper,
                       double scale)
{
  if (!std::isfinite(scale) || scale <= 0.0) {
    throw std::logic_error("row-group normalization requires a positive finite scale");
  }
  row_group_t group;
  group.id        = id;
  group.row       = row;
  group.stable_id = std::move(stable_id);
  group.family    = std::move(family);
  group.members   = members;
  group.member_coefficients.reserve(coefficients.size());
  std::transform(coefficients.begin(),
                 coefficients.end(),
                 std::back_inserter(group.member_coefficients),
                 [scale](double coefficient) { return coefficient / scale; });
  group.has_lower_capacity = std::isfinite(normalized_lower);
  group.has_upper_capacity = std::isfinite(normalized_upper);
  if (group.has_lower_capacity) { group.lower_capacity = normalized_lower / scale; }
  if (group.has_upper_capacity) { group.upper_capacity = normalized_upper / scale; }
  return group;
}

bool assignment_satisfies(const structure_model_t& model,
                          structure_index_t row,
                          structure_index_t first_value,
                          structure_index_t second_value)
{
  const auto& offsets   = model.get_constraint_matrix_offsets();
  const auto& values    = model.get_constraint_matrix_values();
  const auto begin      = offsets[row];
  const double activity = values[begin] * first_value + values[begin + 1] * second_value;
  const auto lower      = model.get_constraint_lower_bounds()[row];
  const auto upper      = model.get_constraint_upper_bounds()[row];
  return (!std::isfinite(lower) || activity >= lower - absolute_tolerance) &&
         (!std::isfinite(upper) || activity <= upper + absolute_tolerance);
}

void detect_binary_implications(const structure_model_t& model,
                                structure_index_t row,
                                model_analysis_t& analysis,
                                row_record_t& record)
{
  const auto& offsets = model.get_constraint_matrix_offsets();
  const auto& indices = model.get_constraint_matrix_indices();
  const auto begin    = offsets[row];
  const auto end      = offsets[row + 1];
  if (end - begin != 2 || !is_binary(model, indices[begin]) ||
      !is_binary(model, indices[begin + 1])) {
    return;
  }

  const auto first  = indices[begin];
  const auto second = indices[begin + 1];
  std::size_t found = 0;
  for (structure_index_t first_value = 0; first_value <= 1; ++first_value) {
    for (structure_index_t second_value = 0; second_value <= 1; ++second_value) {
      if (assignment_satisfies(model, row, first_value, second_value)) { continue; }
      analysis.implications.push_back(
        implication_t{row, {first, first_value != 0}, {second, second_value == 0}});
      analysis.implications.push_back(
        implication_t{row, {second, second_value != 0}, {first, first_value == 0}});
      ++found;
    }
  }
  if (found != 0) { add_family(analysis, record, "binary_implication"); }
}

void add_variable_bound(structure_index_t row,
                        structure_index_t binary,
                        structure_index_t target,
                        double target_coefficient,
                        double binary_coefficient,
                        double rhs,
                        bool is_upper_side,
                        const structure_model_t& model,
                        model_analysis_t& analysis)
{
  variable_bound_t relation;
  relation.row             = row;
  relation.binary_column   = binary;
  relation.target_column   = target;
  relation.bound_when_zero = rhs / target_coefficient;
  relation.bound_when_one  = (rhs - binary_coefficient) / target_coefficient;
  const bool upper_bound   = (target_coefficient > 0.0) == is_upper_side;
  relation.bound_type      = upper_bound ? "upper" : "lower";
  const auto& lower        = model.get_variable_lower_bounds();
  const auto& upper        = model.get_variable_upper_bounds();
  const bool zero_at_off   = nearly_equal(relation.bound_when_zero, 0.0);
  const bool fixes_at_zero = (upper_bound && lower[target] >= -absolute_tolerance) ||
                             (!upper_bound && upper[target] <= absolute_tolerance);
  const bool enables_at_one =
    upper_bound ? relation.bound_when_one > relation.bound_when_zero + absolute_tolerance
                : relation.bound_when_one < relation.bound_when_zero - absolute_tolerance;
  relation.activation = zero_at_off && fixes_at_zero && enables_at_one;
  analysis.variable_bounds.push_back(relation);
}

void detect_variable_bounds(const structure_model_t& model,
                            structure_index_t row,
                            model_analysis_t& analysis,
                            row_record_t& record)
{
  const auto& offsets = model.get_constraint_matrix_offsets();
  const auto& indices = model.get_constraint_matrix_indices();
  const auto& values  = model.get_constraint_matrix_values();
  const auto begin    = offsets[row];
  if (offsets[row + 1] - begin != 2) { return; }

  const bool first_binary  = is_binary(model, indices[begin]);
  const bool second_binary = is_binary(model, indices[begin + 1]);
  if (first_binary == second_binary) { return; }
  const auto binary_position = first_binary ? begin : begin + 1;
  const auto target_position = first_binary ? begin + 1 : begin;
  const auto binary          = indices[binary_position];
  const auto target          = indices[target_position];
  const auto target_coeff    = values[target_position];
  const auto binary_coeff    = values[binary_position];
  const auto lower           = model.get_constraint_lower_bounds()[row];
  const auto upper           = model.get_constraint_upper_bounds()[row];
  if (std::isfinite(upper)) {
    add_variable_bound(
      row, binary, target, target_coeff, binary_coeff, upper, true, model, analysis);
  }
  if (std::isfinite(lower)) {
    add_variable_bound(
      row, binary, target, target_coeff, binary_coeff, lower, false, model, analysis);
  }
  add_family(analysis, record, "variable_bound");
  if (std::any_of(
        analysis.variable_bounds.rbegin(),
        analysis.variable_bounds.rend(),
        [row](const auto& relation) { return relation.row == row && relation.activation; })) {
    add_family(analysis, record, "activation");
  }
}

void detect_affine_definition(const structure_model_t& model,
                              structure_index_t row,
                              model_analysis_t& analysis,
                              row_record_t& record)
{
  if (!row_is_equality(model, row)) { return; }
  const auto& offsets = model.get_constraint_matrix_offsets();
  const auto& indices = model.get_constraint_matrix_indices();
  const auto& types   = model.get_variable_types();
  const auto& lower   = model.get_variable_lower_bounds();
  const auto& upper   = model.get_variable_upper_bounds();
  const auto begin    = offsets[row];
  const auto end      = offsets[row + 1];

  affine_definition_t definition;
  definition.row          = row;
  definition.singleton    = end - begin == 1;
  definition.two_variable = end - begin == 2;
  for (auto entry = begin; entry < end; ++entry) {
    const auto column = indices[entry];
    if (!is_integral_type(types[column]) && !std::isfinite(lower[column]) &&
        !std::isfinite(upper[column])) {
      definition.pivot_columns.push_back(column);
    }
  }
  if (definition.singleton || definition.two_variable || !definition.pivot_columns.empty()) {
    analysis.affine_definitions.push_back(definition);
    add_family(analysis, record, "affine_definition");
    if (definition.singleton) { add_family(analysis, record, "singleton_equality"); }
  }
}

std::string classify_row_domain(const structure_model_t& model, structure_index_t row)
{
  const auto& offsets = model.get_constraint_matrix_offsets();
  const auto& indices = model.get_constraint_matrix_indices();
  const auto& types   = model.get_variable_types();
  bool has_integer    = false;
  bool has_continuous = false;
  for (auto entry = offsets[row]; entry < offsets[row + 1]; ++entry) {
    if (is_integral_type(types[indices[entry]])) {
      has_integer = true;
    } else {
      has_continuous = true;
    }
  }
  if (!has_integer && !has_continuous) { return "empty"; }
  if (has_integer && has_continuous) { return "mixed_response"; }
  return has_integer ? "integer_only" : "continuous_only";
}

void detect_binary_row_families(const structure_model_t& model,
                                structure_index_t row,
                                model_analysis_t& analysis,
                                row_record_t& record)
{
  const auto& offsets = model.get_constraint_matrix_offsets();
  const auto& indices = model.get_constraint_matrix_indices();
  const auto& values  = model.get_constraint_matrix_values();
  const auto begin    = offsets[row];
  const auto end      = offsets[row + 1];
  if (end == begin) { return; }

  std::vector<literal_t> literals;
  std::vector<double> magnitudes;
  literals.reserve(end - begin);
  magnitudes.reserve(end - begin);
  double shift         = 0.0;
  bool all_binary      = true;
  bool equal_magnitude = true;
  double first_abs     = std::abs(values[begin]);
  for (auto entry = begin; entry < end; ++entry) {
    const auto column = indices[entry];
    if (!is_binary(model, column)) {
      all_binary = false;
      break;
    }
    const auto coefficient = values[entry];
    const auto magnitude   = std::abs(coefficient);
    literals.push_back(literal_t{column, coefficient > 0.0});
    magnitudes.push_back(magnitude);
    if (coefficient < 0.0) { shift += magnitude; }
    if (!nearly_equal(magnitude, first_abs)) { equal_magnitude = false; }
  }
  if (!all_binary || first_abs <= absolute_tolerance) { return; }
  record.binary_literals = literals;

  const auto lower            = model.get_constraint_lower_bounds()[row];
  const auto upper            = model.get_constraint_upper_bounds()[row];
  const auto normalized_lower = std::isfinite(lower) ? lower + shift : lower;
  const auto normalized_upper = std::isfinite(upper) ? upper + shift : upper;
  const bool exact_one =
    equal_magnitude && std::isfinite(normalized_lower) && std::isfinite(normalized_upper) &&
    nearly_equal(normalized_lower, first_abs) && nearly_equal(normalized_upper, first_abs);
  const bool set_packing =
    equal_magnitude && std::isfinite(normalized_upper) && nearly_equal(normalized_upper, first_abs);
  const bool set_covering =
    equal_magnitude && std::isfinite(normalized_lower) && nearly_equal(normalized_lower, first_abs);

  if (exact_one) {
    add_family(analysis, record, "exact_one");
    analysis.exact_one_groups.push_back(
      make_group(static_cast<structure_index_t>(analysis.exact_one_groups.size()),
                 row,
                 record.stable_id,
                 "exact_one",
                 literals,
                 magnitudes,
                 normalized_lower,
                 normalized_upper,
                 first_abs));
  } else {
    if (set_packing) {
      add_family(analysis, record, "set_packing");
      analysis.set_packing_groups.push_back(
        make_group(static_cast<structure_index_t>(analysis.set_packing_groups.size()),
                   row,
                   record.stable_id,
                   "set_packing",
                   literals,
                   magnitudes,
                   normalized_lower,
                   normalized_upper,
                   first_abs));
    }
    if (set_covering) {
      add_family(analysis, record, "set_covering");
      analysis.set_covering_groups.push_back(
        make_group(static_cast<structure_index_t>(analysis.set_covering_groups.size()),
                   row,
                   record.stable_id,
                   "set_covering",
                   literals,
                   magnitudes,
                   normalized_lower,
                   normalized_upper,
                   first_abs));
    }
    if (equal_magnitude && std::isfinite(normalized_upper) &&
        normalized_upper >= first_abs - absolute_tolerance && !set_packing) {
      add_family(analysis, record, "gub");
      analysis.gub_groups.push_back(
        make_group(static_cast<structure_index_t>(analysis.gub_groups.size()),
                   row,
                   record.stable_id,
                   "gub",
                   literals,
                   magnitudes,
                   normalized_lower,
                   normalized_upper,
                   first_abs));
    }
  }

  const bool has_finite_capacity =
    std::isfinite(normalized_lower) || std::isfinite(normalized_upper);
  const bool specialized_cardinality = exact_one || set_packing || set_covering;
  if (has_finite_capacity && (!equal_magnitude || !specialized_cardinality)) {
    const auto family = equal_magnitude ? "binary_cardinality" : "binary_knapsack";
    add_family(analysis, record, family);
    analysis.knapsack_rows.push_back(
      make_group(static_cast<structure_index_t>(analysis.knapsack_rows.size()),
                 row,
                 record.stable_id,
                 family,
                 literals,
                 magnitudes,
                 normalized_lower,
                 normalized_upper,
                 equal_magnitude ? first_abs : 1.0));
  }
}

}  // namespace

std::string_view analysis_level_name(analysis_level_t level)
{
  switch (level) {
    case analysis_level_t::basic: return "basic";
    case analysis_level_t::symmetry: return "symmetry";
  }
  return "unknown";
}

}  // namespace cuopt::mathematical_optimization::examples

namespace cuopt::mathematical_optimization::examples {
namespace {

std::vector<std::vector<structure_index_t>> build_column_rows(const structure_model_t& model)
{
  std::vector<std::vector<structure_index_t>> column_rows(model.get_n_variables());
  const auto& offsets = model.get_constraint_matrix_offsets();
  const auto& indices = model.get_constraint_matrix_indices();
  for (structure_index_t row = 0; row < model.get_n_constraints(); ++row) {
    for (auto entry = offsets[row]; entry < offsets[row + 1]; ++entry) {
      column_rows[indices[entry]].push_back(row);
    }
  }
  return column_rows;
}

std::vector<std::vector<structure_index_t>> groups_touched_by_rows(
  const structure_model_t& model,
  const std::vector<std::vector<structure_index_t>>& groups_by_column)
{
  std::vector<std::vector<structure_index_t>> touched(model.get_n_constraints());
  const auto& offsets = model.get_constraint_matrix_offsets();
  const auto& indices = model.get_constraint_matrix_indices();
  for (structure_index_t row = 0; row < model.get_n_constraints(); ++row) {
    auto& row_groups = touched[row];
    for (auto entry = offsets[row]; entry < offsets[row + 1]; ++entry) {
      if (is_fixed(model, indices[entry])) { continue; }
      const auto& column_groups = groups_by_column[indices[entry]];
      row_groups.insert(row_groups.end(), column_groups.begin(), column_groups.end());
    }
    std::sort(row_groups.begin(), row_groups.end());
    row_groups.erase(std::unique(row_groups.begin(), row_groups.end()), row_groups.end());
  }
  return touched;
}

void add_edge_provenance(
  std::map<std::pair<structure_index_t, structure_index_t>, block_edge_t>& edges,
  structure_index_t first,
  structure_index_t second,
  structure_index_t provenance,
  std::string_view kind)
{
  if (first == second) { return; }
  if (first > second) { std::swap(first, second); }
  auto& edge   = edges[{first, second}];
  edge.first   = first;
  edge.second  = second;
  auto* values = kind == "row"       ? &edge.direct_rows
                 : kind == "integer" ? &edge.integer_mediators
                                     : &edge.continuous_mediators;
  if (std::find(values->begin(), values->end(), provenance) == values->end()) {
    values->push_back(provenance);
  }
}

struct block_pair_budget_t {
  explicit block_pair_budget_t(std::size_t limit_) : limit(limit_) {}

  std::size_t limit{};
  std::size_t candidate{};
  std::size_t materialized{};
  bool complete{true};
};

std::size_t saturating_add(std::size_t lhs, std::size_t rhs)
{
  const auto maximum = std::numeric_limits<std::size_t>::max();
  return lhs > maximum - rhs ? maximum : lhs + rhs;
}

std::size_t pair_count(std::size_t count)
{
  if (count < 2) { return 0; }
  auto lhs = count;
  auto rhs = count - 1;
  if (lhs % 2 == 0) {
    lhs /= 2;
  } else {
    rhs /= 2;
  }
  const auto maximum = std::numeric_limits<std::size_t>::max();
  return rhs != 0 && lhs > maximum / rhs ? maximum : lhs * rhs;
}

void materialize_relation_pairs(
  std::map<std::pair<structure_index_t, structure_index_t>, block_edge_t>& edges,
  const std::vector<structure_index_t>& groups,
  structure_index_t provenance,
  std::string_view kind,
  block_pair_budget_t& budget)
{
  budget.candidate = saturating_add(budget.candidate, pair_count(groups.size()));
  if (!budget.complete) { return; }

  for (std::size_t first = 0; first < groups.size(); ++first) {
    for (std::size_t second = first + 1; second < groups.size(); ++second) {
      if (budget.materialized >= budget.limit) {
        budget.complete = false;
        return;
      }
      add_edge_provenance(edges, groups[first], groups[second], provenance, kind);
      ++budget.materialized;
    }
  }
}

std::size_t projection_weight(const block_edge_t& edge, std::string_view projection)
{
  if (projection == "direct_rows") { return edge.direct_rows.size(); }
  if (projection == "integer_mediators") { return edge.integer_mediators.size(); }
  if (projection == "continuous_mediators") { return edge.continuous_mediators.size(); }
  if (projection == "either_mediator") {
    return edge.integer_mediators.size() + edge.continuous_mediators.size();
  }
  return edge.direct_rows.size() + edge.integer_mediators.size() + edge.continuous_mediators.size();
}

graph_projection_summary_t summarize_projection(const std::vector<block_edge_t>& edges,
                                                std::size_t vertices,
                                                std::string name,
                                                bool complete)
{
  graph_projection_summary_t summary;
  summary.name     = std::move(name);
  summary.complete = complete;
  if (vertices == 0) { return summary; }

  disjoint_set_t sets(vertices);
  std::vector<std::size_t> weighted_degrees(vertices, 0);
  for (const auto& edge : edges) {
    const auto weight = projection_weight(edge, summary.name);
    if (weight == 0) { continue; }
    ++summary.edges;
    summary.total_weight += weight;
    weighted_degrees[edge.first] += weight;
    weighted_degrees[edge.second] += weight;
    sets.unite(edge.first, edge.second);
  }

  std::unordered_map<std::size_t, std::size_t> component_sizes;
  for (std::size_t vertex = 0; vertex < vertices; ++vertex) {
    ++component_sizes[sets.find(vertex)];
    if (weighted_degrees[vertex] == 0) { ++summary.isolated_vertices; }
  }
  summary.components = component_sizes.size();
  for (const auto& [unused_root, size] : component_sizes) {
    static_cast<void>(unused_root);
    summary.component_sizes.push_back(size);
    summary.largest_component = std::max(summary.largest_component, size);
  }
  std::sort(summary.component_sizes.begin(), summary.component_sizes.end(), std::greater<>());

  for (std::size_t vertex = 0; vertex < vertices; ++vertex) {
    if (weighted_degrees[vertex] != 0) {
      summary.highest_weighted_degrees.emplace_back(static_cast<structure_index_t>(vertex),
                                                    weighted_degrees[vertex]);
    }
  }
  std::partial_sort(
    summary.highest_weighted_degrees.begin(),
    summary.highest_weighted_degrees.begin() +
      static_cast<std::ptrdiff_t>(std::min(display_limit, summary.highest_weighted_degrees.size())),
    summary.highest_weighted_degrees.end(),
    [](const auto& lhs, const auto& rhs) {
      return lhs.second != rhs.second ? lhs.second > rhs.second : lhs.first < rhs.first;
    });
  if (summary.highest_weighted_degrees.size() > display_limit) {
    summary.highest_weighted_degrees.resize(display_limit);
  }
  return summary;
}

void analyze_exact_one_blocks(const structure_model_t& model,
                              model_analysis_t& analysis,
                              std::size_t pair_provenance_limit)
{
  const auto n_columns = model.get_n_variables();
  std::vector<std::vector<structure_index_t>> groups_by_column(n_columns);
  std::vector<bool> covered(n_columns, false);
  std::size_t largest_group = 0;
  for (const auto& group : analysis.exact_one_groups) {
    largest_group = std::max(largest_group, group.members.size());
    for (const auto& member : group.members) {
      groups_by_column[member.column].push_back(group.id);
      covered[member.column] = true;
    }
  }

  analysis.exact_one_group_size_histogram.assign(largest_group + 1, 0);
  for (const auto& group : analysis.exact_one_groups) {
    ++analysis.exact_one_group_size_histogram[group.members.size()];
  }
  for (structure_index_t column = 0; column < n_columns; ++column) {
    if (is_binary(model, column)) {
      ++analysis.binary_variables;
      if (covered[column]) { ++analysis.exact_one_covered_binary_variables; }
    }
    if (groups_by_column[column].size() > 1) { analysis.exact_one_groups_overlap = true; }
  }

  disjoint_set_t overlap_sets(analysis.exact_one_groups.size());
  for (const auto& groups : groups_by_column) {
    for (std::size_t index = 1; index < groups.size(); ++index) {
      overlap_sets.unite(groups[0], groups[index]);
    }
  }
  std::unordered_map<std::size_t, std::size_t> overlap_sizes;
  for (std::size_t group = 0; group < analysis.exact_one_groups.size(); ++group) {
    ++overlap_sizes[overlap_sets.find(group)];
  }
  for (const auto& [unused_root, size] : overlap_sizes) {
    static_cast<void>(unused_root);
    analysis.exact_one_overlap_component_sizes.push_back(size);
  }
  std::sort(analysis.exact_one_overlap_component_sizes.begin(),
            analysis.exact_one_overlap_component_sizes.end(),
            std::greater<>());

  const auto touched_by_row = groups_touched_by_rows(model, groups_by_column);
  std::map<std::pair<structure_index_t, structure_index_t>, block_edge_t> edge_map;
  block_pair_budget_t direct_budget(pair_provenance_limit);
  block_pair_budget_t integer_budget(pair_provenance_limit);
  block_pair_budget_t continuous_budget(pair_provenance_limit);
  for (structure_index_t row = 0; row < model.get_n_constraints(); ++row) {
    const auto& groups = touched_by_row[row];
    materialize_relation_pairs(edge_map, groups, row, "row", direct_budget);
  }

  const auto column_rows = build_column_rows(model);
  const auto& types      = model.get_variable_types();
  for (structure_index_t mediator = 0; mediator < n_columns; ++mediator) {
    if (!groups_by_column[mediator].empty() || is_fixed(model, mediator)) { continue; }
    std::map<structure_index_t, std::vector<structure_index_t>> incident_rows;
    for (const auto row : column_rows[mediator]) {
      for (const auto group : touched_by_row[row]) {
        incident_rows[group].push_back(row);
      }
    }
    if (incident_rows.size() < 2) { continue; }

    const std::string_view kind = is_integral_type(types[mediator]) ? "integer" : "continuous";
    block_mediator_t mediator_record;
    mediator_record.column = mediator;
    mediator_record.kind   = kind;
    std::vector<structure_index_t> groups;
    groups.reserve(incident_rows.size());
    for (auto& [group, rows] : incident_rows) {
      std::sort(rows.begin(), rows.end());
      rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
      groups.push_back(group);
      mediator_record.incidences.push_back(block_mediator_incidence_t{group, std::move(rows)});
    }
    analysis.exact_one_block_mediators.push_back(std::move(mediator_record));
    auto& budget = kind == std::string_view{"integer"} ? integer_budget : continuous_budget;
    materialize_relation_pairs(edge_map, groups, mediator, kind, budget);
  }

  direct_budget.complete =
    direct_budget.complete && direct_budget.materialized == direct_budget.candidate;
  integer_budget.complete =
    integer_budget.complete && integer_budget.materialized == integer_budget.candidate;
  continuous_budget.complete =
    continuous_budget.complete && continuous_budget.materialized == continuous_budget.candidate;
  analysis.exact_one_block_pair_provenance_limit = pair_provenance_limit;
  analysis.exact_one_block_edges_complete =
    direct_budget.complete && integer_budget.complete && continuous_budget.complete;
  analysis.exact_one_block_candidate_pair_provenance = {
    {"direct_rows", direct_budget.candidate},
    {"integer_mediators", integer_budget.candidate},
    {"continuous_mediators", continuous_budget.candidate}};
  analysis.exact_one_block_materialized_pair_provenance = {
    {"direct_rows", direct_budget.materialized},
    {"integer_mediators", integer_budget.materialized},
    {"continuous_mediators", continuous_budget.materialized}};

  for (auto& [unused_pair, edge] : edge_map) {
    static_cast<void>(unused_pair);
    std::sort(edge.direct_rows.begin(), edge.direct_rows.end());
    std::sort(edge.integer_mediators.begin(), edge.integer_mediators.end());
    std::sort(edge.continuous_mediators.begin(), edge.continuous_mediators.end());
    analysis.exact_one_block_edges.push_back(std::move(edge));
  }
  for (const auto name : {"direct_rows",
                          "integer_mediators",
                          "continuous_mediators",
                          "either_mediator",
                          "all_coupling"}) {
    const bool complete = name == std::string_view{"direct_rows"}         ? direct_budget.complete
                          : name == std::string_view{"integer_mediators"} ? integer_budget.complete
                          : name == std::string_view{"continuous_mediators"}
                            ? continuous_budget.complete
                          : name == std::string_view{"either_mediator"}
                            ? integer_budget.complete && continuous_budget.complete
                            : analysis.exact_one_block_edges_complete;
    analysis.exact_one_block_projections.push_back(summarize_projection(
      analysis.exact_one_block_edges, analysis.exact_one_groups.size(), name, complete));
  }
}

std::string interface_classification(std::size_t width)
{
  if (width <= 1) { return "local"; }
  if (width == 2) { return "coupling"; }
  return "global_interface";
}

std::vector<structure_index_t> assign_colors(const std::vector<std::string>& keys);

struct decomposition_refinement_t {
  std::vector<structure_index_t> row_colors;
  std::vector<structure_index_t> column_colors;
};

decomposition_refinement_t refine_decomposition(const structure_model_t& model,
                                                const std::vector<bool>& seed_rows,
                                                const std::vector<bool>& eligible_columns)
{
  const auto n_rows    = model.get_n_constraints();
  const auto n_columns = model.get_n_variables();
  const auto& offsets  = model.get_constraint_matrix_offsets();
  const auto& indices  = model.get_constraint_matrix_indices();
  const auto& values   = model.get_constraint_matrix_values();
  const auto& types    = model.get_variable_types();

  std::vector<double> row_scales(n_rows, 1.0);
  std::vector<std::string> row_base_keys(n_rows, "inactive-row");
  std::vector<std::string> column_base_keys(n_columns, "inactive-column");
  std::vector<std::vector<std::pair<structure_index_t, double>>> column_edges(n_columns);
  for (structure_index_t row = 0; row < n_rows; ++row) {
    if (!seed_rows[row]) { continue; }
    double scale = 0.0;
    for (auto entry = offsets[row]; entry < offsets[row + 1]; ++entry) {
      if (eligible_columns[indices[entry]]) { scale = std::max(scale, std::abs(values[entry])); }
    }
    if (scale == 0.0) { continue; }
    row_scales[row] = scale;
    row_base_keys[row] =
      join_key({"row",
                "lb:" + canonical_double(model.get_constraint_lower_bounds()[row] / scale),
                "ub:" + canonical_double(model.get_constraint_upper_bounds()[row] / scale)});
    for (auto entry = offsets[row]; entry < offsets[row + 1]; ++entry) {
      const auto column = indices[entry];
      if (eligible_columns[column]) {
        column_edges[column].emplace_back(row, values[entry] / scale);
      }
    }
  }
  for (structure_index_t column = 0; column < n_columns; ++column) {
    if (!eligible_columns[column] || column_edges[column].empty()) { continue; }
    column_base_keys[column] =
      join_key({"column",
                std::string("type:") + types[column],
                "lb:" + canonical_double(model.get_variable_lower_bounds()[column]),
                "ub:" + canonical_double(model.get_variable_upper_bounds()[column]),
                "obj:" + canonical_double(model.get_objective_coefficients()[column])});
  }

  auto row_colors              = assign_colors(row_base_keys);
  auto column_colors           = assign_colors(column_base_keys);
  constexpr std::size_t rounds = 3;
  for (std::size_t round = 0; round < rounds; ++round) {
    std::vector<std::string> row_keys(n_rows, "inactive-row");
    std::vector<std::string> column_keys(n_columns, "inactive-column");
    for (structure_index_t row = 0; row < n_rows; ++row) {
      if (!seed_rows[row]) { continue; }
      std::vector<std::string> parts{row_base_keys[row], "self:" + std::to_string(row_colors[row])};
      for (auto entry = offsets[row]; entry < offsets[row + 1]; ++entry) {
        const auto column = indices[entry];
        if (eligible_columns[column]) {
          parts.push_back(canonical_double(values[entry] / row_scales[row]) + "@" +
                          std::to_string(column_colors[column]));
        }
      }
      std::sort(parts.begin() + 2, parts.end());
      row_keys[row] = join_key(parts);
    }
    for (structure_index_t column = 0; column < n_columns; ++column) {
      if (column_edges[column].empty()) { continue; }
      std::vector<std::string> parts{column_base_keys[column],
                                     "self:" + std::to_string(column_colors[column])};
      for (const auto& [row, normalized_coefficient] : column_edges[column]) {
        parts.push_back(canonical_double(normalized_coefficient) + "@" +
                        std::to_string(row_colors[row]));
      }
      std::sort(parts.begin() + 2, parts.end());
      column_keys[column] = join_key(parts);
    }
    row_colors    = assign_colors(row_keys);
    column_colors = assign_colors(column_keys);
  }
  return decomposition_refinement_t{std::move(row_colors), std::move(column_colors)};
}

std::string component_key(const component_t& component,
                          const decomposition_refinement_t& refinement)
{
  std::vector<std::string> colors;
  colors.reserve(component.rows.size() + component.columns.size() + 2);
  for (const auto row : component.rows) {
    colors.push_back("r:" + std::to_string(refinement.row_colors[row]));
  }
  for (const auto column : component.columns) {
    colors.push_back("c:" + std::to_string(refinement.column_colors[column]));
  }
  std::sort(colors.begin(), colors.end());
  colors.push_back("ncols:" + std::to_string(component.columns.size()));
  colors.push_back("nrows:" + std::to_string(component.rows.size()));
  return join_key(colors);
}

decomposition_t build_decomposition(const structure_model_t& model,
                                    std::string name,
                                    const std::vector<bool>& seed_rows,
                                    const std::vector<bool>& eligible_columns)
{
  decomposition_t decomposition;
  decomposition.name   = std::move(name);
  const auto n_columns = model.get_n_variables();
  const auto n_rows    = model.get_n_constraints();
  const auto& offsets  = model.get_constraint_matrix_offsets();
  const auto& indices  = model.get_constraint_matrix_indices();
  disjoint_set_t sets(n_columns);
  std::vector<bool> touched(n_columns, false);

  for (structure_index_t row = 0; row < n_rows; ++row) {
    if (!seed_rows[row]) { continue; }
    structure_index_t first_column = -1;
    for (auto entry = offsets[row]; entry < offsets[row + 1]; ++entry) {
      const auto column = indices[entry];
      if (!eligible_columns[column]) { continue; }
      touched[column] = true;
      if (first_column < 0) {
        first_column = column;
      } else {
        sets.unite(first_column, column);
      }
    }
  }

  std::map<std::size_t, structure_index_t> component_by_root;
  for (structure_index_t column = 0; column < n_columns; ++column) {
    if (!touched[column]) { continue; }
    const auto root = sets.find(column);
    auto [position, inserted] =
      component_by_root.emplace(root, static_cast<structure_index_t>(component_by_root.size()));
    if (inserted) {
      component_t component;
      component.id = position->second;
      decomposition.components.push_back(std::move(component));
    }
    decomposition.components[position->second].columns.push_back(column);
  }

  std::vector<structure_index_t> column_component(n_columns, -1);
  for (const auto& component : decomposition.components) {
    for (const auto column : component.columns) {
      column_component[column] = component.id;
    }
  }
  for (structure_index_t row = 0; row < n_rows; ++row) {
    std::vector<structure_index_t> components;
    for (auto entry = offsets[row]; entry < offsets[row + 1]; ++entry) {
      const auto component = column_component[indices[entry]];
      if (component >= 0) { components.push_back(component); }
    }
    std::sort(components.begin(), components.end());
    components.erase(std::unique(components.begin(), components.end()), components.end());
    if (components.empty()) { continue; }
    if (seed_rows[row]) {
      for (const auto component : components) {
        decomposition.components[component].rows.push_back(row);
      }
    } else {
      decomposition.interface_rows.push_back(
        interface_row_t{row, interface_classification(components.size()), components});
      ++decomposition.interface_width_histogram[components.size()];
    }
  }

  const auto refinement = refine_decomposition(model, seed_rows, eligible_columns);
  std::map<std::string, std::vector<structure_index_t>> classes;
  for (auto& component : decomposition.components) {
    ++decomposition.component_size_histogram[component.columns.size()];
    const auto key                   = component_key(component, refinement);
    component.refinement_fingerprint = fingerprint(key);
    classes[key].push_back(component.id);
  }
  for (const auto& [key, members] : classes) {
    if (members.size() > 1) {
      decomposition.repeated_component_classes[fingerprint(key)] = members;
    }
  }
  return decomposition;
}

void analyze_decompositions(const structure_model_t& model, model_analysis_t& analysis)
{
  // RESEARCH-BREADCRUMB(mps-structure/matrix-separators) [driver-local]
  // Reuse the incidence views here for articulation rows/columns and biconnected components; keep
  // any heuristic partition score separate from graph-theoretic certificates.
  const auto n_rows    = model.get_n_constraints();
  const auto n_columns = model.get_n_variables();
  std::vector<bool> integer_columns(n_columns, false);
  std::vector<bool> continuous_columns(n_columns, false);
  for (structure_index_t column = 0; column < n_columns; ++column) {
    integer_columns[column]    = is_integral_type(model.get_variable_types()[column]);
    continuous_columns[column] = !integer_columns[column];
  }

  std::vector<bool> integer_rows(n_rows, false);
  std::vector<bool> continuous_equality_rows(n_rows, false);
  std::vector<bool> mixed_rows(n_rows, false);
  std::vector<bool> continuous_projection_rows(n_rows, false);
  const auto& offsets = model.get_constraint_matrix_offsets();
  const auto& indices = model.get_constraint_matrix_indices();
  for (structure_index_t row = 0; row < n_rows; ++row) {
    const auto& domain            = analysis.rows[row].domain;
    integer_rows[row]             = domain == "integer_only";
    mixed_rows[row]               = domain == "mixed_response";
    continuous_equality_rows[row] = domain == "continuous_only" && row_is_equality(model, row);
    for (auto entry = offsets[row]; entry < offsets[row + 1]; ++entry) {
      if (continuous_columns[indices[entry]]) {
        continuous_projection_rows[row] = true;
        break;
      }
    }
  }

  std::vector<bool> all_columns(n_columns, true);
  analysis.decompositions.push_back(
    build_decomposition(model, "integer_only_rows", integer_rows, integer_columns));
  analysis.decompositions.push_back(build_decomposition(
    model, "continuous_equality_rows", continuous_equality_rows, continuous_columns));
  analysis.decompositions.push_back(
    build_decomposition(model, "mixed_response_rows", mixed_rows, all_columns));
  analysis.decompositions.push_back(build_decomposition(
    model, "continuous_projection", continuous_projection_rows, continuous_columns));
}

}  // namespace
}  // namespace cuopt::mathematical_optimization::examples

namespace cuopt::mathematical_optimization::examples {

namespace {
void analyze_refinement(const structure_model_t& model, model_analysis_t& analysis);
void analyze_numerics(const structure_model_t& model, model_analysis_t& analysis);
void analyze_repair_candidates(const structure_model_t& model, model_analysis_t& analysis);
std::size_t tie_class_count(const refinement_summary_t& refinement);
}  // namespace

model_analysis_t analyze_structure(const structure_model_t& model,
                                   std::string scope,
                                   const std::vector<structure_index_t>& original_column_ids,
                                   analysis_level_t level,
                                   const std::vector<structure_index_t>& original_row_ids,
                                   std::size_t block_pair_provenance_limit)
{
  model_analysis_t analysis;
  analysis.scope = std::move(scope);
  analysis.level = level;
  if (original_column_ids.empty()) {
    analysis.original_column_ids.resize(model.get_n_variables());
    std::iota(analysis.original_column_ids.begin(), analysis.original_column_ids.end(), 0);
  } else {
    if (original_column_ids.size() != static_cast<std::size_t>(model.get_n_variables())) {
      throw std::invalid_argument("original column ID map does not match model dimensions");
    }
    analysis.original_column_ids = original_column_ids;
  }
  if (original_row_ids.empty()) {
    analysis.original_row_ids.resize(model.get_n_constraints());
    std::iota(analysis.original_row_ids.begin(), analysis.original_row_ids.end(), 0);
  } else {
    if (original_row_ids.size() != static_cast<std::size_t>(model.get_n_constraints())) {
      throw std::invalid_argument("original row ID map does not match model dimensions");
    }
    analysis.original_row_ids = original_row_ids;
  }

  const auto& offsets = model.get_constraint_matrix_offsets();
  const auto& indices = model.get_constraint_matrix_indices();
  const auto& values  = model.get_constraint_matrix_values();
  analysis.rows.reserve(model.get_n_constraints());
  for (structure_index_t row = 0; row < model.get_n_constraints(); ++row) {
    row_record_t record;
    record.row       = row;
    record.stable_id = stable_row_id(analysis.scope, row);
    record.name      = row_name(model, row);
    record.domain    = classify_row_domain(model, row);
    ++analysis.row_domain_counts[record.domain];
    for (auto entry = offsets[row]; entry < offsets[row + 1]; ++entry) {
      record.columns.push_back(indices[entry]);
      record.coefficients.push_back(values[entry]);
    }

    detect_binary_row_families(model, row, analysis, record);
    detect_binary_implications(model, row, analysis, record);
    detect_variable_bounds(model, row, analysis, record);
    detect_affine_definition(model, row, analysis, record);

    // New per-row detectors belong above this fallback. Post-scan graph or component analyses
    // belong with the typed aggregation calls below; see the Research breadcrumbs in README.md.
    if (record.families.empty()) { add_family(analysis, record, "general_linear"); }
    analysis.rows.push_back(std::move(record));
  }

  analyze_exact_one_blocks(model, analysis, block_pair_provenance_limit);
  analyze_decompositions(model, analysis);
  analyze_refinement(model, analysis);
  analyze_numerics(model, analysis);
  analyze_repair_candidates(model, analysis);
  return analysis;
}

namespace {

std::vector<std::int64_t> group_original_literals(const row_group_t& group,
                                                  const model_analysis_t& analysis)
{
  std::vector<std::int64_t> literals;
  for (const auto& member : group.members) {
    if (member.column < 0 ||
        member.column >= static_cast<structure_index_t>(analysis.original_column_ids.size())) {
      continue;
    }
    const auto original = analysis.original_column_ids[member.column];
    if (original < 0) { continue; }
    literals.push_back(2LL * original + static_cast<std::int64_t>(member.value));
  }
  std::sort(literals.begin(), literals.end());
  literals.erase(std::unique(literals.begin(), literals.end()), literals.end());
  return literals;
}

std::size_t intersection_size(const std::vector<std::int64_t>& lhs,
                              const std::vector<std::int64_t>& rhs)
{
  std::size_t count = 0;
  auto lhs_it       = lhs.begin();
  auto rhs_it       = rhs.begin();
  while (lhs_it != lhs.end() && rhs_it != rhs.end()) {
    if (*lhs_it < *rhs_it) {
      ++lhs_it;
    } else if (*rhs_it < *lhs_it) {
      ++rhs_it;
    } else {
      ++count;
      ++lhs_it;
      ++rhs_it;
    }
  }
  return count;
}

template <typename value_t>
std::map<std::string, std::int64_t> net_changes(const std::map<std::string, value_t>& original,
                                                const std::map<std::string, value_t>& reduced)
{
  std::set<std::string> keys;
  for (const auto& [key, unused_value] : original) {
    static_cast<void>(unused_value);
    keys.insert(key);
  }
  for (const auto& [key, unused_value] : reduced) {
    static_cast<void>(unused_value);
    keys.insert(key);
  }
  std::map<std::string, std::int64_t> result;
  for (const auto& key : keys) {
    const auto original_position = original.find(key);
    const auto reduced_position  = reduced.find(key);
    const auto original_value    = original_position == original.end()
                                     ? 0
                                     : static_cast<std::int64_t>(original_position->second);
    const auto reduced_value =
      reduced_position == reduced.end() ? 0 : static_cast<std::int64_t>(reduced_position->second);
    result[key] = reduced_value - original_value;
  }
  return result;
}

}  // namespace

presolve_structure_delta_t analyze_presolve_structure_delta(
  const model_analysis_t& original,
  const model_analysis_t& reduced,
  const std::vector<structure_index_t>& reduced_to_original,
  const std::vector<structure_index_t>& original_to_reduced,
  const std::vector<structure_index_t>& implied_integer_columns)
{
  // RESEARCH-BREADCRUMB(mps-structure/presolve-lineage) [internal-api]
  // Survivor maps support identity matching only. Consume ordered PaPILO postsolve records before
  // labeling an eliminated column fixed, substituted, or aggregated; never infer the operation.
  presolve_structure_delta_t delta;
  delta.row_domain_net_changes = net_changes(original.row_domain_counts, reduced.row_domain_counts);
  delta.row_family_net_changes = net_changes(original.row_family_counts, reduced.row_family_counts);
  for (const auto reduced_column : implied_integer_columns) {
    if (reduced_column < 0 ||
        static_cast<std::size_t>(reduced_column) >= reduced_to_original.size()) {
      throw std::out_of_range("implied-integer reduced column is outside the survivor map");
    }
    const auto original_column = reduced_to_original[static_cast<std::size_t>(reduced_column)];
    if (original_column < 0) {
      throw std::logic_error("implied-integer reduced column has no original-column identity");
    }
    delta.implied_integer_columns.push_back(mapped_column_t{reduced_column, original_column});
  }
  for (std::size_t original_column = 0; original_column < original_to_reduced.size();
       ++original_column) {
    if (original_to_reduced[original_column] < 0) {
      delta.eliminated_original_columns.push_back(static_cast<structure_index_t>(original_column));
    }
  }
  std::vector<bool> surviving_original_rows(original.original_row_ids.size(), false);
  for (const auto original_row : reduced.original_row_ids) {
    if (original_row >= 0 &&
        static_cast<std::size_t>(original_row) < surviving_original_rows.size()) {
      surviving_original_rows[static_cast<std::size_t>(original_row)] = true;
    }
  }
  for (std::size_t original_row = 0; original_row < surviving_original_rows.size();
       ++original_row) {
    if (!surviving_original_rows[original_row]) {
      delta.eliminated_original_rows.push_back(static_cast<structure_index_t>(original_row));
    }
  }

  std::vector<std::vector<std::int64_t>> original_groups;
  std::vector<std::vector<std::int64_t>> reduced_groups;
  for (const auto& group : original.exact_one_groups) {
    original_groups.push_back(group_original_literals(group, original));
  }
  for (const auto& group : reduced.exact_one_groups) {
    reduced_groups.push_back(group_original_literals(group, reduced));
  }

  for (const auto& original_group : original_groups) {
    const auto exact = std::find(reduced_groups.begin(), reduced_groups.end(), original_group);
    if (exact != reduced_groups.end()) {
      ++delta.exact_one.preserved;
      continue;
    }
    std::vector<std::int64_t> surviving;
    for (const auto encoded_literal : original_group) {
      const auto original_column = static_cast<std::size_t>(encoded_literal / 2);
      if (original_column < original_to_reduced.size() &&
          original_to_reduced[original_column] >= 0) {
        surviving.push_back(encoded_literal);
      }
    }
    if (!surviving.empty() && std::find(reduced_groups.begin(), reduced_groups.end(), surviving) !=
                                reduced_groups.end()) {
      ++delta.exact_one.contracted;
      continue;
    }
    std::size_t intersecting_groups = 0;
    for (const auto& reduced_group : reduced_groups) {
      if (intersection_size(original_group, reduced_group) != 0) { ++intersecting_groups; }
    }
    if (intersecting_groups > 1) {
      ++delta.exact_one.split;
    } else {
      ++delta.exact_one.destroyed;
    }
  }
  for (const auto& reduced_group : reduced_groups) {
    std::size_t contributing_groups = 0;
    for (const auto& original_group : original_groups) {
      if (intersection_size(original_group, reduced_group) != 0) { ++contributing_groups; }
    }
    if (contributing_groups > 1) { ++delta.exact_one.merged; }
  }

  delta.original_refinement_tie_classes = tie_class_count(original.refinement);
  delta.reduced_refinement_tie_classes  = tie_class_count(reduced.refinement);
  return delta;
}

namespace {

template <typename key_t, typename value_t>
void print_map_histogram(const std::map<key_t, value_t>& histogram)
{
  if (histogram.empty()) {
    std::cout << "none";
    return;
  }
  bool first = true;
  for (const auto& [key, value] : histogram) {
    if (!first) { std::cout << ", "; }
    std::cout << key << ':' << value;
    first = false;
  }
}

void print_vector_histogram(const std::vector<std::size_t>& histogram)
{
  bool first = true;
  for (std::size_t key = 0; key < histogram.size(); ++key) {
    if (histogram[key] == 0) { continue; }
    if (!first) { std::cout << ", "; }
    std::cout << key << ':' << histogram[key];
    first = false;
  }
  if (first) { std::cout << "none"; }
}

void print_quantiles(std::string_view label, const quantile_summary_t& summary)
{
  std::cout << "    " << label << ": min=" << std::scientific << std::setprecision(3)
            << summary.minimum << ", p25=" << summary.quartile_1 << ", p50=" << summary.median
            << ", p75=" << summary.quartile_3 << ", p90=" << summary.percentile_90
            << ", p99=" << summary.percentile_99 << ", max=" << summary.maximum << std::fixed
            << '\n';
}

std::size_t class_count(const std::map<std::size_t, std::size_t>& histogram, bool ties_only)
{
  std::size_t result = 0;
  for (const auto& [size, count] : histogram) {
    if (!ties_only || size > 1) { result += count; }
  }
  return result;
}

}  // namespace

void print_special_structure_summary(const structure_model_t& model,
                                     const model_analysis_t& analysis)
{
  std::cout << "  special structure (" << analysis_level_name(analysis.level) << "):\n";
  std::cout << "    row domains: ";
  print_map_histogram(analysis.row_domain_counts);
  std::cout << '\n';
  std::cout << "    row families (non-exclusive): ";
  print_map_histogram(analysis.row_family_counts);
  std::cout << '\n';

  const auto coverage = analysis.binary_variables == 0
                          ? 0.0
                          : 100.0 * analysis.exact_one_covered_binary_variables /
                              static_cast<double>(analysis.binary_variables);
  std::cout << "    exact-one: groups=" << analysis.exact_one_groups.size()
            << ", binary coverage=" << analysis.exact_one_covered_binary_variables << '/'
            << analysis.binary_variables << " (" << std::setprecision(1) << coverage
            << "%), overlap=" << (analysis.exact_one_groups_overlap ? "yes" : "no") << '\n';
  std::cout << "      group-size histogram: ";
  print_vector_histogram(analysis.exact_one_group_size_histogram);
  std::cout << "\n      overlap-component sizes: ";
  if (analysis.exact_one_overlap_component_sizes.empty()) {
    std::cout << "none";
  } else {
    std::map<std::size_t, std::size_t> component_size_histogram;
    for (const auto size : analysis.exact_one_overlap_component_sizes) {
      ++component_size_histogram[size];
    }
    print_map_histogram(component_size_histogram);
  }
  std::cout << '\n';
  const auto groups_to_show = std::min(display_limit, analysis.exact_one_groups.size());
  for (std::size_t group = 0; group < groups_to_show; ++group) {
    const auto& exact_one                      = analysis.exact_one_groups[group];
    constexpr std::size_t member_display_limit = 12;
    const auto members_to_show = std::min(member_display_limit, exact_one.members.size());
    std::cout << "      group " << exact_one.id << " from " << exact_one.stable_id;
    if (analysis.scope != "original" && exact_one.row >= 0 &&
        static_cast<std::size_t>(exact_one.row) < analysis.original_row_ids.size() &&
        analysis.original_row_ids[static_cast<std::size_t>(exact_one.row)] >= 0) {
      std::cout << " <- original:r"
                << analysis.original_row_ids[static_cast<std::size_t>(exact_one.row)];
    }
    std::cout << " (size " << exact_one.members.size() << "):";
    for (std::size_t member_index = 0; member_index < members_to_show; ++member_index) {
      const auto& member = exact_one.members[member_index];
      std::cout << ' ' << column_name(model, member.column) << '=' << member.value;
    }
    if (members_to_show < exact_one.members.size()) { std::cout << " ..."; }
    std::cout << '\n';
  }

  const auto activations = static_cast<std::size_t>(std::count_if(
    analysis.variable_bounds.begin(), analysis.variable_bounds.end(), [](const auto& relation) {
      return relation.activation;
    }));
  std::cout << "    relations: directed binary implications=" << analysis.implications.size()
            << ", variable bounds=" << analysis.variable_bounds.size()
            << " (activations=" << activations
            << "), affine definitions=" << analysis.affine_definitions.size() << '\n';

  std::cout << "    exact-one block pair provenance: "
            << (analysis.exact_one_block_edges_complete ? "complete" : "partial")
            << " (per-kind cap=" << analysis.exact_one_block_pair_provenance_limit
            << ", candidate=";
  print_map_histogram(analysis.exact_one_block_candidate_pair_provenance);
  std::cout << ", materialized=";
  print_map_histogram(analysis.exact_one_block_materialized_pair_provenance);
  std::cout << "); mediator hyperedges=" << analysis.exact_one_block_mediators.size() << '\n';
  std::cout << "    exact-one block projections:\n";
  for (const auto& projection : analysis.exact_one_block_projections) {
    std::cout << "      " << projection.name << ": edges=" << projection.edges
              << ", weight=" << projection.total_weight << ", components=" << projection.components
              << ", isolated=" << projection.isolated_vertices
              << ", largest=" << projection.largest_component;
    if (!projection.complete) { std::cout << " (partial materialization)"; }
    if (!projection.highest_weighted_degrees.empty()) {
      std::cout << ", top=";
      for (std::size_t index = 0; index < projection.highest_weighted_degrees.size(); ++index) {
        if (index != 0) { std::cout << ','; }
        const auto [block, degree] = projection.highest_weighted_degrees[index];
        std::cout << block << ':' << degree;
      }
    }
    std::cout << '\n';
  }

  std::cout << "    domain/component decompositions:\n";
  for (const auto& decomposition : analysis.decompositions) {
    std::map<std::string, std::size_t> interface_classes;
    for (const auto& row : decomposition.interface_rows) {
      ++interface_classes[row.classification];
    }
    std::cout << "      " << decomposition.name
              << ": components=" << decomposition.components.size() << ", component columns=";
    print_map_histogram(decomposition.component_size_histogram);
    std::cout << ", interfaces=";
    print_map_histogram(interface_classes);
    std::cout << ", widths=";
    print_map_histogram(decomposition.interface_width_histogram);
    std::cout << ", repeated refinement classes=" << decomposition.repeated_component_classes.size()
              << '\n';
  }

  std::cout << "    anonymous signatures/refinement: exact duplicate rows="
            << analysis.refinement.exact_duplicate_rows.size()
            << ", columns=" << analysis.refinement.exact_duplicate_columns.size()
            << ", normalized row templates=" << analysis.refinement.normalized_row_templates.size()
            << ", column templates=" << analysis.refinement.normalized_column_templates.size()
            << ", refinement rounds=" << analysis.refinement.rounds << ", unresolved row classes="
            << class_count(analysis.refinement.row_color_class_histogram, true)
            << ", column classes="
            << class_count(analysis.refinement.column_color_class_histogram, true) << '\n';

  std::cout << "    numerical diagnostics:\n";
  print_quantiles("|coefficient|", analysis.numerical.coefficient_magnitudes);
  print_quantiles("row dynamic range", analysis.numerical.row_dynamic_ranges);
  print_quantiles("column L2 norm", analysis.numerical.column_l2_norms);
  std::cout << "      signs: positive=" << analysis.numerical.positive_coefficients
            << ", negative=" << analysis.numerical.negative_coefficients
            << ", near-zero=" << analysis.numerical.near_zero_coefficients
            << "; near-cancellation rows=" << analysis.numerical.near_cancellation_rows.size()
            << "; decimal-unscalable rows=" << analysis.numerical.rows_not_decimal_scalable << '\n';

  std::cout << "    continuous repair candidates: monotone-up="
            << analysis.repair.monotone_increase_columns.size()
            << ", monotone-down=" << analysis.repair.monotone_decrease_columns.size()
            << ", free equality pivots=" << analysis.repair.free_equality_pivots.size()
            << ", two-variable affine rows=" << analysis.repair.two_variable_affine_rows.size()
            << ", difference components=" << analysis.repair.difference_components.size()
            << ", flow-like components=" << analysis.repair.flow_equality_components.size() << '\n';
}

void print_presolve_structure_delta(const presolve_structure_delta_t& delta)
{
  std::cout
    << "  structural-role deltas (net counts; PaPILO row-operation provenance unavailable):\n";
  std::cout << "    row domains: ";
  print_map_histogram(delta.row_domain_net_changes);
  std::cout << "\n    row families: ";
  print_map_histogram(delta.row_family_net_changes);
  std::cout << "\n    exact-one membership mapping: preserved=" << delta.exact_one.preserved
            << ", contracted=" << delta.exact_one.contracted << ", split=" << delta.exact_one.split
            << ", destroyed=" << delta.exact_one.destroyed << ", merged=" << delta.exact_one.merged
            << '\n';
  std::cout << "    eliminated original columns=" << delta.eliminated_original_columns.size()
            << " (fixed/substituted/aggregated reason is not exposed), eliminated original rows="
            << delta.eliminated_original_rows.size()
            << " (row-reduction operation type is not exposed), implied-integer="
            << delta.implied_integer_columns.size() << '\n';
  std::cout << "    refinement tie classes: " << delta.original_refinement_tie_classes << " -> "
            << delta.reduced_refinement_tie_classes << '\n';
}

}  // namespace cuopt::mathematical_optimization::examples

namespace cuopt::mathematical_optimization::examples {
namespace {

std::vector<std::vector<std::pair<structure_index_t, double>>> build_columns(
  const structure_model_t& model)
{
  std::vector<std::vector<std::pair<structure_index_t, double>>> columns(model.get_n_variables());
  const auto& offsets = model.get_constraint_matrix_offsets();
  const auto& indices = model.get_constraint_matrix_indices();
  const auto& values  = model.get_constraint_matrix_values();
  for (structure_index_t row = 0; row < model.get_n_constraints(); ++row) {
    for (auto entry = offsets[row]; entry < offsets[row + 1]; ++entry) {
      columns[indices[entry]].emplace_back(row, values[entry]);
    }
  }
  return columns;
}

std::string exact_row_key(const structure_model_t& model, structure_index_t row)
{
  const auto& offsets = model.get_constraint_matrix_offsets();
  const auto& indices = model.get_constraint_matrix_indices();
  const auto& values  = model.get_constraint_matrix_values();
  std::vector<std::string> parts{
    "lb:" + canonical_double(model.get_constraint_lower_bounds()[row]),
    "ub:" + canonical_double(model.get_constraint_upper_bounds()[row])};
  for (auto entry = offsets[row]; entry < offsets[row + 1]; ++entry) {
    parts.push_back(std::to_string(indices[entry]) + "=" + canonical_double(values[entry]));
  }
  return join_key(parts);
}

std::string normalized_row_key(const structure_model_t& model, structure_index_t row)
{
  const auto& offsets = model.get_constraint_matrix_offsets();
  const auto& indices = model.get_constraint_matrix_indices();
  const auto& values  = model.get_constraint_matrix_values();
  double scale        = 0.0;
  for (auto entry = offsets[row]; entry < offsets[row + 1]; ++entry) {
    scale = std::max(scale, std::abs(values[entry]));
  }
  if (scale == 0.0) { scale = 1.0; }
  std::vector<std::string> parts;
  for (auto entry = offsets[row]; entry < offsets[row + 1]; ++entry) {
    const auto column = indices[entry];
    parts.push_back(std::string(1, model.get_variable_types()[column]) + "=" +
                    canonical_double(values[entry] / scale));
  }
  std::sort(parts.begin(), parts.end());
  parts.push_back("lb:" + canonical_double(model.get_constraint_lower_bounds()[row] / scale));
  parts.push_back("ub:" + canonical_double(model.get_constraint_upper_bounds()[row] / scale));
  return join_key(parts);
}

std::string exact_column_key(
  const structure_model_t& model,
  structure_index_t column,
  const std::vector<std::vector<std::pair<structure_index_t, double>>>& columns)
{
  std::vector<std::string> parts{
    std::string("type:") + model.get_variable_types()[column],
    "lb:" + canonical_double(model.get_variable_lower_bounds()[column]),
    "ub:" + canonical_double(model.get_variable_upper_bounds()[column]),
    "obj:" + canonical_double(model.get_objective_coefficients()[column])};
  for (const auto& [row, coefficient] : columns[column]) {
    parts.push_back(std::to_string(row) + "=" + canonical_double(coefficient));
  }
  return join_key(parts);
}

std::string normalized_column_key(
  const structure_model_t& model,
  structure_index_t column,
  const std::vector<std::vector<std::pair<structure_index_t, double>>>& columns,
  const std::vector<row_record_t>& rows)
{
  double scale = std::abs(model.get_objective_coefficients()[column]);
  for (const auto& [unused_row, coefficient] : columns[column]) {
    static_cast<void>(unused_row);
    scale = std::max(scale, std::abs(coefficient));
  }
  if (scale == 0.0) { scale = 1.0; }
  std::vector<std::string> parts{
    std::string("type:") + model.get_variable_types()[column],
    "lb:" + canonical_double(model.get_variable_lower_bounds()[column]),
    "ub:" + canonical_double(model.get_variable_upper_bounds()[column]),
    "obj:" + canonical_double(model.get_objective_coefficients()[column] / scale)};
  for (const auto& [row, coefficient] : columns[column]) {
    parts.push_back(rows[row].domain + "=" + canonical_double(coefficient / scale));
  }
  std::sort(parts.begin() + 4, parts.end());
  return join_key(parts);
}

std::vector<duplicate_class_t> make_classes(const std::vector<std::string>& keys,
                                            bool retain_singletons)
{
  std::map<std::string, std::vector<structure_index_t>> members_by_key;
  for (std::size_t index = 0; index < keys.size(); ++index) {
    members_by_key[keys[index]].push_back(static_cast<structure_index_t>(index));
  }
  std::vector<duplicate_class_t> classes;
  for (const auto& [key, members] : members_by_key) {
    if (retain_singletons || members.size() > 1) {
      classes.push_back(duplicate_class_t{fingerprint(key), members});
    }
  }
  std::sort(classes.begin(), classes.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.members.size() != rhs.members.size() ? lhs.members.size() > rhs.members.size()
                                                    : lhs.fingerprint < rhs.fingerprint;
  });
  return classes;
}

std::vector<structure_index_t> assign_colors(const std::vector<std::string>& keys)
{
  std::map<std::string, structure_index_t> color_by_key;
  for (const auto& key : keys) {
    color_by_key.emplace(key, 0);
  }
  structure_index_t next_color = 0;
  for (auto& [unused_key, color] : color_by_key) {
    static_cast<void>(unused_key);
    color = next_color++;
  }
  std::vector<structure_index_t> colors;
  colors.reserve(keys.size());
  for (const auto& key : keys) {
    colors.push_back(color_by_key.at(key));
  }
  return colors;
}

std::map<std::size_t, std::size_t> color_class_histogram(
  const std::vector<structure_index_t>& colors)
{
  std::map<structure_index_t, std::size_t> class_sizes;
  for (const auto color : colors) {
    ++class_sizes[color];
  }
  std::map<std::size_t, std::size_t> histogram;
  for (const auto& [unused_color, size] : class_sizes) {
    static_cast<void>(unused_color);
    ++histogram[size];
  }
  return histogram;
}

void analyze_refinement(const structure_model_t& model, model_analysis_t& analysis)
{
  // RESEARCH-BREADCRUMB(mps-structure/refinement-stabilization) [driver-local]
  // Iterate to stability or a reported cap. These color classes are candidates, not automorphism
  // or isomorphism certificates; actual orbits have a separate breadcrumb in the driver.
  const auto columns = build_columns(model);
  std::vector<std::string> exact_row_keys;
  std::vector<std::string> normalized_row_keys;
  std::vector<std::string> exact_column_keys;
  std::vector<std::string> normalized_column_keys;
  exact_row_keys.reserve(model.get_n_constraints());
  normalized_row_keys.reserve(model.get_n_constraints());
  exact_column_keys.reserve(model.get_n_variables());
  normalized_column_keys.reserve(model.get_n_variables());
  for (structure_index_t row = 0; row < model.get_n_constraints(); ++row) {
    exact_row_keys.push_back(exact_row_key(model, row));
    normalized_row_keys.push_back(normalized_row_key(model, row));
  }
  for (structure_index_t column = 0; column < model.get_n_variables(); ++column) {
    exact_column_keys.push_back(exact_column_key(model, column, columns));
    normalized_column_keys.push_back(normalized_column_key(model, column, columns, analysis.rows));
  }

  analysis.refinement.exact_duplicate_rows        = make_classes(exact_row_keys, false);
  analysis.refinement.exact_duplicate_columns     = make_classes(exact_column_keys, false);
  analysis.refinement.normalized_row_templates    = make_classes(normalized_row_keys, false);
  analysis.refinement.normalized_column_templates = make_classes(normalized_column_keys, false);
  analysis.refinement.row_fingerprints.reserve(normalized_row_keys.size());
  analysis.refinement.column_fingerprints.reserve(normalized_column_keys.size());
  for (const auto& key : normalized_row_keys) {
    analysis.refinement.row_fingerprints.push_back(fingerprint(key));
  }
  for (const auto& key : normalized_column_keys) {
    analysis.refinement.column_fingerprints.push_back(fingerprint(key));
  }

  auto row_colors    = assign_colors(normalized_row_keys);
  auto column_colors = assign_colors(normalized_column_keys);
  if (analysis.level == analysis_level_t::symmetry) {
    constexpr std::size_t rounds = 3;
    const auto& offsets          = model.get_constraint_matrix_offsets();
    const auto& indices          = model.get_constraint_matrix_indices();
    const auto& values           = model.get_constraint_matrix_values();
    for (std::size_t round = 0; round < rounds; ++round) {
      std::vector<std::string> row_keys(model.get_n_constraints());
      std::vector<std::string> column_keys(model.get_n_variables());
      for (structure_index_t row = 0; row < model.get_n_constraints(); ++row) {
        std::vector<std::string> neighbors;
        neighbors.push_back("self:" + std::to_string(row_colors[row]));
        for (auto entry = offsets[row]; entry < offsets[row + 1]; ++entry) {
          neighbors.push_back(canonical_double(values[entry]) + "@" +
                              std::to_string(column_colors[indices[entry]]));
        }
        std::sort(neighbors.begin() + 1, neighbors.end());
        row_keys[row] = "R" + join_key(neighbors);
      }
      for (structure_index_t column = 0; column < model.get_n_variables(); ++column) {
        std::vector<std::string> neighbors;
        neighbors.push_back("self:" + std::to_string(column_colors[column]));
        for (const auto& [row, coefficient] : columns[column]) {
          neighbors.push_back(canonical_double(coefficient) + "@" +
                              std::to_string(row_colors[row]));
        }
        std::sort(neighbors.begin() + 1, neighbors.end());
        column_keys[column] = "C" + join_key(neighbors);
      }
      row_colors    = assign_colors(row_keys);
      column_colors = assign_colors(column_keys);
    }
    analysis.refinement.rounds = rounds;
  }
  analysis.refinement.row_colors    = std::move(row_colors);
  analysis.refinement.column_colors = std::move(column_colors);
  analysis.refinement.row_color_class_histogram =
    color_class_histogram(analysis.refinement.row_colors);
  analysis.refinement.column_color_class_histogram =
    color_class_histogram(analysis.refinement.column_colors);
}

double quantile(const std::vector<double>& sorted, double probability)
{
  if (sorted.empty()) { return 0.0; }
  const auto position = probability * static_cast<double>(sorted.size() - 1);
  const auto lower    = static_cast<std::size_t>(std::floor(position));
  const auto upper    = static_cast<std::size_t>(std::ceil(position));
  const auto fraction = position - static_cast<double>(lower);
  return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

quantile_summary_t summarize_quantiles(std::vector<double> values)
{
  if (values.empty()) { return {}; }
  std::sort(values.begin(), values.end());
  return quantile_summary_t{values.front(),
                            quantile(values, 0.25),
                            quantile(values, 0.50),
                            quantile(values, 0.75),
                            quantile(values, 0.90),
                            quantile(values, 0.99),
                            values.back()};
}

int decimal_scale_exponent(const structure_model_t& model, structure_index_t row)
{
  const auto& offsets = model.get_constraint_matrix_offsets();
  const auto& values  = model.get_constraint_matrix_values();
  std::vector<double> row_values;
  for (auto entry = offsets[row]; entry < offsets[row + 1]; ++entry) {
    row_values.push_back(values[entry]);
  }
  const auto lower = model.get_constraint_lower_bounds()[row];
  const auto upper = model.get_constraint_upper_bounds()[row];
  if (std::isfinite(lower)) { row_values.push_back(lower); }
  if (std::isfinite(upper)) { row_values.push_back(upper); }

  double scale = 1.0;
  for (int exponent = 0; exponent <= 6; ++exponent) {
    const bool integral = std::all_of(row_values.begin(), row_values.end(), [scale](double value) {
      const auto scaled = value * scale;
      if (!std::isfinite(scaled)) { return false; }
      const auto nearest = std::round(scaled);
      const auto upward =
        std::abs(std::nextafter(nearest, std::numeric_limits<double>::infinity()) - nearest);
      const auto downward =
        std::abs(nearest - std::nextafter(nearest, -std::numeric_limits<double>::infinity()));
      const auto representation_tolerance = 4.0 * std::max(upward, downward);
      return std::abs(scaled - nearest) <= std::max(absolute_tolerance, representation_tolerance);
    });
    if (integral) { return exponent; }
    scale *= 10.0;
  }
  return -1;
}

void analyze_numerics(const structure_model_t& model, model_analysis_t& analysis)
{
  const auto& offsets = model.get_constraint_matrix_offsets();
  const auto& indices = model.get_constraint_matrix_indices();
  const auto& values  = model.get_constraint_matrix_values();
  std::vector<double> magnitudes;
  std::vector<double> row_ranges;
  std::vector<double> column_norm_squares(model.get_n_variables(), 0.0);
  magnitudes.reserve(values.size());

  for (structure_index_t row = 0; row < model.get_n_constraints(); ++row) {
    double minimum_magnitude = std::numeric_limits<double>::infinity();
    double maximum_magnitude = 0.0;
    double signed_sum        = 0.0;
    double absolute_sum      = 0.0;
    for (auto entry = offsets[row]; entry < offsets[row + 1]; ++entry) {
      const auto value     = values[entry];
      const auto magnitude = std::abs(value);
      if (value > 0.0) {
        ++analysis.numerical.positive_coefficients;
      } else if (value < 0.0) {
        ++analysis.numerical.negative_coefficients;
      }
      magnitudes.push_back(magnitude);
      minimum_magnitude = std::min(minimum_magnitude, magnitude);
      maximum_magnitude = std::max(maximum_magnitude, magnitude);
      signed_sum += value;
      absolute_sum += magnitude;
      column_norm_squares[indices[entry]] += value * value;
    }
    if (maximum_magnitude > 0.0) {
      row_ranges.push_back(maximum_magnitude / minimum_magnitude);
      for (auto entry = offsets[row]; entry < offsets[row + 1]; ++entry) {
        if (std::abs(values[entry]) <= std::max(1e-12, maximum_magnitude * 1e-9)) {
          ++analysis.numerical.near_zero_coefficients;
        }
      }
    }
    if (absolute_sum > 0.0 && std::abs(signed_sum) <= 1e-10 * absolute_sum &&
        offsets[row + 1] - offsets[row] > 1) {
      analysis.numerical.near_cancellation_rows.push_back(row);
    }
    const auto exponent = decimal_scale_exponent(model, row);
    if (exponent < 0) {
      ++analysis.numerical.rows_not_decimal_scalable;
    } else {
      ++analysis.numerical.decimal_scale_exponent_histogram[exponent];
    }
  }

  std::vector<double> column_norms;
  column_norms.reserve(column_norm_squares.size());
  for (const auto norm_square : column_norm_squares) {
    column_norms.push_back(std::sqrt(norm_square));
  }
  analysis.numerical.coefficient_magnitudes = summarize_quantiles(std::move(magnitudes));
  analysis.numerical.row_dynamic_ranges     = summarize_quantiles(std::move(row_ranges));
  analysis.numerical.column_l2_norms        = summarize_quantiles(std::move(column_norms));
}

void analyze_repair_candidates(const structure_model_t& model, model_analysis_t& analysis)
{
  const auto columns = build_columns(model);
  const auto& types  = model.get_variable_types();
  const auto& lower  = model.get_variable_lower_bounds();
  const auto& upper  = model.get_variable_upper_bounds();
  const auto& row_lb = model.get_constraint_lower_bounds();
  const auto& row_ub = model.get_constraint_upper_bounds();
  for (structure_index_t column = 0; column < model.get_n_variables(); ++column) {
    if (is_integral_type(types[column])) { continue; }
    bool increase_safe = !std::isfinite(upper[column]);
    bool decrease_safe = !std::isfinite(lower[column]);
    for (const auto& [row, coefficient] : columns[column]) {
      if (std::isfinite(row_lb[row]) && coefficient < -absolute_tolerance) {
        increase_safe = false;
      }
      if (std::isfinite(row_ub[row]) && coefficient > absolute_tolerance) { increase_safe = false; }
      if (std::isfinite(row_lb[row]) && coefficient > absolute_tolerance) { decrease_safe = false; }
      if (std::isfinite(row_ub[row]) && coefficient < -absolute_tolerance) {
        decrease_safe = false;
      }
    }
    if (increase_safe) { analysis.repair.monotone_increase_columns.push_back(column); }
    if (decrease_safe) { analysis.repair.monotone_decrease_columns.push_back(column); }
  }

  for (const auto& definition : analysis.affine_definitions) {
    analysis.repair.free_equality_pivots.insert(analysis.repair.free_equality_pivots.end(),
                                                definition.pivot_columns.begin(),
                                                definition.pivot_columns.end());
    if (definition.two_variable) {
      analysis.repair.two_variable_affine_rows.push_back(definition.row);
    }
  }
  std::sort(analysis.repair.free_equality_pivots.begin(),
            analysis.repair.free_equality_pivots.end());
  analysis.repair.free_equality_pivots.erase(
    std::unique(analysis.repair.free_equality_pivots.begin(),
                analysis.repair.free_equality_pivots.end()),
    analysis.repair.free_equality_pivots.end());

  const auto& offsets = model.get_constraint_matrix_offsets();
  const auto& indices = model.get_constraint_matrix_indices();
  const auto& values  = model.get_constraint_matrix_values();
  std::vector<bool> continuous_columns(model.get_n_variables(), false);
  for (structure_index_t column = 0; column < model.get_n_variables(); ++column) {
    continuous_columns[column] = !is_integral_type(types[column]);
  }
  std::vector<bool> difference_rows(model.get_n_constraints(), false);
  std::vector<bool> flow_rows(model.get_n_constraints(), false);
  std::vector<std::size_t> flow_column_incidence(model.get_n_variables(), 0);
  for (structure_index_t row = 0; row < model.get_n_constraints(); ++row) {
    if (analysis.rows[row].domain != "continuous_only" || !row_is_equality(model, row)) {
      continue;
    }
    const auto begin = offsets[row];
    const auto end   = offsets[row + 1];
    if (end - begin == 2 && values[begin] * values[begin + 1] < 0.0 &&
        nearly_equal(std::abs(values[begin]), std::abs(values[begin + 1]))) {
      difference_rows[row] = true;
    }
    bool positive = false;
    bool negative = false;
    for (auto entry = begin; entry < end; ++entry) {
      positive = positive || values[entry] > 0.0;
      negative = negative || values[entry] < 0.0;
    }
    if (positive && negative) {
      flow_rows[row] = true;
      for (auto entry = begin; entry < end; ++entry) {
        ++flow_column_incidence[indices[entry]];
      }
    }
  }
  auto difference =
    build_decomposition(model, "difference_candidates", difference_rows, continuous_columns);
  analysis.repair.difference_components = std::move(difference.components);
  auto flow = build_decomposition(model, "flow_candidates", flow_rows, continuous_columns);
  for (auto& component : flow.components) {
    const bool bounded_incidence =
      std::all_of(component.columns.begin(), component.columns.end(), [&](auto column) {
        return flow_column_incidence[column] <= 2;
      });
    if (bounded_incidence) {
      analysis.repair.flow_equality_components.push_back(std::move(component));
    }
  }
}

std::size_t tie_class_count(const refinement_summary_t& refinement)
{
  std::size_t count = 0;
  for (const auto& [class_size, classes] : refinement.row_color_class_histogram) {
    if (class_size > 1) { count += classes; }
  }
  for (const auto& [class_size, classes] : refinement.column_color_class_histogram) {
    if (class_size > 1) { count += classes; }
  }
  return count;
}

}  // namespace
}  // namespace cuopt::mathematical_optimization::examples
