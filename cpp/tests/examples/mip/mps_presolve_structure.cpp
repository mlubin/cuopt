/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>
#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/utilities/internals.hpp>
#include <mip_heuristics/presolve/third_party_presolve.hpp>

#include "mps_conflict_graph.hpp"
#include "mps_lp_overlay.hpp"
#include "mps_structure_analysis.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using index_t           = int;
using value_t           = double;
using model_t           = cuopt::mathematical_optimization::io::mps_data_model_t<index_t, value_t>;
using presolve_status_t = cuopt::mathematical_optimization::mip::third_party_presolve_status_t;
namespace structure     = cuopt::mathematical_optimization::examples;

struct driver_options_t {
  std::string model_path;
  bool fixed_format{};
  structure::analysis_level_t analysis_level{structure::analysis_level_t::basic};
  bool lp_overlay{};
  bool objective_erased_lp{};
  std::optional<std::string> json_path;
};

// RESEARCH-BREADCRUMB(mps-structure/report-detail-levels) [driver-local]
// Add orthogonal stage/detail/resource controls here and serialize their effective values in the
// run manifest. Detection must remain independent of human, JSON, and collection renderers.

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

struct degree_summary_t {
  index_t minimum{};
  index_t maximum{};
  std::size_t empty{};
  std::size_t singleton{};
  double mean{};
};

struct component_t {
  std::size_t rows{};
  std::size_t columns{};
  std::size_t nonzeros{};
};

struct component_summary_t {
  std::size_t count{};
  component_t largest{};
};

struct structure_t {
  std::vector<index_t> row_degrees;
  std::vector<index_t> column_degrees;
  component_summary_t components;
};

degree_summary_t summarize_degrees(const std::vector<index_t>& degrees)
{
  if (degrees.empty()) { return {}; }
  auto [minimum, maximum] = std::minmax_element(degrees.begin(), degrees.end());
  return degree_summary_t{
    *minimum,
    *maximum,
    static_cast<std::size_t>(std::count(degrees.begin(), degrees.end(), 0)),
    static_cast<std::size_t>(std::count(degrees.begin(), degrees.end(), 1)),
    static_cast<double>(std::accumulate(degrees.begin(), degrees.end(), std::int64_t{0})) /
      static_cast<double>(degrees.size())};
}

structure_t inspect_matrix(const model_t& model)
{
  const auto n_rows   = model.get_n_constraints();
  const auto n_cols   = model.get_n_variables();
  const auto& offsets = model.get_constraint_matrix_offsets();
  const auto& indices = model.get_constraint_matrix_indices();

  structure_t structure;
  structure.row_degrees.resize(n_rows, 0);
  structure.column_degrees.resize(n_cols, 0);
  disjoint_set_t sets(static_cast<std::size_t>(n_rows) + static_cast<std::size_t>(n_cols));

  for (index_t row = 0; row < n_rows; ++row) {
    const auto begin           = offsets[static_cast<std::size_t>(row)];
    const auto end             = offsets[static_cast<std::size_t>(row) + 1];
    structure.row_degrees[row] = end - begin;
    const auto row_node        = static_cast<std::size_t>(row);
    for (index_t entry = begin; entry < end; ++entry) {
      const auto column = indices[static_cast<std::size_t>(entry)];
      ++structure.column_degrees[column];
      sets.unite(row_node, static_cast<std::size_t>(n_rows) + static_cast<std::size_t>(column));
    }
  }

  std::unordered_map<std::size_t, component_t> components;
  for (index_t row = 0; row < n_rows; ++row) {
    if (structure.row_degrees[row] == 0) { continue; }
    ++components[sets.find(static_cast<std::size_t>(row))].rows;
  }
  for (index_t column = 0; column < n_cols; ++column) {
    if (structure.column_degrees[column] == 0) { continue; }
    ++components[sets.find(static_cast<std::size_t>(n_rows) + static_cast<std::size_t>(column))]
        .columns;
  }
  for (index_t row = 0; row < n_rows; ++row) {
    if (structure.row_degrees[row] == 0) { continue; }
    components[sets.find(static_cast<std::size_t>(row))].nonzeros +=
      static_cast<std::size_t>(structure.row_degrees[row]);
  }

  structure.components.count = components.size();
  for (const auto& [unused_root, component] : components) {
    static_cast<void>(unused_root);
    const auto component_nodes = component.rows + component.columns;
    const auto largest_nodes =
      structure.components.largest.rows + structure.components.largest.columns;
    if (component_nodes > largest_nodes ||
        (component_nodes == largest_nodes &&
         component.nonzeros > structure.components.largest.nonzeros)) {
      structure.components.largest = component;
    }
  }
  return structure;
}

std::string_view status_name(presolve_status_t status)
{
  switch (status) {
    case presolve_status_t::INFEASIBLE: return "INFEASIBLE";
    case presolve_status_t::UNBOUNDED: return "UNBOUNDED";
    case presolve_status_t::UNBNDORINFEAS: return "UNBOUNDED_OR_INFEASIBLE";
    case presolve_status_t::OPTIMAL: return "OPTIMAL_DURING_PRESOLVE";
    case presolve_status_t::REDUCED: return "REDUCED";
    case presolve_status_t::UNCHANGED: return "UNCHANGED";
  }
  return "UNKNOWN";
}

bool is_terminal_without_model(presolve_status_t status)
{
  return status == presolve_status_t::INFEASIBLE || status == presolve_status_t::UNBOUNDED ||
         status == presolve_status_t::UNBNDORINFEAS;
}

void print_degree_summary(std::string_view label, const std::vector<index_t>& degrees)
{
  const auto summary = summarize_degrees(degrees);
  std::cout << "  " << label << " degree: min=" << summary.minimum << ", mean=" << std::fixed
            << std::setprecision(2) << summary.mean << ", max=" << summary.maximum
            << ", empty=" << summary.empty << ", singleton=" << summary.singleton << '\n';
}

void print_densest(std::string_view label,
                   const std::vector<index_t>& degrees,
                   const std::vector<std::string>& names)
{
  constexpr std::size_t limit = 5;
  std::vector<std::size_t> order(degrees.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  const auto count = std::min(limit, order.size());
  std::partial_sort(order.begin(),
                    order.begin() + static_cast<std::ptrdiff_t>(count),
                    order.end(),
                    [&degrees](std::size_t lhs, std::size_t rhs) {
                      if (degrees[lhs] != degrees[rhs]) { return degrees[lhs] > degrees[rhs]; }
                      return lhs < rhs;
                    });

  std::cout << "  densest " << label << ':';
  if (count == 0) {
    std::cout << " none\n";
    return;
  }
  std::cout << '\n';
  for (std::size_t rank = 0; rank < count; ++rank) {
    const auto index = order[rank];
    std::cout << "    [" << index << ']';
    if (index < names.size() && !names[index].empty()) { std::cout << ' ' << names[index]; }
    std::cout << ": " << degrees[index] << " nonzeros\n";
  }
}

void print_model_summary(std::string_view heading, const model_t& model)
{
  const auto rows        = model.get_n_constraints();
  const auto columns     = model.get_n_variables();
  const auto nonzeros    = model.get_nnz();
  const auto matrix_size = static_cast<double>(rows) * static_cast<double>(columns);
  const auto density     = matrix_size == 0.0 ? 0.0 : static_cast<double>(nonzeros) / matrix_size;

  std::size_t continuous      = 0;
  std::size_t integer         = 0;
  std::size_t semi_continuous = 0;
  std::size_t binary          = 0;
  std::size_t unfixed_binary  = 0;
  const auto& variable_types  = model.get_variable_types();
  const auto& variable_lower  = model.get_variable_lower_bounds();
  const auto& variable_upper  = model.get_variable_upper_bounds();
  for (std::size_t column = 0; column < variable_types.size(); ++column) {
    const auto type = variable_types[column];
    if (type == 'I' || type == 'B') {
      ++integer;
      if (column < variable_lower.size() && column < variable_upper.size() &&
          (variable_lower[column] == 0.0 || variable_lower[column] == 1.0) &&
          (variable_upper[column] == 0.0 || variable_upper[column] == 1.0) &&
          variable_lower[column] <= variable_upper[column]) {
        ++binary;
        if (variable_lower[column] == 0.0 && variable_upper[column] == 1.0) { ++unfixed_binary; }
      }
    } else if (type == 'S') {
      ++semi_continuous;
    } else {
      ++continuous;
    }
  }

  std::size_t fixed      = 0;
  std::size_t free       = 0;
  std::size_t lower_only = 0;
  std::size_t upper_only = 0;
  std::size_t boxed      = 0;
  for (std::size_t column = 0; column < variable_lower.size(); ++column) {
    const auto lower_finite = std::isfinite(variable_lower[column]);
    const auto upper_finite = std::isfinite(variable_upper[column]);
    if (lower_finite && upper_finite && variable_lower[column] == variable_upper[column]) {
      ++fixed;
    } else if (!lower_finite && !upper_finite) {
      ++free;
    } else if (lower_finite && !upper_finite) {
      ++lower_only;
    } else if (!lower_finite && upper_finite) {
      ++upper_only;
    } else {
      ++boxed;
    }
  }

  std::size_t equalities       = 0;
  std::size_t less_equal       = 0;
  std::size_t greater_equal    = 0;
  std::size_t ranged           = 0;
  std::size_t free_rows        = 0;
  const auto& constraint_lower = model.get_constraint_lower_bounds();
  const auto& constraint_upper = model.get_constraint_upper_bounds();
  for (std::size_t row = 0; row < constraint_lower.size(); ++row) {
    const auto lower_finite = std::isfinite(constraint_lower[row]);
    const auto upper_finite = std::isfinite(constraint_upper[row]);
    if (lower_finite && upper_finite && constraint_lower[row] == constraint_upper[row]) {
      ++equalities;
    } else if (!lower_finite && !upper_finite) {
      ++free_rows;
    } else if (!lower_finite) {
      ++less_equal;
    } else if (!upper_finite) {
      ++greater_equal;
    } else {
      ++ranged;
    }
  }

  const auto objective_nonzeros =
    std::count_if(model.get_objective_coefficients().begin(),
                  model.get_objective_coefficients().end(),
                  [](value_t coefficient) { return coefficient != 0.0; });
  const auto structure = inspect_matrix(model);

  std::cout << "\n" << heading << '\n';
  if (!model.get_problem_name().empty()) {
    std::cout << "  name: " << model.get_problem_name() << '\n';
  }
  std::cout << "  dimensions: " << rows << " rows, " << columns << " columns, " << nonzeros
            << " nonzeros\n";
  std::cout << "  density: " << std::scientific << std::setprecision(4) << density << std::fixed
            << '\n';
  std::cout << "  objective: " << (model.get_sense() ? "maximize" : "minimize") << ", "
            << objective_nonzeros << " nonzero coefficients\n";
  std::cout << "  variables: continuous=" << continuous << ", integer=" << integer
            << " (binary-domain=" << binary << ", unfixed-binary=" << unfixed_binary
            << "), semi-continuous=" << semi_continuous << '\n';
  std::cout << "  variable bounds: fixed=" << fixed << ", free=" << free
            << ", lower-only=" << lower_only << ", upper-only=" << upper_only << ", boxed=" << boxed
            << '\n';
  std::cout << "  constraints: equality=" << equalities << ", <= " << less_equal
            << ", >= " << greater_equal << ", ranged=" << ranged << ", free=" << free_rows << '\n';
  print_degree_summary("row", structure.row_degrees);
  print_degree_summary("column", structure.column_degrees);
  std::cout << "  coupled matrix components: " << structure.components.count;
  if (structure.components.count != 0) {
    const auto& largest = structure.components.largest;
    std::cout << " (largest: " << largest.rows << " rows, " << largest.columns << " columns, "
              << largest.nonzeros << " nonzeros)";
  }
  std::cout << '\n';
  print_densest("rows", structure.row_degrees, model.get_row_names());
  print_densest("columns", structure.column_degrees, model.get_variable_names());
  structure::print_conflict_graph_summary(model);
}

void print_reduction(std::string_view label, std::size_t before, std::size_t after)
{
  const auto removed = before - after;
  const auto percent = before == 0 ? 0.0 : 100.0 * static_cast<double>(removed) / before;
  std::cout << "  " << label << ": " << before << " -> " << after << " (removed " << removed << ", "
            << std::fixed << std::setprecision(1) << percent << "%)\n";
}

bool has_discrete_variables(const model_t& model)
{
  return std::any_of(model.get_variable_types().begin(),
                     model.get_variable_types().end(),
                     [](char type) { return type == 'I' || type == 'B' || type == 'S'; });
}

bool has_semi_continuous_variables(const model_t& model)
{
  return std::find(model.get_variable_types().begin(), model.get_variable_types().end(), 'S') !=
         model.get_variable_types().end();
}

void print_usage(const char* executable)
{
  std::cerr << "Usage: " << executable
            << " <model.mps> [--fixed] [--level basic|lp|symmetry]..."
               " [--objective-erased-lp] [--json <path>]\n";
}

driver_options_t parse_options(int argc, char** argv)
{
  if (argc < 2) { throw std::invalid_argument("missing MPS path"); }
  driver_options_t options;
  options.model_path = argv[1];
  for (int argument = 2; argument < argc; ++argument) {
    const std::string_view value = argv[argument];
    if (value == "--fixed") {
      options.fixed_format = true;
    } else if (value == "--json") {
      if (++argument == argc) { throw std::invalid_argument("--json requires an output path"); }
      options.json_path = argv[argument];
    } else if (value == "--level") {
      if (++argument == argc) { throw std::invalid_argument("--level requires a value"); }
      const std::string_view level = argv[argument];
      if (level == "basic") {
        continue;
      } else if (level == "lp") {
        options.lp_overlay = true;
      } else if (level == "symmetry") {
        options.analysis_level = structure::analysis_level_t::symmetry;

        // RESEARCH-BREADCRUMB(mps-structure/automorphism-orbits) [internal-api]
        // The current mode is deterministic color refinement. Add actual orbits only through a
        // guarded diagnostic API that returns and validates unfiltered generators.
      } else if (level == "probing") {
        // RESEARCH-BREADCRUMB(mps-structure/probing-overlay) [solver-invasive]
        // Keep this rejected until probing can return an immutable, copy-safe, budgeted snapshot
        // with infeasible probes, target bounds, source reasons, and explicit completeness.
        throw std::invalid_argument(
          "probing diagnostics are not exposed safely by the current internal API");
      } else {
        throw std::invalid_argument("unknown analysis level: " + std::string(level));
      }
    } else if (value == "--objective-erased-lp") {
      options.objective_erased_lp = true;
      options.lp_overlay          = true;
    } else {
      throw std::invalid_argument("unknown option: " + std::string(value));
    }
  }
  return options;
}

void restore_reduced_column_names(model_t& reduced,
                                  const model_t& original,
                                  const std::vector<index_t>& reduced_to_original)
{
  if (!reduced.get_variable_names().empty() || original.get_variable_names().empty()) { return; }
  std::vector<std::string> names(reduced.get_n_variables());
  for (index_t column = 0; column < reduced.get_n_variables(); ++column) {
    const auto original_column = reduced_to_original[column];
    if (original_column >= 0 &&
        original_column < static_cast<index_t>(original.get_variable_names().size())) {
      names[column] = original.get_variable_names()[original_column];
    }
  }
  reduced.set_variable_names(names);
}

void restore_reduced_row_names(model_t& reduced,
                               const model_t& original,
                               const std::vector<index_t>& reduced_to_original)
{
  if (!reduced.get_row_names().empty() || original.get_row_names().empty()) { return; }
  std::vector<std::string> names(reduced.get_n_constraints());
  for (index_t row = 0; row < reduced.get_n_constraints(); ++row) {
    const auto original_row = reduced_to_original[row];
    if (original_row >= 0 && original_row < static_cast<index_t>(original.get_row_names().size())) {
      names[row] = original.get_row_names()[original_row];
    }
  }
  reduced.set_row_names(names);
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    const auto options = parse_options(argc, argv);

    // RESEARCH-BREADCRUMB(mps-structure/safe-json-sidecar) [driver-local]
    // Before reading or analyzing, reject an output equivalent to the input, including aliases.
    // Commit a completed sidecar through a same-directory temporary file and atomic rename.

    // RESEARCH-BREADCRUMB(mps-structure/run-manifest) [driver-local]
    // Capture input identity, effective settings, statuses, timings, and build identity once here;
    // pass that typed manifest to every renderer instead of reconstructing it during JSON output.
    auto original = cuopt::mathematical_optimization::io::read_mps<index_t, value_t>(
      options.model_path, options.fixed_format);
    const auto original_analysis =
      structure::analyze_structure(original, "original", {}, options.analysis_level);
    print_model_summary("Original model", original);
    structure::print_special_structure_summary(original, original_analysis);

    std::optional<structure::lp_overlay_summary_t> source_lp;
    std::optional<structure::lp_overlay_summary_t> objective_erased_lp;
    if (options.lp_overlay) {
      // RESEARCH-BREADCRUMB(mps-structure/lp-structure-overlays) [driver-local]
      // Make the requested stage explicit before adding a presolved LP overlay; comparisons must
      // retain each stage's coordinate system and mapping completeness.
      structure::lp_overlay_options_t lp_options;
      source_lp = structure::analyze_root_lp_overlay(original, original_analysis, lp_options);
      structure::print_root_lp_overlay_summary(*source_lp);
      if (options.objective_erased_lp) {
        lp_options.objective_mode = structure::lp_objective_mode_t::erased;
        objective_erased_lp =
          structure::analyze_root_lp_overlay(original, original_analysis, lp_options);
        structure::print_root_lp_overlay_summary(*objective_erased_lp);
      }
    }

    if (original.has_quadratic_objective() || original.has_quadratic_constraints()) {
      throw std::invalid_argument(
        "This presolve example supports linear LP/MIP models, not QP/QCQP models.");
    }
    if (has_semi_continuous_variables(original)) {
      throw std::invalid_argument(
        "This host-only example does not perform cuOpt's GPU-side semi-continuous variable "
        "reformulation before presolve.");
    }

    // The internal PaPILO adapter distinguishes LP from models containing any discrete variable.
    const auto category = has_discrete_variables(original)
                            ? cuopt::mathematical_optimization::problem_category_t::MIP
                            : cuopt::mathematical_optimization::problem_category_t::LP;
    cuopt::mathematical_optimization::mip::third_party_presolve_t<index_t, value_t> presolver;
    auto result = presolver.apply_presolve_from_mps_data(
      original,
      category,
      cuopt::mathematical_optimization::presolver_t::Papilo,
      false,  // Dual postsolve data is unnecessary for structural inspection.
      1e-6,
      1e-12,
      20.0,
      1);

    std::cout << "\nPresolve status: " << status_name(result.status) << '\n';
    if (is_terminal_without_model(result.status)) {
      if (options.json_path) {
        structure::write_structure_json(*options.json_path,
                                        original,
                                        original_analysis,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        source_lp ? &*source_lp : nullptr,
                                        objective_erased_lp ? &*objective_erased_lp : nullptr);
      }
      return 0;
    }

    restore_reduced_column_names(result.reduced_problem, original, result.reduced_to_original_map);
    const auto& reduced_to_original_rows = presolver.get_reduced_to_original_row_map();
    restore_reduced_row_names(result.reduced_problem, original, reduced_to_original_rows);
    const auto reduced_analysis = structure::analyze_structure(result.reduced_problem,
                                                               "presolved",
                                                               result.reduced_to_original_map,
                                                               options.analysis_level,
                                                               reduced_to_original_rows);
    print_model_summary("Presolved model", result.reduced_problem);
    structure::print_special_structure_summary(result.reduced_problem, reduced_analysis);
    std::cout << "\nPresolve reduction\n";
    print_reduction("rows",
                    static_cast<std::size_t>(original.get_n_constraints()),
                    static_cast<std::size_t>(result.reduced_problem.get_n_constraints()));
    print_reduction("columns",
                    static_cast<std::size_t>(original.get_n_variables()),
                    static_cast<std::size_t>(result.reduced_problem.get_n_variables()));
    print_reduction("nonzeros",
                    static_cast<std::size_t>(original.get_nnz()),
                    static_cast<std::size_t>(result.reduced_problem.get_nnz()));
    std::cout << "  implied integer columns in reduced model: "
              << result.implied_integer_indices.size() << '\n';
    std::cout << "  surviving column mappings: " << result.reduced_to_original_map.size() << '\n';
    std::cout << "  surviving row mappings: " << reduced_to_original_rows.size() << '\n';
    const auto structure_delta =
      structure::analyze_presolve_structure_delta(original_analysis,
                                                  reduced_analysis,
                                                  result.reduced_to_original_map,
                                                  result.original_to_reduced_map,
                                                  result.implied_integer_indices);
    structure::print_presolve_structure_delta(structure_delta);
    if (options.json_path) {
      structure::write_structure_json(*options.json_path,
                                      original,
                                      original_analysis,
                                      &result.reduced_problem,
                                      &reduced_analysis,
                                      &structure_delta,
                                      source_lp ? &*source_lp : nullptr,
                                      objective_erased_lp ? &*objective_erased_lp : nullptr);
      std::cout << "\nWrote structural detail to " << *options.json_path << '\n';
    }
  } catch (const std::exception& error) {
    print_usage(argv[0]);
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
