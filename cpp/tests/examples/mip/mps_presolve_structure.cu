/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/mathematical_optimization/cpu_optimization_problem.hpp>
#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>
#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/optimization_problem_interface.hpp>
#include <cuopt/mathematical_optimization/optimization_problem_utils.hpp>
#include <cuopt/mathematical_optimization/utilities/internals.hpp>
#include <mip_heuristics/presolve/conflict_graph/clique_table.cuh>
#include <mip_heuristics/presolve/third_party_presolve.hpp>
#include <mip_heuristics/problem/problem.cuh>
#include <pdlp/translate.hpp>
#include <utilities/timer.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

// The internal problem translation headers require this example to be compiled by NVCC.
namespace {

using index_t           = int;
using value_t           = double;
using model_t           = cuopt::mathematical_optimization::io::mps_data_model_t<index_t, value_t>;
using presolve_status_t = cuopt::mathematical_optimization::mip::third_party_presolve_status_t;

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

struct conflict_graph_summary_t {
  std::size_t binary_variables{};
  std::size_t base_cliques{};
  std::size_t clique_extensions{};
  std::size_t largest_clique{};
  double mean_clique_size{};
  std::size_t row_derived_edges{};
  std::size_t complement_edges{};
  std::size_t components{};
  std::size_t largest_component{};
  std::vector<index_t> literal_degrees;
  std::vector<index_t> literal_ids;
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

bool is_binary_column(const model_t& model, std::size_t column)
{
  const auto& types = model.get_variable_types();
  const auto& lower = model.get_variable_lower_bounds();
  const auto& upper = model.get_variable_upper_bounds();
  return column < types.size() && column < lower.size() && column < upper.size() &&
         (types[column] == 'I' || types[column] == 'B') && lower[column] == 0.0 &&
         upper[column] == 1.0;
}

conflict_graph_summary_t inspect_conflict_graph(const model_t& model)
{
  conflict_graph_summary_t summary;
  const auto n_columns = model.get_n_variables();
  std::vector<index_t> literal_to_local(static_cast<std::size_t>(2 * n_columns), -1);
  for (index_t column = 0; column < n_columns; ++column) {
    if (!is_binary_column(model, static_cast<std::size_t>(column))) { continue; }
    literal_to_local[column] = static_cast<index_t>(summary.literal_ids.size());
    summary.literal_ids.push_back(column);
    literal_to_local[column + n_columns] = static_cast<index_t>(summary.literal_ids.size());
    summary.literal_ids.push_back(column + n_columns);
    ++summary.binary_variables;
  }
  if (summary.binary_variables == 0) { return summary; }

  cuopt::mathematical_optimization::cpu_optimization_problem_t<index_t, value_t> cpu_problem;
  cuopt::mathematical_optimization::populate_from_mps_data_model(&cpu_problem, model);
  auto user_problem =
    cuopt::mathematical_optimization::cuopt_problem_to_user_problem<index_t, value_t>(nullptr,
                                                                                      cpu_problem);

  cuopt::mathematical_optimization::mip::clique_config_t config;
  cuopt::mathematical_optimization::mip::clique_table_t<index_t, value_t> clique_table(
    2 * n_columns, config.min_clique_size, config.max_clique_size_for_extension);
  using settings_t = cuopt::mathematical_optimization::mip_solver_settings_t<index_t, value_t>;
  cuopt::timer_t timer(std::numeric_limits<double>::infinity());
  cuopt::mathematical_optimization::mip::build_clique_table(
    user_problem,
    clique_table,
    typename settings_t::tolerances_t{},
    false,  // Preserve cliques rather than demoting small ones to pairwise storage.
    true,
    timer);

  summary.base_cliques           = clique_table.first.size();
  summary.clique_extensions      = clique_table.addtl_cliques.size();
  std::size_t clique_memberships = 0;
  for (const auto& clique : clique_table.first) {
    summary.largest_clique = std::max(summary.largest_clique, clique.size());
    clique_memberships += clique.size();
  }
  if (summary.base_cliques != 0) {
    summary.mean_clique_size =
      static_cast<double>(clique_memberships) / static_cast<double>(summary.base_cliques);
  }

  summary.literal_degrees.resize(summary.literal_ids.size(), 0);
  disjoint_set_t components(summary.literal_ids.size());
  for (std::size_t local_literal = 0; local_literal < summary.literal_ids.size(); ++local_literal) {
    const auto literal   = summary.literal_ids[local_literal];
    const auto adjacency = clique_table.get_adj_set_of_var(literal);
    for (const auto adjacent_literal : adjacency) {
      if (adjacent_literal < 0 ||
          adjacent_literal >= static_cast<index_t>(literal_to_local.size())) {
        continue;
      }
      const auto adjacent_local = literal_to_local[adjacent_literal];
      if (adjacent_local < 0) { continue; }
      ++summary.literal_degrees[local_literal];
      if (local_literal >= static_cast<std::size_t>(adjacent_local)) { continue; }

      components.unite(local_literal, static_cast<std::size_t>(adjacent_local));
      const auto complement = literal < n_columns ? literal + n_columns : literal - n_columns;
      if (adjacent_literal == complement) {
        ++summary.complement_edges;
      } else {
        ++summary.row_derived_edges;
      }
    }
  }

  std::unordered_map<std::size_t, std::size_t> component_sizes;
  for (std::size_t literal = 0; literal < summary.literal_ids.size(); ++literal) {
    auto& size = component_sizes[components.find(literal)];
    ++size;
    summary.largest_component = std::max(summary.largest_component, size);
  }
  summary.components = component_sizes.size();
  return summary;
}

std::string literal_name(index_t literal, const model_t& model)
{
  const auto n_columns = model.get_n_variables();
  const auto column    = literal < n_columns ? literal : literal - n_columns;
  const auto& names    = model.get_variable_names();
  const auto name      = column < static_cast<index_t>(names.size()) && !names[column].empty()
                           ? names[column]
                           : "x[" + std::to_string(column) + "]";
  return name + (literal < n_columns ? "=1" : "=0");
}

void print_conflict_graph_summary(const model_t& model)
{
  if (model.has_quadratic_objective() || model.has_quadratic_constraints()) {
    std::cout << "  row-derived conflict graph:\n"
              << "    unavailable for quadratic models in this example\n";
    return;
  }
  const auto summary = inspect_conflict_graph(model);
  std::cout << "  row-derived conflict graph:\n";
  if (summary.binary_variables == 0) {
    std::cout << "    not applicable (no binary variables)\n";
    return;
  }

  const auto total_edges    = summary.row_derived_edges + summary.complement_edges;
  const auto degree_summary = summarize_degrees(summary.literal_degrees);
  std::cout << "    binary variables: " << summary.binary_variables << " ("
            << summary.literal_ids.size() << " literal vertices)\n";
  std::cout << "    base cliques: " << summary.base_cliques
            << ", extensions: " << summary.clique_extensions << ", mean base size=" << std::fixed
            << std::setprecision(2) << summary.mean_clique_size
            << ", largest=" << summary.largest_clique << '\n';
  std::cout << "    edges: row-derived=" << summary.row_derived_edges
            << ", literal-complement=" << summary.complement_edges << ", total=" << total_edges
            << '\n';
  std::cout << "    literal degree (including complement): min=" << degree_summary.minimum
            << ", mean=" << std::fixed << std::setprecision(2) << degree_summary.mean
            << ", max=" << degree_summary.maximum << '\n';
  std::cout << "    connected components: " << summary.components
            << " (largest: " << summary.largest_component << " literals)\n";

  if (summary.row_derived_edges == 0) { return; }
  constexpr std::size_t limit = 5;
  std::vector<std::size_t> order;
  order.reserve(summary.literal_ids.size());
  for (std::size_t literal = 0; literal < summary.literal_ids.size(); ++literal) {
    if (summary.literal_degrees[literal] > 1) { order.push_back(literal); }
  }
  const auto count = std::min(limit, order.size());
  std::partial_sort(order.begin(),
                    order.begin() + static_cast<std::ptrdiff_t>(count),
                    order.end(),
                    [&summary](std::size_t lhs, std::size_t rhs) {
                      if (summary.literal_degrees[lhs] != summary.literal_degrees[rhs]) {
                        return summary.literal_degrees[lhs] > summary.literal_degrees[rhs];
                      }
                      return summary.literal_ids[lhs] < summary.literal_ids[rhs];
                    });
  std::cout << "    most-conflicted literals:\n";
  for (std::size_t rank = 0; rank < count; ++rank) {
    const auto local = order[rank];
    std::cout << "      " << literal_name(summary.literal_ids[local], model) << ": degree "
              << summary.literal_degrees[local] << '\n';
  }
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
  const auto& variable_types  = model.get_variable_types();
  const auto& variable_lower  = model.get_variable_lower_bounds();
  const auto& variable_upper  = model.get_variable_upper_bounds();
  for (std::size_t column = 0; column < variable_types.size(); ++column) {
    const auto type = variable_types[column];
    if (type == 'I' || type == 'B') {
      ++integer;
      if (column < variable_lower.size() && column < variable_upper.size() &&
          variable_lower[column] == 0.0 && variable_upper[column] == 1.0) {
        ++binary;
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
            << " (binary=" << binary << "), semi-continuous=" << semi_continuous << '\n';
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
  print_conflict_graph_summary(model);
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
  std::cerr << "Usage: " << executable << " <model.mps> [--fixed]\n";
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc < 2 || argc > 3) {
    print_usage(argv[0]);
    return 1;
  }

  bool fixed_format = false;
  if (argc == 3) {
    if (std::string_view(argv[2]) != "--fixed") {
      print_usage(argv[0]);
      return 1;
    }
    fixed_format = true;
  }

  try {
    auto original =
      cuopt::mathematical_optimization::io::read_mps<index_t, value_t>(argv[1], fixed_format);
    print_model_summary("Original model", original);

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
    if (is_terminal_without_model(result.status)) { return 0; }

    print_model_summary("Presolved model", result.reduced_problem);
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
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
