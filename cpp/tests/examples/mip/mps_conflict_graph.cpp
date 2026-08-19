/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/mathematical_optimization/cpu_optimization_problem.hpp>
#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>
#include <cuopt/mathematical_optimization/optimization_problem_interface.hpp>
#include <cuopt/mathematical_optimization/optimization_problem_utils.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_solution.hpp>
#include <mip_heuristics/presolve/conflict_graph/clique_table.cuh>
#include <utilities/timer.hpp>

#include "mps_conflict_graph.hpp"

namespace cuopt::mathematical_optimization::mip {
template <typename i_t, typename f_t>
class problem_t;
}  // namespace cuopt::mathematical_optimization::mip

#include <pdlp/translate.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace cuopt::mathematical_optimization::examples {
namespace {

using index_t = int;
using value_t = double;
using model_t = io::mps_data_model_t<index_t, value_t>;

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

struct degree_summary_t {
  index_t minimum{};
  index_t maximum{};
  double mean{};
};

degree_summary_t summarize_degrees(const std::vector<index_t>& degrees)
{
  if (degrees.empty()) { return {}; }
  const auto [minimum, maximum] = std::minmax_element(degrees.begin(), degrees.end());
  return degree_summary_t{
    *minimum,
    *maximum,
    static_cast<double>(std::accumulate(degrees.begin(), degrees.end(), std::int64_t{0})) /
      static_cast<double>(degrees.size())};
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

  cpu_optimization_problem_t<index_t, value_t> cpu_problem;
  populate_from_mps_data_model(&cpu_problem, model);
  auto user_problem = cuopt_problem_to_user_problem<index_t, value_t>(nullptr, cpu_problem);

  mip::clique_config_t config;
  mip::clique_table_t<index_t, value_t> clique_table(
    2 * n_columns, config.min_clique_size, config.max_clique_size_for_extension);
  using settings_t = mip_solver_settings_t<index_t, value_t>;
  cuopt::timer_t timer(std::numeric_limits<double>::infinity());
  mip::build_clique_table(
    user_problem, clique_table, typename settings_t::tolerances_t{}, false, true, timer);

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

}  // namespace

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
    std::cout << "    not applicable (no unfixed binary variables)\n";
    return;
  }

  const auto total_edges    = summary.row_derived_edges + summary.complement_edges;
  const auto degree_summary = summarize_degrees(summary.literal_degrees);
  std::cout << "    unfixed binary variables: " << summary.binary_variables << " ("
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

}  // namespace cuopt::mathematical_optimization::examples
