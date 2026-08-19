/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "mps_structure_analysis.hpp"

#include "mps_lp_overlay.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::examples {
namespace {

using row_t = std::vector<std::pair<structure_index_t, structure_value_t>>;

constexpr auto infinity = std::numeric_limits<structure_value_t>::infinity();
constexpr std::size_t size_zero{0};
constexpr std::size_t size_one{1};
constexpr std::size_t size_two{2};
constexpr std::size_t size_three{3};
constexpr std::size_t size_four{4};
constexpr std::size_t size_eight{8};

structure_model_t make_model(const std::vector<row_t>& rows,
                             const std::vector<structure_value_t>& row_lower,
                             const std::vector<structure_value_t>& row_upper,
                             const std::vector<char>& variable_types,
                             const std::vector<structure_value_t>& variable_lower,
                             const std::vector<structure_value_t>& variable_upper,
                             std::string problem_name = "structure-test")
{
  if (rows.size() != row_lower.size() || rows.size() != row_upper.size() ||
      variable_types.size() != variable_lower.size() ||
      variable_types.size() != variable_upper.size()) {
    throw std::invalid_argument("inconsistent synthetic model dimensions");
  }

  std::vector<structure_value_t> values;
  std::vector<structure_index_t> indices;
  std::vector<structure_index_t> offsets{0};
  for (const auto& row : rows) {
    for (const auto& [column, value] : row) {
      if (column < 0 || static_cast<std::size_t>(column) >= variable_types.size()) {
        throw std::invalid_argument("synthetic model column is out of range");
      }
      indices.push_back(column);
      values.push_back(value);
    }
    offsets.push_back(static_cast<structure_index_t>(values.size()));
  }

  std::vector<structure_value_t> objective(variable_types.size(), 0.0);
  std::vector<std::string> variable_names;
  std::vector<std::string> row_names;
  for (std::size_t column = 0; column < variable_types.size(); ++column) {
    variable_names.push_back("x" + std::to_string(column));
  }
  for (std::size_t row = 0; row < rows.size(); ++row) {
    row_names.push_back("r" + std::to_string(row));
  }

  structure_model_t model;
  model.set_csr_constraint_matrix(values, indices, offsets);
  model.set_objective_coefficients(objective);
  model.set_variable_types(variable_types);
  model.set_variable_lower_bounds(variable_lower);
  model.set_variable_upper_bounds(variable_upper);
  model.set_constraint_lower_bounds(row_lower);
  model.set_constraint_upper_bounds(row_upper);
  model.set_variable_names(variable_names);
  model.set_row_names(row_names);
  model.set_problem_name(problem_name);
  model.set_objective_name("objective");
  return model;
}

bool has_family(const row_record_t& row, std::string_view family)
{
  return std::find(row.families.begin(), row.families.end(), family) != row.families.end();
}

const graph_projection_summary_t& find_projection(const model_analysis_t& analysis,
                                                  std::string_view name)
{
  const auto position =
    std::find_if(analysis.exact_one_block_projections.begin(),
                 analysis.exact_one_block_projections.end(),
                 [name](const auto& projection) { return projection.name == name; });
  if (position == analysis.exact_one_block_projections.end()) {
    throw std::logic_error("missing exact-one block projection");
  }
  return *position;
}

const decomposition_t& find_decomposition(const model_analysis_t& analysis, std::string_view name)
{
  const auto position = std::find_if(analysis.decompositions.begin(),
                                     analysis.decompositions.end(),
                                     [&](const auto& d) { return d.name == name; });
  if (position == analysis.decompositions.end()) {
    throw std::logic_error("missing domain decomposition");
  }
  return *position;
}

bool has_implication(const model_analysis_t& analysis,
                     structure_index_t row,
                     literal_t antecedent,
                     literal_t consequent)
{
  return std::any_of(
    analysis.implications.begin(), analysis.implications.end(), [&](const auto& implication) {
      return implication.row == row && implication.antecedent.column == antecedent.column &&
             implication.antecedent.value == antecedent.value &&
             implication.consequent.column == consequent.column &&
             implication.consequent.value == consequent.value;
    });
}

bool has_balanced_json_containers(std::string_view document)
{
  const auto first = document.find_first_not_of(" \t\r\n");
  const auto last  = document.find_last_not_of(" \t\r\n");
  if (first == std::string_view::npos || document[first] != '{' || document[last] != '}') {
    return false;
  }

  std::vector<char> containers;
  bool in_string = false;
  bool escaped   = false;
  for (const auto character : document) {
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (character == '\\') {
        escaped = true;
      } else if (character == '"') {
        in_string = false;
      } else if (static_cast<unsigned char>(character) < 0x20) {
        return false;
      }
      continue;
    }

    if (character == '"') {
      in_string = true;
    } else if (character == '{' || character == '[') {
      containers.push_back(character);
    } else if (character == '}' || character == ']') {
      if (containers.empty()) { return false; }
      const auto expected = character == '}' ? '{' : '[';
      if (containers.back() != expected) { return false; }
      containers.pop_back();
    }
  }
  return !in_string && !escaped && containers.empty();
}

class temporary_file_t {
 public:
  explicit temporary_file_t(std::filesystem::path path) : path_(std::move(path)) {}
  ~temporary_file_t()
  {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

TEST(MpsStructureAnalysis, PreservesComplementedAndFixedBinaryExactOneProvenanceAndOverlap)
{
  const auto model = make_model({{{0, -1.0}, {1, 1.0}, {2, 1.0}}, {{2, 1.0}, {3, 1.0}}},
                                {0.0, 1.0},
                                {0.0, 1.0},
                                {'B', 'B', 'B', 'B'},
                                {0.0, 0.0, 0.0, 0.0},
                                {1.0, 1.0, 1.0, 0.0});

  const auto analysis = analyze_structure(model, "original", {}, analysis_level_t::basic);

  ASSERT_EQ(analysis.exact_one_groups.size(), size_two);
  ASSERT_EQ(analysis.exact_one_groups[0].members.size(), size_three);
  EXPECT_EQ(analysis.exact_one_groups[0].stable_id, "original:r0");
  EXPECT_EQ(analysis.exact_one_groups[0].members[0].column, 0);
  EXPECT_FALSE(analysis.exact_one_groups[0].members[0].value);
  EXPECT_EQ(analysis.exact_one_groups[0].members[1].column, 1);
  EXPECT_TRUE(analysis.exact_one_groups[0].members[1].value);
  EXPECT_EQ(analysis.exact_one_groups[0].members[2].column, 2);
  EXPECT_TRUE(analysis.exact_one_groups[0].members[2].value);
  EXPECT_TRUE(has_family(analysis.rows[0], "exact_one"));
  EXPECT_FALSE(has_family(analysis.rows[0], "set_packing"));
  EXPECT_FALSE(has_family(analysis.rows[0], "set_covering"));

  EXPECT_EQ(analysis.binary_variables, size_four);
  EXPECT_EQ(analysis.exact_one_covered_binary_variables, size_four);
  EXPECT_TRUE(analysis.exact_one_groups_overlap);
  ASSERT_GT(analysis.exact_one_group_size_histogram.size(), size_three);
  EXPECT_EQ(analysis.exact_one_group_size_histogram[2], size_one);
  EXPECT_EQ(analysis.exact_one_group_size_histogram[3], size_one);
  EXPECT_EQ(analysis.exact_one_overlap_component_sizes, std::vector<std::size_t>({2}));
}

TEST(MpsStructureAnalysis, RecognizesSingletonBinaryExactOneRows)
{
  const auto model = make_model({{{0, 1.0}}}, {1.0}, {1.0}, {'B'}, {0.0}, {1.0});

  const auto analysis = analyze_structure(model, "singleton", {}, analysis_level_t::basic);

  ASSERT_EQ(analysis.exact_one_groups.size(), size_one);
  ASSERT_EQ(analysis.exact_one_groups[0].members.size(), size_one);
  EXPECT_EQ(analysis.exact_one_groups[0].members[0].column, 0);
  EXPECT_TRUE(analysis.exact_one_groups[0].members[0].value);
  EXPECT_TRUE(has_family(analysis.rows[0], "exact_one"));
  EXPECT_TRUE(has_family(analysis.rows[0], "singleton_equality"));
  EXPECT_EQ(analysis.binary_variables, size_one);
  EXPECT_EQ(analysis.exact_one_covered_binary_variables, size_one);
}

TEST(MpsStructureAnalysis, NormalizesScaledGroupsAndClassifiesBinaryCapacities)
{
  const auto model = make_model({{{0, -2.0}, {1, 2.0}, {2, 2.0}},
                                 {{0, 1.0}, {1, 1.0}, {2, 1.0}},
                                 {{0, 1.0}, {1, 1.0}, {2, 1.0}},
                                 {{0, 2.0}, {1, 3.0}}},
                                {0.0, -infinity, 2.0, -infinity},
                                {0.0, 2.0, infinity, 4.0},
                                {'B', 'B', 'B'},
                                {0.0, 0.0, 0.0},
                                {1.0, 1.0, 1.0});

  const auto analysis = analyze_structure(model, "capacities", {}, analysis_level_t::basic);

  ASSERT_EQ(analysis.exact_one_groups.size(), size_one);
  const auto& exact_one = analysis.exact_one_groups[0];
  ASSERT_EQ(exact_one.members.size(), size_three);
  EXPECT_EQ(exact_one.members[0].column, 0);
  EXPECT_FALSE(exact_one.members[0].value);
  EXPECT_EQ(exact_one.members[1].column, 1);
  EXPECT_TRUE(exact_one.members[1].value);
  EXPECT_EQ(exact_one.members[2].column, 2);
  EXPECT_TRUE(exact_one.members[2].value);
  EXPECT_EQ(exact_one.member_coefficients, std::vector<double>({1.0, 1.0, 1.0}));
  EXPECT_DOUBLE_EQ(exact_one.lower_capacity, 1.0);
  EXPECT_DOUBLE_EQ(exact_one.upper_capacity, 1.0);

  EXPECT_TRUE(has_family(analysis.rows[1], "gub"));
  EXPECT_TRUE(has_family(analysis.rows[1], "binary_cardinality"));
  EXPECT_TRUE(has_family(analysis.rows[2], "binary_cardinality"));
  EXPECT_TRUE(has_family(analysis.rows[3], "binary_knapsack"));
  ASSERT_EQ(analysis.knapsack_rows.size(), size_three);
  EXPECT_EQ(analysis.knapsack_rows[0].family, "binary_cardinality");
  EXPECT_EQ(analysis.knapsack_rows[0].member_coefficients, std::vector<double>({1.0, 1.0, 1.0}));
  EXPECT_DOUBLE_EQ(analysis.knapsack_rows[0].upper_capacity, 2.0);
  EXPECT_EQ(analysis.knapsack_rows[1].family, "binary_cardinality");
  EXPECT_DOUBLE_EQ(analysis.knapsack_rows[1].lower_capacity, 2.0);
  EXPECT_EQ(analysis.knapsack_rows[2].family, "binary_knapsack");
  EXPECT_EQ(analysis.knapsack_rows[2].member_coefficients, std::vector<double>({2.0, 3.0}));
  EXPECT_DOUBLE_EQ(analysis.knapsack_rows[2].upper_capacity, 4.0);
}

TEST(MpsStructureAnalysis, ReportsNonexclusiveRowFamiliesRelationsAndAffinePivots)
{
  const auto model =
    make_model({{{0, 1.0}, {1, 1.0}},
                {{0, 1.0}, {1, 1.0}},
                {{0, 1.0}, {1, 1.0}, {2, 1.0}},
                {{0, 2.0}, {1, 3.0}},
                {{0, 1.0}, {1, -1.0}},
                {{0, -5.0}, {3, 1.0}},
                {{4, 1.0}, {5, 1.0}},
                {{4, 1.0}}},
               {-infinity, 1.0, -infinity, -infinity, -infinity, -infinity, 2.0, 1.0},
               {1.0, infinity, 2.0, 3.0, 0.0, 0.0, 2.0, 1.0},
               {'B', 'B', 'B', 'C', 'C', 'C'},
               {0.0, 0.0, 0.0, 0.0, -infinity, 0.0},
               {1.0, 1.0, 1.0, infinity, infinity, 10.0});

  const auto analysis = analyze_structure(model, "relations", {}, analysis_level_t::basic);

  ASSERT_EQ(analysis.rows.size(), size_eight);
  EXPECT_TRUE(has_family(analysis.rows[0], "set_packing"));
  EXPECT_TRUE(has_family(analysis.rows[0], "binary_implication"));
  EXPECT_TRUE(has_family(analysis.rows[1], "set_covering"));
  EXPECT_TRUE(has_family(analysis.rows[1], "binary_implication"));
  EXPECT_TRUE(has_family(analysis.rows[2], "gub"));
  EXPECT_TRUE(has_family(analysis.rows[3], "binary_knapsack"));
  EXPECT_TRUE(has_family(analysis.rows[3], "binary_implication"));
  EXPECT_TRUE(has_family(analysis.rows[4], "set_packing"));
  EXPECT_TRUE(has_family(analysis.rows[4], "binary_implication"));

  EXPECT_TRUE(has_implication(analysis, 4, literal_t{0, true}, literal_t{1, true}));
  EXPECT_TRUE(has_implication(analysis, 4, literal_t{1, false}, literal_t{0, false}));
  EXPECT_EQ(std::count_if(analysis.implications.begin(),
                          analysis.implications.end(),
                          [](const auto& implication) { return implication.row == 4; }),
            std::ptrdiff_t{2});

  const auto activation = std::find_if(analysis.variable_bounds.begin(),
                                       analysis.variable_bounds.end(),
                                       [](const auto& b) { return b.row == 5 && b.activation; });
  ASSERT_NE(activation, analysis.variable_bounds.end());
  EXPECT_EQ(activation->binary_column, 0);
  EXPECT_EQ(activation->target_column, 3);
  EXPECT_EQ(activation->bound_type, "upper");
  EXPECT_DOUBLE_EQ(activation->bound_when_zero, 0.0);
  EXPECT_DOUBLE_EQ(activation->bound_when_one, 5.0);
  EXPECT_TRUE(has_family(analysis.rows[5], "variable_bound"));
  EXPECT_TRUE(has_family(analysis.rows[5], "activation"));

  const auto two_variable =
    std::find_if(analysis.affine_definitions.begin(),
                 analysis.affine_definitions.end(),
                 [](const auto& definition) { return definition.row == 6; });
  ASSERT_NE(two_variable, analysis.affine_definitions.end());
  EXPECT_TRUE(two_variable->two_variable);
  EXPECT_FALSE(two_variable->singleton);
  EXPECT_EQ(two_variable->pivot_columns, std::vector<structure_index_t>({4}));
  const auto singleton = std::find_if(analysis.affine_definitions.begin(),
                                      analysis.affine_definitions.end(),
                                      [](const auto& definition) { return definition.row == 7; });
  ASSERT_NE(singleton, analysis.affine_definitions.end());
  EXPECT_TRUE(singleton->singleton);
  EXPECT_TRUE(has_family(analysis.rows[7], "singleton_equality"));
  EXPECT_EQ(analysis.repair.free_equality_pivots, std::vector<structure_index_t>({4}));
  EXPECT_NE(std::find(analysis.repair.two_variable_affine_rows.begin(),
                      analysis.repair.two_variable_affine_rows.end(),
                      6),
            analysis.repair.two_variable_affine_rows.end());
}

TEST(MpsStructureAnalysis, SeparatesDirectIntegerAndContinuousBlockProjections)
{
  const auto model =
    make_model({{{0, 1.0}, {1, 1.0}},
                {{2, 1.0}, {3, 1.0}},
                {{4, 1.0}, {5, 1.0}},
                {{0, 1.0}, {2, 1.0}},
                {{1, 1.0}, {6, 1.0}},
                {{4, 1.0}, {6, 1.0}},
                {{3, 1.0}, {7, 1.0}},
                {{5, 1.0}, {7, 1.0}}},
               {1.0, 1.0, 1.0, -infinity, -infinity, -infinity, -infinity, -infinity},
               {1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0},
               {'B', 'B', 'B', 'B', 'B', 'B', 'I', 'C'},
               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
               {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, infinity});

  const auto analysis = analyze_structure(model, "blocks", {}, analysis_level_t::basic);

  ASSERT_EQ(analysis.exact_one_groups.size(), size_three);
  ASSERT_EQ(analysis.exact_one_block_edges.size(), size_three);
  const auto& direct     = find_projection(analysis, "direct_rows");
  const auto& integer    = find_projection(analysis, "integer_mediators");
  const auto& continuous = find_projection(analysis, "continuous_mediators");
  const auto& mediated   = find_projection(analysis, "either_mediator");
  const auto& all        = find_projection(analysis, "all_coupling");

  EXPECT_EQ(direct.edges, size_one);
  EXPECT_EQ(direct.total_weight, size_one);
  EXPECT_EQ(direct.components, size_two);
  EXPECT_EQ(direct.isolated_vertices, size_one);
  EXPECT_EQ(integer.edges, size_one);
  EXPECT_EQ(integer.total_weight, size_one);
  EXPECT_EQ(continuous.edges, size_one);
  EXPECT_EQ(continuous.total_weight, size_one);
  EXPECT_EQ(mediated.edges, size_two);
  EXPECT_EQ(mediated.components, size_one);
  EXPECT_EQ(mediated.largest_component, size_three);
  EXPECT_EQ(all.edges, size_three);
  EXPECT_EQ(all.total_weight, size_three);
  EXPECT_EQ(all.components, size_one);

  const auto edge_has_provenance = [&](structure_index_t first,
                                       structure_index_t second,
                                       std::string_view kind,
                                       structure_index_t provenance) {
    const auto position =
      std::find_if(analysis.exact_one_block_edges.begin(),
                   analysis.exact_one_block_edges.end(),
                   [&](const auto& edge) { return edge.first == first && edge.second == second; });
    if (position == analysis.exact_one_block_edges.end()) { return false; }
    const auto* values = kind == "row"       ? &position->direct_rows
                         : kind == "integer" ? &position->integer_mediators
                                             : &position->continuous_mediators;
    return std::find(values->begin(), values->end(), provenance) != values->end();
  };
  EXPECT_TRUE(edge_has_provenance(0, 1, "row", 3));
  EXPECT_TRUE(edge_has_provenance(0, 2, "integer", 6));
  EXPECT_TRUE(edge_has_provenance(1, 2, "continuous", 7));
}

TEST(MpsStructureAnalysis, CapsPairMaterializationButRetainsCompleteMediatorRowPaths)
{
  const auto model = make_model({{{0, 1.0}, {1, 1.0}},
                                 {{2, 1.0}, {3, 1.0}},
                                 {{4, 1.0}, {5, 1.0}},
                                 {{0, 1.0}, {6, 1.0}},
                                 {{2, 1.0}, {6, 1.0}},
                                 {{4, 1.0}, {6, 1.0}}},
                                {1.0, 1.0, 1.0, -infinity, -infinity, -infinity},
                                {1.0, 1.0, 1.0, 10.0, 10.0, 10.0},
                                {'B', 'B', 'B', 'B', 'B', 'B', 'C'},
                                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, infinity},
                                "capped-mediator");

  const auto analysis =
    analyze_structure(model, "capped", {}, analysis_level_t::basic, {}, size_one);

  EXPECT_FALSE(analysis.exact_one_block_edges_complete);
  EXPECT_EQ(analysis.exact_one_block_pair_provenance_limit, size_one);
  EXPECT_EQ(analysis.exact_one_block_candidate_pair_provenance.at("continuous_mediators"),
            size_three);
  EXPECT_EQ(analysis.exact_one_block_materialized_pair_provenance.at("continuous_mediators"),
            size_one);
  ASSERT_EQ(analysis.exact_one_block_edges.size(), size_one);
  const auto& continuous = find_projection(analysis, "continuous_mediators");
  EXPECT_FALSE(continuous.complete);
  EXPECT_EQ(continuous.edges, size_one);

  ASSERT_EQ(analysis.exact_one_block_mediators.size(), size_one);
  const auto& mediator = analysis.exact_one_block_mediators[0];
  EXPECT_EQ(mediator.column, 6);
  EXPECT_EQ(mediator.kind, "continuous");
  ASSERT_EQ(mediator.incidences.size(), size_three);
  EXPECT_EQ(mediator.incidences[0].block, 0);
  EXPECT_EQ(mediator.incidences[0].rows, std::vector<structure_index_t>({3}));
  EXPECT_EQ(mediator.incidences[1].block, 1);
  EXPECT_EQ(mediator.incidences[1].rows, std::vector<structure_index_t>({4}));
  EXPECT_EQ(mediator.incidences[2].block, 2);
  EXPECT_EQ(mediator.incidences[2].rows, std::vector<structure_index_t>({5}));

  const temporary_file_t output(std::filesystem::temp_directory_path() /
                                "cuopt_mps_structure_capped_graph_test.json");
  ASSERT_NO_THROW(
    write_structure_json(output.path().string(), model, analysis, nullptr, nullptr, nullptr));
  std::ifstream input(output.path(), std::ios::binary);
  ASSERT_TRUE(input);
  std::ostringstream buffer;
  buffer << input.rdbuf();
  const auto document = buffer.str();
  EXPECT_NE(document.find("\"pair_edges_complete\": false"), std::string::npos);
  EXPECT_NE(document.find("\"mediators\""), std::string::npos);
  EXPECT_NE(document.find("\"incidences\""), std::string::npos);
  EXPECT_NE(document.find("\"row_provenance\""), std::string::npos);
}

TEST(MpsStructureAnalysis, DecomposesDomainsInterfacesAndRepeatedComponents)
{
  const auto model = make_model({{{0, 1.0}, {1, 1.0}},
                                 {{2, 1.0}},
                                 {{3, 1.0}},
                                 {{0, 1.0}, {4, 1.0}},
                                 {{0, 1.0}, {2, 1.0}, {5, 1.0}},
                                 {{0, 1.0}, {2, 1.0}, {3, 1.0}, {6, 1.0}},
                                 {{4, 1.0}, {5, -1.0}},
                                 {{6, 1.0}}},
                                {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                {'I', 'I', 'I', 'I', 'C', 'C', 'C'},
                                {0.0, 0.0, 0.0, 0.0, -infinity, -infinity, -infinity},
                                {10.0, 10.0, 10.0, 10.0, infinity, infinity, infinity});

  const auto analysis = analyze_structure(model, "domains", {}, analysis_level_t::symmetry);

  EXPECT_EQ(analysis.row_domain_counts.at("integer_only"), size_three);
  EXPECT_EQ(analysis.row_domain_counts.at("mixed_response"), size_three);
  EXPECT_EQ(analysis.row_domain_counts.at("continuous_only"), size_two);

  const auto& integers = find_decomposition(analysis, "integer_only_rows");
  ASSERT_EQ(integers.components.size(), size_three);
  EXPECT_EQ(integers.component_size_histogram.at(1), size_two);
  EXPECT_EQ(integers.component_size_histogram.at(2), size_one);
  EXPECT_EQ(integers.interface_width_histogram.at(1), size_one);
  EXPECT_EQ(integers.interface_width_histogram.at(2), size_one);
  EXPECT_EQ(integers.interface_width_histogram.at(3), size_one);
  EXPECT_EQ(std::count_if(integers.interface_rows.begin(),
                          integers.interface_rows.end(),
                          [](const auto& row) { return row.classification == "local"; }),
            std::ptrdiff_t{1});
  EXPECT_EQ(std::count_if(integers.interface_rows.begin(),
                          integers.interface_rows.end(),
                          [](const auto& row) { return row.classification == "coupling"; }),
            std::ptrdiff_t{1});
  EXPECT_EQ(std::count_if(integers.interface_rows.begin(),
                          integers.interface_rows.end(),
                          [](const auto& row) { return row.classification == "global_interface"; }),
            std::ptrdiff_t{1});
  ASSERT_EQ(integers.repeated_component_classes.size(), size_one);
  EXPECT_EQ(integers.repeated_component_classes.begin()->second.size(), size_two);

  const auto& continuous_equalities = find_decomposition(analysis, "continuous_equality_rows");
  EXPECT_EQ(continuous_equalities.components.size(), size_two);
  EXPECT_EQ(continuous_equalities.component_size_histogram.at(1), size_one);
  EXPECT_EQ(continuous_equalities.component_size_histogram.at(2), size_one);
  EXPECT_EQ(find_decomposition(analysis, "mixed_response_rows").components.size(), size_one);
  EXPECT_EQ(find_decomposition(analysis, "continuous_projection").components.size(), size_two);

  ASSERT_EQ(analysis.repair.difference_components.size(), size_one);
  EXPECT_EQ(analysis.repair.difference_components[0].columns.size(), size_two);
  ASSERT_EQ(analysis.repair.flow_equality_components.size(), size_one);
  EXPECT_EQ(analysis.repair.flow_equality_components[0].columns.size(), size_two);
}

TEST(MpsStructureAnalysis, ComponentRefinementDistinguishesIncidenceNotJustRowTemplates)
{
  const auto model = make_model({{{0, 1.0}, {1, 1.0}},
                                 {{1, 1.0}, {2, 1.0}},
                                 {{2, 1.0}, {3, 1.0}},
                                 {{3, 1.0}, {0, 1.0}},
                                 {{4, 1.0}, {6, 1.0}},
                                 {{6, 1.0}, {5, 1.0}},
                                 {{5, 1.0}, {7, 1.0}},
                                 {{7, 1.0}, {4, 1.0}},
                                 {{8, 1.0}, {9, 1.0}},
                                 {{9, 1.0}, {10, 1.0}},
                                 {{10, 1.0}, {8, 1.0}},
                                 {{10, 1.0}, {11, 1.0}}},
                                std::vector<double>(12, 0.0),
                                std::vector<double>(12, 0.0),
                                std::vector<char>(12, 'I'),
                                std::vector<double>(12, 0.0),
                                std::vector<double>(12, 10.0));

  const auto analysis       = analyze_structure(model, "incidence", {}, analysis_level_t::basic);
  const auto& decomposition = find_decomposition(analysis, "integer_only_rows");

  ASSERT_EQ(decomposition.components.size(), size_three);
  EXPECT_EQ(decomposition.components[0].refinement_fingerprint,
            decomposition.components[1].refinement_fingerprint);
  EXPECT_NE(decomposition.components[0].refinement_fingerprint,
            decomposition.components[2].refinement_fingerprint);
  ASSERT_EQ(decomposition.repeated_component_classes.size(), size_one);
  EXPECT_EQ(decomposition.repeated_component_classes.begin()->second,
            (std::vector<structure_index_t>{0, 1}));
}

TEST(MpsStructureAnalysis, ProducesDeterministicFingerprintsAndRefinementColors)
{
  const auto model = make_model({{{0, 1.0}, {1, 1.0}}, {{0, 1.0}, {1, 1.0}}},
                                {-infinity, -infinity},
                                {1.0, 1.0},
                                {'B', 'B'},
                                {0.0, 0.0},
                                {1.0, 1.0});

  const auto first  = analyze_structure(model, "symmetry", {}, analysis_level_t::symmetry);
  const auto second = analyze_structure(model, "symmetry", {}, analysis_level_t::symmetry);

  EXPECT_EQ(first.refinement.rounds, size_three);
  ASSERT_EQ(first.refinement.exact_duplicate_rows.size(), size_one);
  EXPECT_EQ(first.refinement.exact_duplicate_rows[0].members,
            std::vector<structure_index_t>({0, 1}));
  ASSERT_EQ(first.refinement.exact_duplicate_columns.size(), size_one);
  EXPECT_EQ(first.refinement.exact_duplicate_columns[0].members,
            std::vector<structure_index_t>({0, 1}));
  ASSERT_EQ(first.refinement.normalized_row_templates.size(), size_one);
  ASSERT_EQ(first.refinement.normalized_column_templates.size(), size_one);
  EXPECT_EQ(first.refinement.row_fingerprints[0], first.refinement.row_fingerprints[1]);
  EXPECT_EQ(first.refinement.column_fingerprints[0], first.refinement.column_fingerprints[1]);
  EXPECT_EQ(first.refinement.row_colors, second.refinement.row_colors);
  EXPECT_EQ(first.refinement.column_colors, second.refinement.column_colors);
  EXPECT_EQ(first.refinement.row_fingerprints, second.refinement.row_fingerprints);
  EXPECT_EQ(first.refinement.column_fingerprints, second.refinement.column_fingerprints);
  EXPECT_EQ(first.refinement.row_color_class_histogram.at(2), size_one);
  EXPECT_EQ(first.refinement.column_color_class_histogram.at(2), size_one);
}

TEST(MpsStructureAnalysis, SummarizesNumericalScaleNormsSignsAndCancellation)
{
  constexpr double pi = 3.14159265358979323846;
  const auto model    = make_model({{{0, 1e-12}, {1, 1.0}}, {{0, 2.0}, {1, -2.0}}, {{2, pi}}},
                                   {-infinity, 0.0, -infinity},
                                   {1.25, 0.0, 1.0},
                                   {'C', 'C', 'C'},
                                   {-infinity, -infinity, -infinity},
                                   {infinity, infinity, infinity});

  const auto analysis = analyze_structure(model, "numerics", {}, analysis_level_t::basic);
  const auto& summary = analysis.numerical;

  EXPECT_EQ(summary.positive_coefficients, size_four);
  EXPECT_EQ(summary.negative_coefficients, size_one);
  EXPECT_EQ(summary.near_zero_coefficients, size_one);
  EXPECT_EQ(summary.near_cancellation_rows, std::vector<structure_index_t>({1}));
  EXPECT_DOUBLE_EQ(summary.coefficient_magnitudes.minimum, 1e-12);
  EXPECT_DOUBLE_EQ(summary.coefficient_magnitudes.maximum, pi);
  EXPECT_DOUBLE_EQ(summary.row_dynamic_ranges.maximum, 1e12);
  EXPECT_DOUBLE_EQ(summary.column_l2_norms.maximum, pi);
  EXPECT_EQ(summary.rows_not_decimal_scalable, size_one);
  EXPECT_EQ(summary.decimal_scale_exponent_histogram.at(0), size_one);
  EXPECT_EQ(summary.decimal_scale_exponent_histogram.at(2), size_one);
}

TEST(MpsStructureAnalysis, DecimalScalingRejectsLargeRelativeFractionButAllowsUlpNoise)
{
  const auto one_ulp_above_billion =
    std::nextafter(1'000'000'000.0, std::numeric_limits<double>::infinity());
  const auto model = make_model({{{0, 1'000'000'000.25}}, {{1, one_ulp_above_billion}}},
                                {-infinity, -infinity},
                                {0.0, 0.0},
                                {'C', 'C'},
                                {-infinity, -infinity},
                                {infinity, infinity});

  const auto analysis = analyze_structure(model, "decimal", {}, analysis_level_t::basic);

  EXPECT_EQ(analysis.numerical.rows_not_decimal_scalable, size_zero);
  EXPECT_EQ(analysis.numerical.decimal_scale_exponent_histogram.at(0), size_one);
  EXPECT_EQ(analysis.numerical.decimal_scale_exponent_histogram.at(2), size_one);
}

TEST(MpsStructureAnalysis, MapsPresolveStructuralDeltasThroughOriginalColumnIds)
{
  const auto original_model = make_model({{{0, 1.0}, {1, 1.0}, {2, 1.0}}, {{3, 1.0}, {4, 1.0}}},
                                         {1.0, 1.0},
                                         {1.0, 1.0},
                                         {'B', 'B', 'B', 'B', 'B'},
                                         {0.0, 0.0, 0.0, 0.0, 0.0},
                                         {1.0, 1.0, 1.0, 1.0, 1.0});
  const auto reduced_model  = make_model({{{0, 1.0}, {1, 1.0}}, {{2, 1.0}, {3, 1.0}}},
                                         {1.0, 1.0},
                                         {1.0, 1.0},
                                         {'B', 'B', 'B', 'B'},
                                         {0.0, 0.0, 0.0, 0.0},
                                         {1.0, 1.0, 1.0, 1.0});
  const auto original =
    analyze_structure(original_model, "original", {}, analysis_level_t::symmetry);
  const auto reduced =
    analyze_structure(reduced_model, "presolved", {0, 1, 3, 4}, analysis_level_t::symmetry);

  const auto delta =
    analyze_presolve_structure_delta(original, reduced, {0, 1, 3, 4}, {0, 1, -1, 2, 3}, {2});

  EXPECT_EQ(delta.exact_one.preserved, size_one);
  EXPECT_EQ(delta.exact_one.contracted, size_one);
  EXPECT_EQ(delta.exact_one.split, size_zero);
  EXPECT_EQ(delta.exact_one.destroyed, size_zero);
  EXPECT_EQ(delta.exact_one.merged, size_zero);
  EXPECT_EQ(delta.eliminated_original_columns, std::vector<structure_index_t>({2}));
  EXPECT_TRUE(delta.eliminated_original_rows.empty());
  ASSERT_EQ(delta.implied_integer_columns.size(), size_one);
  EXPECT_EQ(delta.implied_integer_columns[0].reduced_column, 2);
  EXPECT_EQ(delta.implied_integer_columns[0].original_column, 3);
  EXPECT_EQ(delta.row_domain_net_changes.at("integer_only"), 0);
  EXPECT_EQ(delta.row_family_net_changes.at("exact_one"), 0);
}

TEST(MpsStructureAnalysis, WritesEscapedBalancedMachineReadableSidecar)
{
  auto model = make_model({{{0, 1.0}, {1, 1.0}}},
                          {1.0},
                          {1.0},
                          {'B', 'B'},
                          {0.0, 0.0},
                          {1.0, 1.0},
                          "json \"smoke\"\nmodel");
  model.set_variable_names({"first\\column", "second\tcolumn"});
  const auto analysis = analyze_structure(model, "original", {}, analysis_level_t::symmetry);
  lp_overlay_summary_t source_lp;
  source_lp.attempted                    = true;
  source_lp.has_optimal_solution         = true;
  source_lp.objective_mode               = lp_objective_mode_t::source;
  source_lp.status                       = lp_overlay_status_t::optimal;
  source_lp.formulation                  = "source-objective root LP without cuts";
  source_lp.relaxation_objective         = 0.5;
  source_lp.source_objective_at_solution = 0.5;
  source_lp.integer_variables            = 2;
  source_lp.binary_variables             = 2;
  source_lp.fractional_integer_variables = 2;
  source_lp.fractional_binary_variables  = 2;
  source_lp.primal_values                = {0.25, 0.75};
  source_lp.row_duals                    = {1.5};
  source_lp.reduced_costs                = {-0.25, 0.25};
  lp_variable_record_t variable;
  variable.column                       = 0;
  variable.original_column              = 0;
  variable.name                         = "first\\column";
  variable.source_type                  = 'B';
  variable.family                       = "binary";
  variable.value                        = 0.25;
  variable.source_objective_coefficient = 1.0;
  variable.reduced_cost                 = -0.25;
  variable.fractional_part              = 0.25;
  variable.distance_to_integrality      = 0.25;
  variable.fractional                   = true;
  source_lp.variables.push_back(variable);
  lp_row_record_t row;
  row.row                    = 0;
  row.stable_id              = "original:r0";
  row.name                   = "r0";
  row.domain                 = "integer_only";
  row.families               = {"exact_one"};
  row.activity               = 1.0;
  row.lower_bound            = 1.0;
  row.upper_bound            = 1.0;
  row.lower_slack            = 0.0;
  row.upper_slack            = 0.0;
  row.nearest_bound_distance = 0.0;
  row.dual                   = 1.5;
  row.active                 = true;
  source_lp.row_records.push_back(row);
  lp_exact_one_group_record_t group;
  group.group                = 0;
  group.row                  = 0;
  group.stable_id            = "original:r0:exact_one";
  group.members              = {{0, true}, {1, true}};
  group.literal_values       = {0.25, 0.75};
  group.literal_sum          = 1.0;
  group.largest_value        = 0.75;
  group.second_largest_value = 0.25;
  group.margin               = 0.5;
  group.leading_literal      = literal_t{1, true};
  group.second_literal       = literal_t{0, true};
  group.fractional_members   = 2;
  source_lp.exact_one_groups.push_back(group);
  presolve_structure_delta_t delta;
  delta.implied_integer_columns.push_back(mapped_column_t{2, 3});
  const temporary_file_t output(std::filesystem::temp_directory_path() /
                                "cuopt_mps_structure_analysis_test.json");

  ASSERT_NO_THROW(write_structure_json(
    output.path().string(), model, analysis, &model, &analysis, &delta, &source_lp, nullptr));
  std::ifstream input(output.path(), std::ios::binary);
  ASSERT_TRUE(input);
  std::ostringstream buffer;
  buffer << input.rdbuf();
  const auto document = buffer.str();

  EXPECT_TRUE(has_balanced_json_containers(document));
  EXPECT_NE(document.find("\"schema_version\": \"1.2\""), std::string::npos);
  EXPECT_NE(document.find("json \\\"smoke\\\"\\nmodel"), std::string::npos);
  EXPECT_NE(document.find("first\\\\column"), std::string::npos);
  EXPECT_NE(document.find("second\\tcolumn"), std::string::npos);
  EXPECT_NE(document.find("\"stable_id\": \"original:r0\""), std::string::npos);
  EXPECT_NE(document.find("\"exact_one\""), std::string::npos);
  EXPECT_NE(document.find("\"source_objective_lp_overlay_included\": true"), std::string::npos);
  EXPECT_NE(document.find("\"objective_erased_lp_overlay_included\": false"), std::string::npos);
  EXPECT_NE(document.find("\"primal_values\""), std::string::npos);
  EXPECT_NE(document.find("\"row_records\""), std::string::npos);
  EXPECT_NE(document.find("\"exact_one_groups\""), std::string::npos);
  EXPECT_NE(document.find("\"reduced_column\": 2"), std::string::npos);
  EXPECT_NE(document.find("\"original_column\": 3"), std::string::npos);
}

}  // namespace
}  // namespace cuopt::mathematical_optimization::examples
