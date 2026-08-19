/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace cuopt::mathematical_optimization::examples {

using structure_index_t = int;
using structure_value_t = double;
using structure_model_t = io::mps_data_model_t<structure_index_t, structure_value_t>;

enum class analysis_level_t { basic, symmetry };

inline constexpr std::size_t default_block_pair_provenance_limit = 100'000;

struct literal_t {
  structure_index_t column{};
  bool value{};
};

struct row_record_t {
  structure_index_t row{};
  std::string stable_id;
  std::string name;
  std::string domain;
  std::vector<std::string> families;
  std::vector<structure_index_t> columns;
  std::vector<double> coefficients;
  std::vector<literal_t> binary_literals;
};

struct row_group_t {
  structure_index_t id{};
  structure_index_t row{};
  std::string stable_id;
  std::string family;
  std::vector<literal_t> members;
  std::vector<double> member_coefficients;
  double lower_capacity{};
  double upper_capacity{};
  bool has_lower_capacity{};
  bool has_upper_capacity{};
};

struct implication_t {
  structure_index_t row{};
  literal_t antecedent;
  literal_t consequent;
};

struct variable_bound_t {
  structure_index_t row{};
  structure_index_t binary_column{};
  structure_index_t target_column{};
  std::string bound_type;
  double bound_when_zero{};
  double bound_when_one{};
  bool activation{};
};

struct affine_definition_t {
  structure_index_t row{};
  std::vector<structure_index_t> pivot_columns;
  bool singleton{};
  bool two_variable{};
};

struct block_edge_t {
  structure_index_t first{};
  structure_index_t second{};
  std::vector<structure_index_t> direct_rows;
  std::vector<structure_index_t> integer_mediators;
  std::vector<structure_index_t> continuous_mediators;
};

struct block_mediator_incidence_t {
  structure_index_t block{};
  std::vector<structure_index_t> rows;
};

struct block_mediator_t {
  structure_index_t column{};
  std::string kind;
  std::vector<block_mediator_incidence_t> incidences;
};

struct graph_projection_summary_t {
  std::string name;
  bool complete{true};
  std::size_t edges{};
  std::size_t total_weight{};
  std::size_t components{};
  std::size_t isolated_vertices{};
  std::size_t largest_component{};
  std::vector<std::size_t> component_sizes;
  std::vector<std::pair<structure_index_t, std::size_t>> highest_weighted_degrees;
};

struct component_t {
  structure_index_t id{};
  std::vector<structure_index_t> rows;
  std::vector<structure_index_t> columns;
  std::string refinement_fingerprint;
};

struct interface_row_t {
  structure_index_t row{};
  std::string classification;
  std::vector<structure_index_t> component_ids;
};

struct decomposition_t {
  std::string name;
  std::vector<component_t> components;
  std::vector<interface_row_t> interface_rows;
  std::map<std::size_t, std::size_t> component_size_histogram;
  std::map<std::size_t, std::size_t> interface_width_histogram;
  std::map<std::string, std::vector<structure_index_t>> repeated_component_classes;
};

struct duplicate_class_t {
  std::string fingerprint;
  std::vector<structure_index_t> members;
};

struct refinement_summary_t {
  std::size_t rounds{};
  std::vector<std::string> row_fingerprints;
  std::vector<std::string> column_fingerprints;
  std::vector<structure_index_t> row_colors;
  std::vector<structure_index_t> column_colors;
  std::vector<duplicate_class_t> exact_duplicate_rows;
  std::vector<duplicate_class_t> exact_duplicate_columns;
  std::vector<duplicate_class_t> normalized_row_templates;
  std::vector<duplicate_class_t> normalized_column_templates;
  std::map<std::size_t, std::size_t> row_color_class_histogram;
  std::map<std::size_t, std::size_t> column_color_class_histogram;
};

struct quantile_summary_t {
  double minimum{};
  double quartile_1{};
  double median{};
  double quartile_3{};
  double percentile_90{};
  double percentile_99{};
  double maximum{};
};

struct numerical_summary_t {
  quantile_summary_t coefficient_magnitudes;
  quantile_summary_t row_dynamic_ranges;
  quantile_summary_t column_l2_norms;
  std::size_t positive_coefficients{};
  std::size_t negative_coefficients{};
  std::size_t near_zero_coefficients{};
  std::vector<structure_index_t> near_cancellation_rows;
  std::map<int, std::size_t> decimal_scale_exponent_histogram;
  std::size_t rows_not_decimal_scalable{};
};

struct repair_summary_t {
  std::vector<structure_index_t> monotone_increase_columns;
  std::vector<structure_index_t> monotone_decrease_columns;
  std::vector<structure_index_t> free_equality_pivots;
  std::vector<structure_index_t> two_variable_affine_rows;
  std::vector<component_t> difference_components;
  std::vector<component_t> flow_equality_components;
};

struct model_analysis_t {
  std::string scope;
  analysis_level_t level{analysis_level_t::basic};
  std::vector<structure_index_t> original_column_ids;
  std::vector<structure_index_t> original_row_ids;
  std::map<std::string, std::size_t> row_domain_counts;
  std::map<std::string, std::size_t> row_family_counts;
  std::vector<row_record_t> rows;
  std::vector<row_group_t> exact_one_groups;
  std::vector<row_group_t> set_packing_groups;
  std::vector<row_group_t> set_covering_groups;
  std::vector<row_group_t> gub_groups;
  std::vector<row_group_t> knapsack_rows;
  std::vector<implication_t> implications;
  std::vector<variable_bound_t> variable_bounds;
  std::vector<affine_definition_t> affine_definitions;
  std::size_t binary_variables{};
  std::size_t exact_one_covered_binary_variables{};
  bool exact_one_groups_overlap{};
  std::vector<std::size_t> exact_one_group_size_histogram;
  std::vector<std::size_t> exact_one_overlap_component_sizes;
  std::vector<block_edge_t> exact_one_block_edges;
  std::vector<block_mediator_t> exact_one_block_mediators;
  std::size_t exact_one_block_pair_provenance_limit{};
  bool exact_one_block_edges_complete{true};
  std::map<std::string, std::size_t> exact_one_block_candidate_pair_provenance;
  std::map<std::string, std::size_t> exact_one_block_materialized_pair_provenance;
  std::vector<graph_projection_summary_t> exact_one_block_projections;
  std::vector<decomposition_t> decompositions;
  refinement_summary_t refinement;
  numerical_summary_t numerical;
  repair_summary_t repair;
};

struct exact_one_delta_t {
  std::size_t preserved{};
  std::size_t contracted{};
  std::size_t split{};
  std::size_t destroyed{};
  std::size_t merged{};
};

struct mapped_column_t {
  structure_index_t reduced_column{};
  structure_index_t original_column{};
};

struct presolve_structure_delta_t {
  std::map<std::string, std::int64_t> row_domain_net_changes;
  std::map<std::string, std::int64_t> row_family_net_changes;
  exact_one_delta_t exact_one;
  std::vector<structure_index_t> eliminated_original_columns;
  std::vector<structure_index_t> eliminated_original_rows;
  std::vector<mapped_column_t> implied_integer_columns;
  std::size_t original_refinement_tie_classes{};
  std::size_t reduced_refinement_tie_classes{};
};

struct lp_overlay_summary_t;

model_analysis_t analyze_structure(
  const structure_model_t& model,
  std::string scope,
  const std::vector<structure_index_t>& original_column_ids,
  analysis_level_t level,
  const std::vector<structure_index_t>& original_row_ids = {},
  std::size_t block_pair_provenance_limit                = default_block_pair_provenance_limit);

presolve_structure_delta_t analyze_presolve_structure_delta(
  const model_analysis_t& original,
  const model_analysis_t& reduced,
  const std::vector<structure_index_t>& reduced_to_original,
  const std::vector<structure_index_t>& original_to_reduced,
  const std::vector<structure_index_t>& implied_integer_columns);

void print_special_structure_summary(const structure_model_t& model,
                                     const model_analysis_t& analysis);

void print_presolve_structure_delta(const presolve_structure_delta_t& delta);

void write_structure_json(const std::string& path,
                          const structure_model_t& original_model,
                          const model_analysis_t& original,
                          const structure_model_t* reduced_model,
                          const model_analysis_t* reduced,
                          const presolve_structure_delta_t* delta,
                          const lp_overlay_summary_t* source_lp           = nullptr,
                          const lp_overlay_summary_t* objective_erased_lp = nullptr);

std::string_view analysis_level_name(analysis_level_t level);

}  // namespace cuopt::mathematical_optimization::examples
