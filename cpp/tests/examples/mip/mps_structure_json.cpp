/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "mps_structure_analysis.hpp"

#include "mps_lp_overlay.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::examples {
namespace {

class json_writer_t {
 public:
  explicit json_writer_t(std::ostream& output) : output_(output) {}

  void begin_object()
  {
    before_value();
    output_ << '{';
    stack_.push_back(context_t{context_kind_t::object});
  }

  void end_object()
  {
    require_context(context_kind_t::object);
    if (stack_.back().expecting_value) { throw std::logic_error("JSON object key has no value"); }
    const auto empty = stack_.back().first;
    stack_.pop_back();
    if (!empty) {
      output_ << '\n';
      write_indent(stack_.size());
    }
    output_ << '}';
  }

  void begin_array()
  {
    before_value();
    output_ << '[';
    stack_.push_back(context_t{context_kind_t::array});
  }

  void end_array()
  {
    require_context(context_kind_t::array);
    const auto empty = stack_.back().first;
    stack_.pop_back();
    if (!empty) {
      output_ << '\n';
      write_indent(stack_.size());
    }
    output_ << ']';
  }

  void key(std::string_view name)
  {
    require_context(context_kind_t::object);
    auto& context = stack_.back();
    if (context.expecting_value) { throw std::logic_error("JSON object key has no value"); }
    if (!context.first) { output_ << ','; }
    output_ << '\n';
    write_indent(stack_.size());
    write_escaped_string(name);
    output_ << ": ";
    context.first           = false;
    context.expecting_value = true;
  }

  void string(std::string_view value)
  {
    before_value();
    write_escaped_string(value);
  }

  void boolean(bool value)
  {
    before_value();
    output_ << (value ? "true" : "false");
  }

  void null()
  {
    before_value();
    output_ << "null";
  }

  template <typename integer_t,
            std::enable_if_t<std::is_integral_v<integer_t> &&
                               !std::is_same_v<std::remove_cv_t<integer_t>, bool>,
                             int> = 0>
  void integer(integer_t value)
  {
    before_value();
    output_ << value;
  }

  void number(double value)
  {
    before_value();
    if (!std::isfinite(value)) {
      output_ << "null";
      return;
    }
    output_ << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
  }

  bool complete() const { return stack_.empty() && wrote_root_; }

 private:
  enum class context_kind_t { object, array };

  struct context_t {
    explicit context_t(context_kind_t kind_) : kind(kind_) {}
    context_kind_t kind;
    bool first{true};
    bool expecting_value{false};
  };

  void before_value()
  {
    if (stack_.empty()) {
      if (wrote_root_) { throw std::logic_error("JSON document has multiple root values"); }
      wrote_root_ = true;
      return;
    }

    auto& context = stack_.back();
    if (context.kind == context_kind_t::object) {
      if (!context.expecting_value) { throw std::logic_error("JSON object value has no key"); }
      context.expecting_value = false;
      return;
    }

    if (!context.first) { output_ << ','; }
    output_ << '\n';
    write_indent(stack_.size());
    context.first = false;
  }

  void require_context(context_kind_t kind) const
  {
    if (stack_.empty() || stack_.back().kind != kind) {
      throw std::logic_error("mismatched JSON container");
    }
  }

  void write_indent(std::size_t depth)
  {
    for (std::size_t i = 0; i < depth * 2; ++i) {
      output_.put(' ');
    }
  }

  void write_escaped_string(std::string_view value)
  {
    static constexpr char hexadecimal[] = "0123456789abcdef";
    output_.put('"');
    for (const auto character : value) {
      const auto byte = static_cast<unsigned char>(character);
      switch (byte) {
        case '"': output_ << "\\\""; break;
        case '\\': output_ << "\\\\"; break;
        case '\b': output_ << "\\b"; break;
        case '\f': output_ << "\\f"; break;
        case '\n': output_ << "\\n"; break;
        case '\r': output_ << "\\r"; break;
        case '\t': output_ << "\\t"; break;
        default:
          if (byte < 0x20) {
            output_ << "\\u00" << hexadecimal[(byte >> 4) & 0x0f] << hexadecimal[byte & 0x0f];
          } else {
            output_.put(static_cast<char>(byte));
          }
      }
    }
    output_.put('"');
  }

  std::ostream& output_;
  std::vector<context_t> stack_;
  bool wrote_root_{false};
};

struct stage_context_t {
  const structure_model_t& model;
  const model_analysis_t& analysis;
};

std::string_view variable_name(const stage_context_t& stage, structure_index_t column)
{
  const auto& names = stage.model.get_variable_names();
  if (column < 0 || static_cast<std::size_t>(column) >= names.size()) { return {}; }
  return names[static_cast<std::size_t>(column)];
}

std::string_view row_name(const stage_context_t& stage, structure_index_t row)
{
  const auto& names = stage.model.get_row_names();
  if (row < 0 || static_cast<std::size_t>(row) >= names.size()) { return {}; }
  return names[static_cast<std::size_t>(row)];
}

const row_record_t* find_row_record(const stage_context_t& stage, structure_index_t row)
{
  if (row >= 0 && static_cast<std::size_t>(row) < stage.analysis.rows.size()) {
    const auto& candidate = stage.analysis.rows[static_cast<std::size_t>(row)];
    if (candidate.row == row) { return &candidate; }
  }
  for (const auto& record : stage.analysis.rows) {
    if (record.row == row) { return &record; }
  }
  return nullptr;
}

void write_original_column(json_writer_t& writer,
                           const stage_context_t& stage,
                           structure_index_t column)
{
  if (column < 0 || static_cast<std::size_t>(column) >= stage.analysis.original_column_ids.size() ||
      stage.analysis.original_column_ids[static_cast<std::size_t>(column)] < 0) {
    writer.null();
    return;
  }
  writer.integer(stage.analysis.original_column_ids[static_cast<std::size_t>(column)]);
}

void write_original_row(json_writer_t& writer, const stage_context_t& stage, structure_index_t row)
{
  if (row < 0 || static_cast<std::size_t>(row) >= stage.analysis.original_row_ids.size() ||
      stage.analysis.original_row_ids[static_cast<std::size_t>(row)] < 0) {
    writer.null();
    return;
  }
  writer.integer(stage.analysis.original_row_ids[static_cast<std::size_t>(row)]);
}

template <typename value_t>
void write_integer_array(json_writer_t& writer, const std::vector<value_t>& values)
{
  writer.begin_array();
  for (const auto value : values) {
    writer.integer(value);
  }
  writer.end_array();
}

void write_double_array(json_writer_t& writer, const std::vector<double>& values)
{
  writer.begin_array();
  for (const auto value : values) {
    writer.number(value);
  }
  writer.end_array();
}

void write_string_array(json_writer_t& writer, const std::vector<std::string>& values)
{
  writer.begin_array();
  for (const auto& value : values) {
    writer.string(value);
  }
  writer.end_array();
}

void write_original_columns(json_writer_t& writer,
                            const stage_context_t& stage,
                            const std::vector<structure_index_t>& columns)
{
  writer.begin_array();
  for (const auto column : columns) {
    write_original_column(writer, stage, column);
  }
  writer.end_array();
}

void write_count_map(json_writer_t& writer, const std::map<std::string, std::size_t>& counts)
{
  writer.begin_object();
  for (const auto& [name, count] : counts) {
    writer.key(name);
    writer.integer(count);
  }
  writer.end_object();
}

template <typename key_t>
void write_numeric_histogram(json_writer_t& writer, const std::map<key_t, std::size_t>& histogram)
{
  writer.begin_array();
  for (const auto& [value, count] : histogram) {
    writer.begin_object();
    writer.key("value");
    writer.integer(value);
    writer.key("count");
    writer.integer(count);
    writer.end_object();
  }
  writer.end_array();
}

void write_variable_ref(json_writer_t& writer,
                        const stage_context_t& stage,
                        structure_index_t column)
{
  writer.begin_object();
  writer.key("column");
  writer.integer(column);
  writer.key("original_column");
  write_original_column(writer, stage, column);
  writer.key("name");
  writer.string(variable_name(stage, column));
  writer.end_object();
}

void write_row_ref(json_writer_t& writer, const stage_context_t& stage, structure_index_t row)
{
  writer.begin_object();
  writer.key("row");
  writer.integer(row);
  writer.key("original_row");
  write_original_row(writer, stage, row);
  writer.key("stable_id");
  const auto* record = find_row_record(stage, row);
  writer.string(record == nullptr ? std::string_view{} : std::string_view{record->stable_id});
  writer.key("name");
  writer.string(row_name(stage, row));
  writer.end_object();
}

void write_literal(json_writer_t& writer, const stage_context_t& stage, const literal_t& literal)
{
  writer.begin_object();
  writer.key("column");
  writer.integer(literal.column);
  writer.key("original_column");
  write_original_column(writer, stage, literal.column);
  writer.key("name");
  writer.string(variable_name(stage, literal.column));
  writer.key("value");
  writer.boolean(literal.value);
  writer.end_object();
}

void write_literals(json_writer_t& writer,
                    const stage_context_t& stage,
                    const std::vector<literal_t>& literals)
{
  writer.begin_array();
  for (const auto& literal : literals) {
    write_literal(writer, stage, literal);
  }
  writer.end_array();
}

void write_detector_metadata(json_writer_t& writer,
                             const model_analysis_t& analysis,
                             bool root_lp_overlay_included)
{
  writer.begin_object();
  writer.key("name");
  writer.string("cuopt_static_mps_structure");
  writer.key("analysis_level");
  writer.string(analysis_level_name(analysis.level));
  writer.key("scope");
  writer.string("static row- and matrix-derived analysis of recognized structures");
  writer.key("completeness");
  writer.begin_object();
  writer.key("row_records");
  writer.boolean(true);
  writer.key("recognized_group_members");
  writer.boolean(true);
  writer.key("reported_block_edge_provenance");
  writer.boolean(analysis.exact_one_block_edges_complete);
  writer.key("reported_block_mediator_paths");
  writer.boolean(true);
  writer.key("reported_decomposition_members");
  writer.boolean(true);
  writer.key("surviving_row_identity_map");
  writer.boolean(true);
  writer.key("presolve_operation_trace");
  writer.boolean(false);
  writer.key("anonymous_refinement");
  writer.boolean(analysis.level == analysis_level_t::symmetry);
  writer.key("root_lp_overlay");
  writer.boolean(root_lp_overlay_included);
  writer.key("probing_implications");
  writer.boolean(false);
  writer.key("search_learned_conflicts");
  writer.boolean(false);
  writer.end_object();
  writer.end_object();
}

void write_model(json_writer_t& writer, const stage_context_t& stage)
{
  const auto& model = stage.model;
  writer.begin_object();
  writer.key("name");
  writer.string(model.get_problem_name());
  writer.key("objective_name");
  writer.string(model.get_objective_name());
  writer.key("objective_sense");
  writer.string(model.get_sense() ? "maximize" : "minimize");
  writer.key("dimensions");
  writer.begin_object();
  writer.key("rows");
  writer.integer(model.get_n_constraints());
  writer.key("columns");
  writer.integer(model.get_n_variables());
  writer.key("nonzeros");
  writer.integer(model.get_nnz());
  writer.end_object();
  writer.key("objective_scaling_factor");
  writer.number(model.get_objective_scaling_factor());
  writer.key("objective_offset");
  writer.number(model.get_objective_offset());
  writer.key("row_names");
  write_string_array(writer, model.get_row_names());
  writer.key("column_names");
  write_string_array(writer, model.get_variable_names());

  const auto& types      = model.get_variable_types();
  const auto& lower      = model.get_variable_lower_bounds();
  const auto& upper      = model.get_variable_upper_bounds();
  const auto& objective  = model.get_objective_coefficients();
  const auto n_variables = static_cast<std::size_t>(model.get_n_variables());
  writer.key("variables");
  writer.begin_array();
  for (std::size_t column = 0; column < n_variables; ++column) {
    writer.begin_object();
    writer.key("column");
    writer.integer(column);
    writer.key("original_column");
    write_original_column(writer, stage, static_cast<structure_index_t>(column));
    writer.key("name");
    writer.string(column < model.get_variable_names().size()
                    ? std::string_view{model.get_variable_names()[column]}
                    : std::string_view{});
    writer.key("type");
    if (column < types.size()) {
      const char type[] = {types[column], '\0'};
      writer.string(type);
    } else {
      writer.string("");
    }
    writer.key("lower_bound");
    column < lower.size() ? writer.number(lower[column]) : writer.null();
    writer.key("upper_bound");
    column < upper.size() ? writer.number(upper[column]) : writer.null();
    writer.key("objective_coefficient");
    column < objective.size() ? writer.number(objective[column]) : writer.null();
    writer.end_object();
  }
  writer.end_array();
  writer.end_object();
}

void write_rows(json_writer_t& writer, const stage_context_t& stage)
{
  const auto& lower = stage.model.get_constraint_lower_bounds();
  const auto& upper = stage.model.get_constraint_upper_bounds();
  writer.begin_array();
  for (const auto& row : stage.analysis.rows) {
    writer.begin_object();
    writer.key("row");
    writer.integer(row.row);
    writer.key("original_row");
    write_original_row(writer, stage, row.row);
    writer.key("stable_id");
    writer.string(row.stable_id);
    writer.key("name");
    writer.string(row.name);
    writer.key("domain");
    writer.string(row.domain);
    writer.key("families");
    write_string_array(writer, row.families);
    writer.key("lower_bound");
    row.row >= 0 && static_cast<std::size_t>(row.row) < lower.size()
      ? writer.number(lower[static_cast<std::size_t>(row.row)])
      : writer.null();
    writer.key("upper_bound");
    row.row >= 0 && static_cast<std::size_t>(row.row) < upper.size()
      ? writer.number(upper[static_cast<std::size_t>(row.row)])
      : writer.null();
    writer.key("columns");
    write_integer_array(writer, row.columns);
    writer.key("original_columns");
    write_original_columns(writer, stage, row.columns);
    writer.key("coefficients");
    write_double_array(writer, row.coefficients);
    writer.key("binary_literals");
    write_literals(writer, stage, row.binary_literals);
    writer.end_object();
  }
  writer.end_array();
}

void write_groups(json_writer_t& writer,
                  const stage_context_t& stage,
                  const std::vector<row_group_t>& groups)
{
  writer.begin_array();
  for (const auto& group : groups) {
    writer.begin_object();
    writer.key("id");
    writer.integer(group.id);
    writer.key("stable_id");
    writer.string(group.stable_id);
    writer.key("family");
    writer.string(group.family);
    writer.key("row");
    writer.integer(group.row);
    writer.key("row_provenance");
    write_row_ref(writer, stage, group.row);
    writer.key("members");
    write_literals(writer, stage, group.members);
    writer.key("member_coefficients");
    write_double_array(writer, group.member_coefficients);
    writer.key("lower_capacity");
    group.has_lower_capacity ? writer.number(group.lower_capacity) : writer.null();
    writer.key("upper_capacity");
    group.has_upper_capacity ? writer.number(group.upper_capacity) : writer.null();
    writer.end_object();
  }
  writer.end_array();
}

void write_implications(json_writer_t& writer, const stage_context_t& stage)
{
  writer.begin_array();
  for (const auto& implication : stage.analysis.implications) {
    writer.begin_object();
    writer.key("row");
    writer.integer(implication.row);
    writer.key("row_provenance");
    write_row_ref(writer, stage, implication.row);
    writer.key("antecedent");
    write_literal(writer, stage, implication.antecedent);
    writer.key("consequent");
    write_literal(writer, stage, implication.consequent);
    writer.end_object();
  }
  writer.end_array();
}

void write_variable_bounds(json_writer_t& writer, const stage_context_t& stage)
{
  writer.begin_array();
  for (const auto& bound : stage.analysis.variable_bounds) {
    writer.begin_object();
    writer.key("row");
    writer.integer(bound.row);
    writer.key("row_provenance");
    write_row_ref(writer, stage, bound.row);
    writer.key("binary_column");
    write_variable_ref(writer, stage, bound.binary_column);
    writer.key("target_column");
    write_variable_ref(writer, stage, bound.target_column);
    writer.key("bound_type");
    writer.string(bound.bound_type);
    writer.key("bound_when_zero");
    writer.number(bound.bound_when_zero);
    writer.key("bound_when_one");
    writer.number(bound.bound_when_one);
    writer.key("activation");
    writer.boolean(bound.activation);
    writer.end_object();
  }
  writer.end_array();
}

void write_affine_definitions(json_writer_t& writer, const stage_context_t& stage)
{
  writer.begin_array();
  for (const auto& definition : stage.analysis.affine_definitions) {
    writer.begin_object();
    writer.key("row");
    writer.integer(definition.row);
    writer.key("row_provenance");
    write_row_ref(writer, stage, definition.row);
    writer.key("pivot_columns");
    write_integer_array(writer, definition.pivot_columns);
    writer.key("original_pivot_columns");
    write_original_columns(writer, stage, definition.pivot_columns);
    writer.key("singleton");
    writer.boolean(definition.singleton);
    writer.key("two_variable");
    writer.boolean(definition.two_variable);
    writer.end_object();
  }
  writer.end_array();
}

void write_block_edges(json_writer_t& writer, const stage_context_t& stage)
{
  writer.begin_array();
  for (const auto& edge : stage.analysis.exact_one_block_edges) {
    writer.begin_object();
    writer.key("first");
    writer.integer(edge.first);
    writer.key("second");
    writer.integer(edge.second);
    writer.key("direct_rows");
    write_integer_array(writer, edge.direct_rows);
    writer.key("direct_row_provenance");
    writer.begin_array();
    for (const auto row : edge.direct_rows) {
      write_row_ref(writer, stage, row);
    }
    writer.end_array();
    writer.key("integer_mediators");
    write_integer_array(writer, edge.integer_mediators);
    writer.key("integer_mediator_provenance");
    writer.begin_array();
    for (const auto column : edge.integer_mediators) {
      write_variable_ref(writer, stage, column);
    }
    writer.end_array();
    writer.key("continuous_mediators");
    write_integer_array(writer, edge.continuous_mediators);
    writer.key("continuous_mediator_provenance");
    writer.begin_array();
    for (const auto column : edge.continuous_mediators) {
      write_variable_ref(writer, stage, column);
    }
    writer.end_array();
    writer.end_object();
  }
  writer.end_array();
}

void write_block_mediators(json_writer_t& writer, const stage_context_t& stage)
{
  writer.begin_array();
  for (const auto& mediator : stage.analysis.exact_one_block_mediators) {
    writer.begin_object();
    writer.key("column");
    writer.integer(mediator.column);
    writer.key("mediator_provenance");
    write_variable_ref(writer, stage, mediator.column);
    writer.key("kind");
    writer.string(mediator.kind);
    writer.key("incidences");
    writer.begin_array();
    for (const auto& incidence : mediator.incidences) {
      writer.begin_object();
      writer.key("block");
      writer.integer(incidence.block);
      writer.key("rows");
      write_integer_array(writer, incidence.rows);
      writer.key("row_provenance");
      writer.begin_array();
      for (const auto row : incidence.rows) {
        write_row_ref(writer, stage, row);
      }
      writer.end_array();
      writer.end_object();
    }
    writer.end_array();
    writer.end_object();
  }
  writer.end_array();
}

void write_graph_projections(json_writer_t& writer, const stage_context_t& stage)
{
  writer.begin_array();
  for (const auto& projection : stage.analysis.exact_one_block_projections) {
    writer.begin_object();
    writer.key("name");
    writer.string(projection.name);
    writer.key("complete");
    writer.boolean(projection.complete);
    writer.key("edges");
    writer.integer(projection.edges);
    writer.key("total_weight");
    writer.integer(projection.total_weight);
    writer.key("components");
    writer.integer(projection.components);
    writer.key("isolated_vertices");
    writer.integer(projection.isolated_vertices);
    writer.key("largest_component");
    writer.integer(projection.largest_component);
    writer.key("component_sizes");
    write_integer_array(writer, projection.component_sizes);
    writer.key("highest_weighted_degrees");
    writer.begin_array();
    for (const auto& [block, degree] : projection.highest_weighted_degrees) {
      writer.begin_object();
      writer.key("block");
      writer.integer(block);
      writer.key("weighted_degree");
      writer.integer(degree);
      writer.end_object();
    }
    writer.end_array();
    writer.end_object();
  }
  writer.end_array();
}

void write_component(json_writer_t& writer,
                     const stage_context_t& stage,
                     const component_t& component)
{
  writer.begin_object();
  writer.key("id");
  writer.integer(component.id);
  writer.key("rows");
  write_integer_array(writer, component.rows);
  writer.key("row_provenance");
  writer.begin_array();
  for (const auto row : component.rows) {
    write_row_ref(writer, stage, row);
  }
  writer.end_array();
  writer.key("columns");
  write_integer_array(writer, component.columns);
  writer.key("original_columns");
  write_original_columns(writer, stage, component.columns);
  writer.key("column_names");
  writer.begin_array();
  for (const auto column : component.columns) {
    writer.string(variable_name(stage, column));
  }
  writer.end_array();
  writer.key("refinement_fingerprint");
  writer.string(component.refinement_fingerprint);
  writer.end_object();
}

void write_components(json_writer_t& writer,
                      const stage_context_t& stage,
                      const std::vector<component_t>& components)
{
  writer.begin_array();
  for (const auto& component : components) {
    write_component(writer, stage, component);
  }
  writer.end_array();
}

void write_decompositions(json_writer_t& writer, const stage_context_t& stage)
{
  writer.begin_array();
  for (const auto& decomposition : stage.analysis.decompositions) {
    writer.begin_object();
    writer.key("name");
    writer.string(decomposition.name);
    writer.key("components");
    write_components(writer, stage, decomposition.components);
    writer.key("interface_rows");
    writer.begin_array();
    for (const auto& interface_row : decomposition.interface_rows) {
      writer.begin_object();
      writer.key("row");
      writer.integer(interface_row.row);
      writer.key("row_provenance");
      write_row_ref(writer, stage, interface_row.row);
      writer.key("classification");
      writer.string(interface_row.classification);
      writer.key("component_ids");
      write_integer_array(writer, interface_row.component_ids);
      writer.end_object();
    }
    writer.end_array();
    writer.key("component_size_histogram");
    write_numeric_histogram(writer, decomposition.component_size_histogram);
    writer.key("interface_width_histogram");
    write_numeric_histogram(writer, decomposition.interface_width_histogram);
    writer.key("repeated_component_classes");
    writer.begin_array();
    for (const auto& [fingerprint, component_ids] : decomposition.repeated_component_classes) {
      writer.begin_object();
      writer.key("fingerprint");
      writer.string(fingerprint);
      writer.key("component_ids");
      write_integer_array(writer, component_ids);
      writer.end_object();
    }
    writer.end_array();
    writer.end_object();
  }
  writer.end_array();
}

void write_duplicate_classes(json_writer_t& writer, const std::vector<duplicate_class_t>& classes)
{
  writer.begin_array();
  for (const auto& duplicate_class : classes) {
    writer.begin_object();
    writer.key("fingerprint");
    writer.string(duplicate_class.fingerprint);
    writer.key("members");
    write_integer_array(writer, duplicate_class.members);
    writer.end_object();
  }
  writer.end_array();
}

void write_refinement(json_writer_t& writer, const refinement_summary_t& refinement)
{
  writer.begin_object();
  writer.key("rounds");
  writer.integer(refinement.rounds);
  writer.key("row_fingerprints");
  write_string_array(writer, refinement.row_fingerprints);
  writer.key("column_fingerprints");
  write_string_array(writer, refinement.column_fingerprints);
  writer.key("row_colors");
  write_integer_array(writer, refinement.row_colors);
  writer.key("column_colors");
  write_integer_array(writer, refinement.column_colors);
  writer.key("exact_duplicate_rows");
  write_duplicate_classes(writer, refinement.exact_duplicate_rows);
  writer.key("exact_duplicate_columns");
  write_duplicate_classes(writer, refinement.exact_duplicate_columns);
  writer.key("normalized_row_templates");
  write_duplicate_classes(writer, refinement.normalized_row_templates);
  writer.key("normalized_column_templates");
  write_duplicate_classes(writer, refinement.normalized_column_templates);
  writer.key("row_color_class_histogram");
  write_numeric_histogram(writer, refinement.row_color_class_histogram);
  writer.key("column_color_class_histogram");
  write_numeric_histogram(writer, refinement.column_color_class_histogram);
  writer.end_object();
}

void write_quantiles(json_writer_t& writer, const quantile_summary_t& quantiles)
{
  writer.begin_object();
  writer.key("minimum");
  writer.number(quantiles.minimum);
  writer.key("quartile_1");
  writer.number(quantiles.quartile_1);
  writer.key("median");
  writer.number(quantiles.median);
  writer.key("quartile_3");
  writer.number(quantiles.quartile_3);
  writer.key("percentile_90");
  writer.number(quantiles.percentile_90);
  writer.key("percentile_99");
  writer.number(quantiles.percentile_99);
  writer.key("maximum");
  writer.number(quantiles.maximum);
  writer.end_object();
}

void write_optional_number(json_writer_t& writer, const std::optional<double>& value)
{
  value.has_value() ? writer.number(*value) : writer.null();
}

void write_lp_histogram(json_writer_t& writer, const lp_histogram_t& histogram)
{
  writer.begin_object();
  writer.key("quantity");
  writer.string(histogram.quantity);
  writer.key("bins");
  writer.begin_array();
  for (const auto& bin : histogram.bins) {
    writer.begin_object();
    writer.key("lower_bound");
    writer.number(bin.lower_bound);
    writer.key("upper_bound");
    writer.number(bin.upper_bound);
    writer.key("lower_bound_inclusive");
    writer.boolean(bin.lower_bound_inclusive);
    writer.key("upper_bound_inclusive");
    writer.boolean(bin.upper_bound_inclusive);
    writer.key("count");
    writer.integer(bin.count);
    writer.end_object();
  }
  writer.end_array();
  writer.end_object();
}

void write_lp_distribution(json_writer_t& writer, const lp_scalar_distribution_t& distribution)
{
  writer.begin_object();
  writer.key("count");
  writer.integer(distribution.count);
  writer.key("quantiles");
  write_quantiles(writer, distribution.quantiles);
  writer.end_object();
}

void write_lp_row_summary(json_writer_t& writer, const lp_row_summary_t& summary)
{
  writer.begin_object();
  writer.key("rows");
  writer.integer(summary.rows);
  writer.key("active_rows");
  writer.integer(summary.active_rows);
  writer.key("violated_rows");
  writer.integer(summary.violated_rows);
  writer.key("nonzero_duals");
  writer.integer(summary.nonzero_duals);
  writer.key("nearest_bound_distance");
  write_lp_distribution(writer, summary.nearest_bound_distance);
  writer.key("signed_dual");
  write_lp_distribution(writer, summary.signed_dual);
  writer.key("absolute_dual");
  write_lp_distribution(writer, summary.absolute_dual);
  writer.end_object();
}

void write_lp_exact_one_summary(json_writer_t& writer, const lp_exact_one_summary_t& summary)
{
  writer.begin_object();
  writer.key("groups");
  writer.integer(summary.groups);
  writer.key("integral_groups");
  writer.integer(summary.integral_groups);
  writer.key("nearly_integral_groups");
  writer.integer(summary.nearly_integral_groups);
  writer.key("fractional_groups");
  writer.integer(summary.fractional_groups);
  writer.key("entropy");
  write_lp_distribution(writer, summary.entropy);
  writer.key("effective_support");
  write_lp_distribution(writer, summary.effective_support);
  writer.key("largest_value");
  write_lp_distribution(writer, summary.largest_value);
  writer.key("second_largest_value");
  write_lp_distribution(writer, summary.second_largest_value);
  writer.key("margin");
  write_lp_distribution(writer, summary.margin);
  writer.end_object();
}

std::string_view lp_objective_mode_json_name(lp_objective_mode_t mode)
{
  switch (mode) {
    case lp_objective_mode_t::source: return "source";
    case lp_objective_mode_t::erased: return "erased";
  }
  return "unknown";
}

std::string_view lp_overlay_status_json_name(lp_overlay_status_t status)
{
  switch (status) {
    case lp_overlay_status_t::not_run: return "not_run";
    case lp_overlay_status_t::optimal: return "optimal";
    case lp_overlay_status_t::infeasible: return "infeasible";
    case lp_overlay_status_t::unbounded: return "unbounded";
    case lp_overlay_status_t::unbounded_or_infeasible: return "unbounded_or_infeasible";
    case lp_overlay_status_t::iteration_limit: return "iteration_limit";
    case lp_overlay_status_t::time_limit: return "time_limit";
    case lp_overlay_status_t::numerical_issues: return "numerical_issues";
    case lp_overlay_status_t::cutoff: return "cutoff";
    case lp_overlay_status_t::concurrent_limit: return "concurrent_limit";
    case lp_overlay_status_t::work_limit: return "work_limit";
    case lp_overlay_status_t::unsupported: return "unsupported";
    case lp_overlay_status_t::invalid_options: return "invalid_options";
    case lp_overlay_status_t::error: return "error";
  }
  return "unknown";
}

void write_lp_overlay(json_writer_t& writer,
                      const stage_context_t& stage,
                      const lp_overlay_summary_t& summary)
{
  writer.begin_object();
  writer.key("metadata");
  writer.begin_object();
  writer.key("attempted");
  writer.boolean(summary.attempted);
  writer.key("has_optimal_solution");
  writer.boolean(summary.has_optimal_solution);
  writer.key("cuts_added");
  writer.boolean(summary.cuts_added);
  writer.key("objective_mode");
  writer.string(lp_objective_mode_json_name(summary.objective_mode));
  writer.key("status");
  writer.string(lp_overlay_status_json_name(summary.status));
  writer.key("formulation");
  writer.string(summary.formulation);
  writer.key("detail");
  writer.string(summary.detail);
  writer.key("indexing");
  writer.string("source model row and column order");
  writer.key("row_and_family_provenance");
  writer.boolean(true);
  writer.end_object();

  writer.key("limits");
  writer.begin_object();
  writer.key("time_seconds");
  writer.number(summary.time_limit_seconds);
  writer.key("iterations");
  writer.integer(summary.iteration_limit);
  writer.key("threads");
  writer.integer(summary.threads);
  writer.end_object();

  writer.key("solve");
  writer.begin_object();
  writer.key("seconds");
  writer.number(summary.solve_seconds);
  writer.key("iterations");
  writer.integer(summary.iterations);
  writer.key("relaxation_objective");
  write_optional_number(writer, summary.relaxation_objective);
  writer.key("source_objective_at_solution");
  write_optional_number(writer, summary.source_objective_at_solution);
  writer.key("l2_primal_residual");
  write_optional_number(writer, summary.l2_primal_residual);
  writer.key("l2_dual_residual");
  write_optional_number(writer, summary.l2_dual_residual);
  writer.end_object();

  writer.key("integrality");
  writer.begin_object();
  writer.key("integer_variables");
  writer.integer(summary.integer_variables);
  writer.key("binary_variables");
  writer.integer(summary.binary_variables);
  writer.key("fractional_integer_variables");
  writer.integer(summary.fractional_integer_variables);
  writer.key("fractional_binary_variables");
  writer.integer(summary.fractional_binary_variables);
  writer.key("fractional_part_histogram");
  write_lp_histogram(writer, summary.fractional_part_histogram);
  writer.key("distance_to_integrality_histogram");
  write_lp_histogram(writer, summary.distance_to_integrality_histogram);
  writer.end_object();

  writer.key("row_summary");
  write_lp_row_summary(writer, summary.rows);
  writer.key("row_family_summaries");
  writer.begin_array();
  for (const auto& family : summary.row_family_summaries) {
    writer.begin_object();
    writer.key("family");
    writer.string(family.family);
    writer.key("summary");
    write_lp_row_summary(writer, family.summary);
    writer.end_object();
  }
  writer.end_array();

  writer.key("reduced_cost_family_summaries");
  writer.begin_array();
  for (const auto& family : summary.reduced_cost_family_summaries) {
    writer.begin_object();
    writer.key("family");
    writer.string(family.family);
    writer.key("variables");
    writer.integer(family.variables);
    writer.key("negative_reduced_costs");
    writer.integer(family.negative_reduced_costs);
    writer.key("positive_reduced_costs");
    writer.integer(family.positive_reduced_costs);
    writer.key("zero_reduced_costs");
    writer.integer(family.zero_reduced_costs);
    writer.key("signed_reduced_cost");
    write_lp_distribution(writer, family.signed_reduced_cost);
    writer.key("absolute_reduced_cost");
    write_lp_distribution(writer, family.absolute_reduced_cost);
    writer.end_object();
  }
  writer.end_array();

  writer.key("exact_one_summary");
  write_lp_exact_one_summary(writer, summary.exact_one);

  writer.key("vectors");
  writer.begin_object();
  writer.key("primal_values");
  write_double_array(writer, summary.primal_values);
  writer.key("row_duals");
  write_double_array(writer, summary.row_duals);
  writer.key("reduced_costs");
  write_double_array(writer, summary.reduced_costs);
  writer.end_object();

  writer.key("variables");
  writer.begin_array();
  for (const auto& variable : summary.variables) {
    writer.begin_object();
    writer.key("column");
    writer.integer(variable.column);
    writer.key("original_column");
    writer.integer(variable.original_column);
    writer.key("name");
    writer.string(variable.name);
    writer.key("source_type");
    const char source_type[] = {variable.source_type, '\0'};
    writer.string(source_type);
    writer.key("family");
    writer.string(variable.family);
    writer.key("value");
    writer.number(variable.value);
    writer.key("source_objective_coefficient");
    writer.number(variable.source_objective_coefficient);
    writer.key("reduced_cost");
    writer.number(variable.reduced_cost);
    writer.key("fractional_part");
    write_optional_number(writer, variable.fractional_part);
    writer.key("distance_to_integrality");
    write_optional_number(writer, variable.distance_to_integrality);
    writer.key("fractional");
    writer.boolean(variable.fractional);
    writer.end_object();
  }
  writer.end_array();

  writer.key("row_records");
  writer.begin_array();
  for (const auto& row : summary.row_records) {
    writer.begin_object();
    writer.key("row");
    writer.integer(row.row);
    writer.key("stable_id");
    writer.string(row.stable_id);
    writer.key("name");
    writer.string(row.name);
    writer.key("domain");
    writer.string(row.domain);
    writer.key("families");
    write_string_array(writer, row.families);
    writer.key("activity");
    writer.number(row.activity);
    writer.key("lower_bound");
    write_optional_number(writer, row.lower_bound);
    writer.key("upper_bound");
    write_optional_number(writer, row.upper_bound);
    writer.key("lower_slack");
    write_optional_number(writer, row.lower_slack);
    writer.key("upper_slack");
    write_optional_number(writer, row.upper_slack);
    writer.key("nearest_bound_distance");
    write_optional_number(writer, row.nearest_bound_distance);
    writer.key("violation");
    writer.number(row.violation);
    writer.key("dual");
    writer.number(row.dual);
    writer.key("active");
    writer.boolean(row.active);
    writer.end_object();
  }
  writer.end_array();

  writer.key("exact_one_groups");
  writer.begin_array();
  for (const auto& group : summary.exact_one_groups) {
    writer.begin_object();
    writer.key("group");
    writer.integer(group.group);
    writer.key("row");
    writer.integer(group.row);
    writer.key("stable_id");
    writer.string(group.stable_id);
    writer.key("row_provenance");
    write_row_ref(writer, stage, group.row);
    writer.key("members");
    write_literals(writer, stage, group.members);
    writer.key("literal_values");
    write_double_array(writer, group.literal_values);
    writer.key("literal_sum");
    writer.number(group.literal_sum);
    writer.key("entropy");
    writer.number(group.entropy);
    writer.key("effective_support");
    writer.number(group.effective_support);
    writer.key("largest_value");
    writer.number(group.largest_value);
    writer.key("second_largest_value");
    writer.number(group.second_largest_value);
    writer.key("margin");
    writer.number(group.margin);
    writer.key("leading_literal");
    group.leading_literal.has_value() ? write_literal(writer, stage, *group.leading_literal)
                                      : writer.null();
    writer.key("second_literal");
    group.second_literal.has_value() ? write_literal(writer, stage, *group.second_literal)
                                     : writer.null();
    writer.key("fractional_members");
    writer.integer(group.fractional_members);
    writer.key("integral");
    writer.boolean(group.integral);
    writer.key("nearly_integral");
    writer.boolean(group.nearly_integral);
    writer.end_object();
  }
  writer.end_array();
  writer.end_object();
}

void write_numerical(json_writer_t& writer, const numerical_summary_t& numerical)
{
  writer.begin_object();
  writer.key("coefficient_magnitudes");
  write_quantiles(writer, numerical.coefficient_magnitudes);
  writer.key("row_dynamic_ranges");
  write_quantiles(writer, numerical.row_dynamic_ranges);
  writer.key("column_l2_norms");
  write_quantiles(writer, numerical.column_l2_norms);
  writer.key("positive_coefficients");
  writer.integer(numerical.positive_coefficients);
  writer.key("negative_coefficients");
  writer.integer(numerical.negative_coefficients);
  writer.key("near_zero_coefficients");
  writer.integer(numerical.near_zero_coefficients);
  writer.key("near_cancellation_rows");
  write_integer_array(writer, numerical.near_cancellation_rows);
  writer.key("decimal_scale_exponent_histogram");
  write_numeric_histogram(writer, numerical.decimal_scale_exponent_histogram);
  writer.key("rows_not_decimal_scalable");
  writer.integer(numerical.rows_not_decimal_scalable);
  writer.end_object();
}

void write_repair(json_writer_t& writer, const stage_context_t& stage)
{
  const auto& repair = stage.analysis.repair;
  writer.begin_object();
  writer.key("monotone_increase_columns");
  write_integer_array(writer, repair.monotone_increase_columns);
  writer.key("monotone_increase_original_columns");
  write_original_columns(writer, stage, repair.monotone_increase_columns);
  writer.key("monotone_decrease_columns");
  write_integer_array(writer, repair.monotone_decrease_columns);
  writer.key("monotone_decrease_original_columns");
  write_original_columns(writer, stage, repair.monotone_decrease_columns);
  writer.key("free_equality_pivots");
  write_integer_array(writer, repair.free_equality_pivots);
  writer.key("free_equality_original_pivots");
  write_original_columns(writer, stage, repair.free_equality_pivots);
  writer.key("two_variable_affine_rows");
  write_integer_array(writer, repair.two_variable_affine_rows);
  writer.key("difference_components");
  write_components(writer, stage, repair.difference_components);
  writer.key("flow_equality_components");
  write_components(writer, stage, repair.flow_equality_components);
  writer.end_object();
}

void write_stage(json_writer_t& writer,
                 const structure_model_t& model,
                 const model_analysis_t& analysis,
                 bool root_lp_overlay_included)
{
  const stage_context_t stage{model, analysis};
  writer.begin_object();
  writer.key("scope");
  writer.string(analysis.scope);
  writer.key("detector_metadata");
  write_detector_metadata(writer, analysis, root_lp_overlay_included);
  writer.key("model");
  write_model(writer, stage);
  writer.key("original_column_ids");
  write_integer_array(writer, analysis.original_column_ids);
  writer.key("original_row_ids");
  write_integer_array(writer, analysis.original_row_ids);
  writer.key("row_domain_counts");
  write_count_map(writer, analysis.row_domain_counts);
  writer.key("row_family_counts");
  write_count_map(writer, analysis.row_family_counts);
  writer.key("rows");
  write_rows(writer, stage);
  writer.key("recognized_groups");
  writer.begin_object();
  writer.key("exact_one");
  write_groups(writer, stage, analysis.exact_one_groups);
  writer.key("set_packing");
  write_groups(writer, stage, analysis.set_packing_groups);
  writer.key("set_covering");
  write_groups(writer, stage, analysis.set_covering_groups);
  writer.key("gub");
  write_groups(writer, stage, analysis.gub_groups);
  writer.key("knapsack");
  write_groups(writer, stage, analysis.knapsack_rows);
  writer.end_object();
  writer.key("exact_one_system");
  writer.begin_object();
  writer.key("binary_variables");
  writer.integer(analysis.binary_variables);
  writer.key("covered_binary_variables");
  writer.integer(analysis.exact_one_covered_binary_variables);
  writer.key("covered_binary_percentage");
  writer.number(analysis.binary_variables == 0
                  ? 0.0
                  : 100.0 * static_cast<double>(analysis.exact_one_covered_binary_variables) /
                      static_cast<double>(analysis.binary_variables));
  writer.key("groups_overlap");
  writer.boolean(analysis.exact_one_groups_overlap);
  writer.key("group_size_histogram");
  write_integer_array(writer, analysis.exact_one_group_size_histogram);
  writer.key("overlap_component_sizes");
  write_integer_array(writer, analysis.exact_one_overlap_component_sizes);
  writer.end_object();
  writer.key("implications");
  write_implications(writer, stage);
  writer.key("variable_bounds");
  write_variable_bounds(writer, stage);
  writer.key("affine_definitions");
  write_affine_definitions(writer, stage);
  writer.key("exact_one_block_graph");
  writer.begin_object();
  writer.key("materialization");
  writer.begin_object();
  writer.key("pair_provenance_limit_per_kind");
  writer.integer(analysis.exact_one_block_pair_provenance_limit);
  writer.key("pair_edges_complete");
  writer.boolean(analysis.exact_one_block_edges_complete);
  writer.key("candidate_pair_provenance");
  write_count_map(writer, analysis.exact_one_block_candidate_pair_provenance);
  writer.key("materialized_pair_provenance");
  write_count_map(writer, analysis.exact_one_block_materialized_pair_provenance);
  writer.key("partial_projection_fields");
  writer.string(
    "when complete=false, pair-derived edge/weight/connectivity fields describe only the "
    "materialized prefix; mediator hyperedges and their incident-row paths remain complete");
  writer.end_object();
  writer.key("edges");
  write_block_edges(writer, stage);
  writer.key("mediators");
  write_block_mediators(writer, stage);
  writer.key("projections");
  write_graph_projections(writer, stage);
  writer.end_object();
  writer.key("decompositions");
  write_decompositions(writer, stage);
  writer.key("refinement");
  write_refinement(writer, analysis.refinement);
  writer.key("numerical");
  write_numerical(writer, analysis.numerical);
  writer.key("repair");
  write_repair(writer, stage);
  writer.end_object();
}

void write_signed_count_map(json_writer_t& writer,
                            const std::map<std::string, std::int64_t>& counts)
{
  writer.begin_object();
  for (const auto& [name, count] : counts) {
    writer.key(name);
    writer.integer(count);
  }
  writer.end_object();
}

void write_presolve_delta(json_writer_t& writer, const presolve_structure_delta_t& delta)
{
  writer.begin_object();
  writer.key("row_domain_net_changes");
  write_signed_count_map(writer, delta.row_domain_net_changes);
  writer.key("row_family_net_changes");
  write_signed_count_map(writer, delta.row_family_net_changes);
  writer.key("exact_one");
  writer.begin_object();
  writer.key("preserved");
  writer.integer(delta.exact_one.preserved);
  writer.key("contracted");
  writer.integer(delta.exact_one.contracted);
  writer.key("split");
  writer.integer(delta.exact_one.split);
  writer.key("destroyed");
  writer.integer(delta.exact_one.destroyed);
  writer.key("merged");
  writer.integer(delta.exact_one.merged);
  writer.end_object();
  writer.key("eliminated_original_columns");
  write_integer_array(writer, delta.eliminated_original_columns);
  writer.key("eliminated_original_rows");
  write_integer_array(writer, delta.eliminated_original_rows);
  writer.key("implied_integer_columns");
  writer.begin_array();
  for (const auto& column : delta.implied_integer_columns) {
    writer.begin_object();
    writer.key("reduced_column");
    writer.integer(column.reduced_column);
    writer.key("original_column");
    writer.integer(column.original_column);
    writer.end_object();
  }
  writer.end_array();
  writer.key("original_refinement_tie_classes");
  writer.integer(delta.original_refinement_tie_classes);
  writer.key("reduced_refinement_tie_classes");
  writer.integer(delta.reduced_refinement_tie_classes);
  writer.end_object();
}

}  // namespace

void write_structure_json(const std::string& path,
                          const structure_model_t& original_model,
                          const model_analysis_t& original,
                          const structure_model_t* reduced_model,
                          const model_analysis_t* reduced,
                          const presolve_structure_delta_t* delta,
                          const lp_overlay_summary_t* source_lp,
                          const lp_overlay_summary_t* objective_erased_lp)
{
  // RESEARCH-BREADCRUMB(mps-structure/report-detail-levels) [driver-local]
  // Let a typed render policy choose summary, relation, or full sections without rerunning any
  // detector. Keep schema versioning and completeness semantics identical across detail levels.

  // RESEARCH-BREADCRUMB(mps-structure/safe-json-sidecar) [driver-local]
  // Write to a same-directory temporary file and rename only after writer.complete() and stream
  // checks succeed. Input/output equivalence must already have been rejected by the driver.
  if ((reduced_model == nullptr) != (reduced == nullptr)) {
    throw std::invalid_argument("reduced model and analysis must either both be present or absent");
  }
  if (delta != nullptr && reduced == nullptr) {
    throw std::invalid_argument("presolve structure delta requires a reduced model analysis");
  }

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) { throw std::runtime_error("could not open JSON output file: " + path); }
  output.imbue(std::locale::classic());

  json_writer_t writer(output);
  writer.begin_object();
  writer.key("schema_version");
  writer.string("1.2");
  writer.key("detector");
  writer.begin_object();
  writer.key("name");
  writer.string("cuopt_static_mps_structure");
  writer.key("machine_readable_detail");
  writer.boolean(true);
  writer.key("root_lp_overlay_included");
  writer.boolean(source_lp != nullptr || objective_erased_lp != nullptr);
  writer.key("source_objective_lp_overlay_included");
  writer.boolean(source_lp != nullptr);
  writer.key("objective_erased_lp_overlay_included");
  writer.boolean(objective_erased_lp != nullptr);
  writer.key("probing_overlay_included");
  writer.boolean(false);
  writer.end_object();
  writer.key("stages");
  writer.begin_object();
  writer.key("original");
  write_stage(
    writer, original_model, original, source_lp != nullptr || objective_erased_lp != nullptr);
  if (reduced_model != nullptr) {
    writer.key("presolved");
    write_stage(writer, *reduced_model, *reduced, false);
  }
  writer.end_object();
  writer.key("root_lp_overlays");
  writer.begin_object();
  writer.key("source_objective");
  if (source_lp == nullptr) {
    writer.null();
  } else {
    write_lp_overlay(writer, stage_context_t{original_model, original}, *source_lp);
  }
  writer.key("objective_erased");
  if (objective_erased_lp == nullptr) {
    writer.null();
  } else {
    write_lp_overlay(writer, stage_context_t{original_model, original}, *objective_erased_lp);
  }
  writer.end_object();
  if (delta != nullptr) {
    writer.key("presolve_delta");
    write_presolve_delta(writer, *delta);
  }
  writer.end_object();
  output << '\n';

  if (!writer.complete() || !output) {
    throw std::runtime_error("failed to write JSON output file: " + path);
  }
}

}  // namespace cuopt::mathematical_optimization::examples
