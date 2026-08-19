/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>

namespace cuopt::mathematical_optimization::examples {

// RESEARCH-BREADCRUMB(mps-structure/typed-conflict-report) [driver-local]
// Return a budgeted typed result here and make this printer plus JSON consume it. Preserve complete
// clique hyperedges and cap only pair expansion; follow the shared contract in README.md.
void print_conflict_graph_summary(const io::mps_data_model_t<int, double>& model);

}  // namespace cuopt::mathematical_optimization::examples
