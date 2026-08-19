/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>

namespace cuopt::mathematical_optimization::examples {

void print_conflict_graph_summary(const io::mps_data_model_t<int, double>& model);

}  // namespace cuopt::mathematical_optimization::examples
