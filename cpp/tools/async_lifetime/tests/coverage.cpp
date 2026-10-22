/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "apis.hpp"

// A manifest entry marked unsupported must never silently certify a call.
// expected-no-diagnostics
// expected-incomplete {{cudaGraphLaunch}}
void graph_replay_needs_a_supported_contract(cudaGraphExec_t graph, cudaStream_t stream)
{
  cudaGraphLaunch(graph, stream);
}
