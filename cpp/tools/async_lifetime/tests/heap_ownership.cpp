/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "apis.hpp"

void delete_input_early(float* destination, cuda::stream_ref stream)
{
  auto* value = new float(1.0f);
  raft::copy(destination, value, 1, stream);
  delete value;  // expected-warning {{asynchronous}}
}

void delete_output_early(const float* source, cuda::stream_ref stream)
{
  auto* value = new float;
  raft::update_host(value, source, 1, stream);
  delete value;  // expected-warning {{asynchronous}}
}

void delete_array_alias_early(float* destination, cuda::stream_ref stream)
{
  auto* values = new float[4];
  float* alias = values;
  raft::copy(destination, values, 4, stream);
  delete[] alias;  // expected-warning {{asynchronous}}
}

void synchronize_before_delete(float* destination, cuda::stream_ref stream)
{
  auto* value = new float(1.0f);
  raft::copy(destination, value, 1, stream);
  stream.sync();
  delete value;
}

float* pointer_handle_is_not_allocation(float* destination, cuda::stream_ref stream)
{
  auto* value = new float(1.0f);
  raft::copy(destination, value, 1, stream);
  return value;
}
