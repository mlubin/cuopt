/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "apis.hpp"

#include <utility>
#include <vector>

void vector_input_expires(float* destination, cuda::stream_ref stream)
{
  std::vector<float> values(4, 1.0f);
  raft::copy(destination, values.data(), values.size(), stream);
}  // expected-warning {{asynchronous}}

void vector_input_synchronized(float* destination, cuda::stream_ref stream)
{
  std::vector<float> values(4, 1.0f);
  raft::copy(destination, values.data(), values.size(), stream);
  stream.sync();
}

void vector_storage_survives_move(std::vector<float>& owner,
                                  float* destination,
                                  cuda::stream_ref stream)
{
  std::vector<float> values(4, 1.0f);
  raft::copy(destination, values.data(), values.size(), stream);
  owner = std::move(values);
}

std::vector<float> vector_storage_survives_return(float* destination, cuda::stream_ref stream)
{
  std::vector<float> values(4, 1.0f);
  raft::copy(destination, values.data(), values.size(), stream);
  return values;
}

void vector_clear_does_not_release_storage(float* destination, cuda::stream_ref stream)
{
  std::vector<float> values(4, 1.0f);
  raft::copy(destination, values.data(), values.size(), stream);
  values.clear();
  stream.sync();
}
