/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "apis.hpp"

void raft_source(float* destination, cuda::stream_ref stream)
{  // expected-warning {{asynchronous}}
  float value = 1.0f;
  raft::copy(destination, &value, 1, stream);
}

void pointer_alias(float* destination, cuda::stream_ref stream)
{  // expected-warning {{asynchronous}}
  float value        = 1.0f;
  const float* alias = &value;
  raft::update_device(destination, alias, 1, stream);
}

struct host_buffer {
  float values[4];
  float* data() { return values; }
};

void container_storage(float* destination, cuda::stream_ref stream)
{  // expected-warning {{asynchronous}}
  host_buffer buffer{};
  raft::copy(destination, buffer.data(), 4, stream);
}

void deferred_output(const float* source, cuda::stream_ref stream)
{  // expected-warning {{asynchronous}}
  float value;
  raft::update_host(&value, source, 1, stream);
}

float read_output_early(const float* source, cuda::stream_ref stream)
{
  float value;
  raft::update_host(&value, source, 1, stream);
  float result = value;  // expected-warning {{asynchronous}}
  stream.sync();
  return result;
}

void mutate_input_early(float* destination, cuda::stream_ref stream)
{
  float value = 1.0f;
  raft::copy(destination, &value, 1, stream);
  value = 2.0f;  // expected-warning {{asynchronous}}
  stream.sync();
}

void mutate_static_input(float* destination, cuda::stream_ref stream)
{
  static float value = 1.0f;
  raft::copy(destination, &value, 1, stream);
  value = 2.0f;  // expected-warning {{asynchronous}}
  stream.sync();
}

void mutate_reference_input(float* destination, float& value, cuda::stream_ref stream)
{
  raft::copy(destination, &value, 1, stream);
  value = 2.0f;  // expected-warning {{asynchronous}}
  stream.sync();
}

inline void submit_copy(float* destination, const float* source, cuda::stream_ref stream)
{
  raft::copy(destination, source, 1, stream);
}

void wrapper_body_is_followed(float* destination, cuda::stream_ref stream)
{  // expected-warning {{asynchronous}}
  float value = 1.0f;
  submit_copy(destination, &value, stream);
}

void raw_cuda_source(float* destination, cudaStream_t stream)
{  // expected-warning {{asynchronous}}
  float value = 1.0f;
  cudaMemcpyAsync(destination, &value, sizeof(value), cudaMemcpyHostToDevice, stream);
}
