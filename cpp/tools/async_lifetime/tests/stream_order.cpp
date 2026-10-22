/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "apis.hpp"

void wrong_stream(rmm::device_scalar<float>& weight,
                  cuda::stream_ref stream,
                  cuda::stream_ref other)
{  // expected-warning {{asynchronous}}
  float value = 1.0f;
  weight.set_value_async(value, stream);
  other.sync();
}

void conditional_sync(rmm::device_scalar<float>& weight, cuda::stream_ref stream, bool sync)
{  // expected-warning {{asynchronous}}
  float value = 1.0f;
  weight.set_value_async(value, stream);
  if (sync) stream.sync();
}

void failed_synchronization(rmm::device_scalar<float>& weight, cuda::stream_ref stream)
{
  float value = 1.0f;
  weight.set_value_async(value, stream);
  if (cudaStreamSynchronize(stream.get()) != cudaSuccess) {
    return;  // expected-warning {{asynchronous}}
  }
}

void successful_synchronization(rmm::device_scalar<float>& weight, cuda::stream_ref stream)
{
  float value = 1.0f;
  weight.set_value_async(value, stream);
  if (cudaStreamSynchronize(stream.get()) != cudaSuccess) __builtin_unreachable();
}
