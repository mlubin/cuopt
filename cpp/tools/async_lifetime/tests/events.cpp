/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "apis.hpp"

void event_covers_submission(rmm::device_scalar<float>& weight,
                             cuda::stream_ref stream,
                             cudaEvent_t event)
{
  float value = 1.0f;
  weight.set_value_async(value, stream);
  if (cudaEventRecord(event, stream.get()) != cudaSuccess) __builtin_unreachable();
  if (cudaEventSynchronize(event) != cudaSuccess) __builtin_unreachable();
}

void event_precedes_submission(rmm::device_scalar<float>& weight,
                               cuda::stream_ref stream,
                               cudaEvent_t event)
{  // expected-warning {{asynchronous}}
  if (cudaEventRecord(event, stream.get()) != cudaSuccess) __builtin_unreachable();
  float value = 1.0f;
  weight.set_value_async(value, stream);
  if (cudaEventSynchronize(event) != cudaSuccess) __builtin_unreachable();
}

void wait_does_not_complete_host_access(rmm::device_scalar<float>& weight,
                                        cuda::stream_ref stream,
                                        cuda::stream_ref consumer,
                                        cudaEvent_t event)
{  // expected-warning {{asynchronous}}
  float value = 1.0f;
  weight.set_value_async(value, stream);
  if (cudaEventRecord(event, stream.get()) != cudaSuccess) __builtin_unreachable();
  if (cudaStreamWaitEvent(consumer.get(), event, 0) != cudaSuccess) __builtin_unreachable();
}

void wait_then_sync_completes_dependency(rmm::device_scalar<float>& weight,
                                         cuda::stream_ref stream,
                                         cuda::stream_ref consumer,
                                         cudaEvent_t event)
{
  float value = 1.0f;
  weight.set_value_async(value, stream);
  if (cudaEventRecord(event, stream.get()) != cudaSuccess) __builtin_unreachable();
  if (cudaStreamWaitEvent(consumer.get(), event, 0) != cudaSuccess) __builtin_unreachable();
  consumer.sync();
}

void successful_query(rmm::device_scalar<float>& weight, cuda::stream_ref stream)
{
  float value = 1.0f;
  weight.set_value_async(value, stream);
  if (cudaStreamQuery(stream.get()) != cudaSuccess) __builtin_unreachable();
}

void pending_query(rmm::device_scalar<float>& weight, cuda::stream_ref stream)
{  // expected-warning {{asynchronous}}
  float value = 1.0f;
  weight.set_value_async(value, stream);
  if (cudaStreamQuery(stream.get()) == cudaSuccess) __builtin_unreachable();
}
