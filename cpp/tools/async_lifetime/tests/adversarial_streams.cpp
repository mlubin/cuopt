/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "apis.hpp"

void reassigned_stream_alias(rmm::device_scalar<float>& weight,
                             cuda::stream_ref stream,
                             cuda::stream_ref other)
{  // expected-warning {{asynchronous}}
  float value = 1.0f;
  auto alias  = stream;
  weight.set_value_async(value, alias);
  alias = other;
  alias.sync();
}

void synchronize_original_after_alias_changes(rmm::device_scalar<float>& weight,
                                              cuda::stream_ref stream,
                                              cuda::stream_ref other)
{
  float value = 1.0f;
  auto alias  = stream;
  weight.set_value_async(value, alias);
  alias = other;
  stream.sync();
}

void failed_record_preserves_old_event_fence(rmm::device_scalar<float>& weight,
                                             cuda::stream_ref stream,
                                             cudaEvent_t event)
{  // expected-warning {{asynchronous}}
  if (cudaEventRecord(event, stream.get()) != cudaSuccess) __builtin_unreachable();
  float value = 1.0f;
  weight.set_value_async(value, stream);
  if (cudaEventRecord(event, stream.get()) == cudaSuccess) __builtin_unreachable();
  if (cudaEventSynchronize(event) != cudaSuccess) __builtin_unreachable();
}

void failed_wait_does_not_create_dependency(rmm::device_scalar<float>& weight,
                                            cuda::stream_ref stream,
                                            cuda::stream_ref consumer,
                                            cudaEvent_t event)
{  // expected-warning {{asynchronous}}
  float value = 1.0f;
  weight.set_value_async(value, stream);
  if (cudaEventRecord(event, stream.get()) != cudaSuccess) __builtin_unreachable();
  if (cudaStreamWaitEvent(consumer.get(), event, 0) == cudaSuccess) __builtin_unreachable();
  consumer.sync();
}
