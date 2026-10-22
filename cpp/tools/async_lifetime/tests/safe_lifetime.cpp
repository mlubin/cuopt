/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// expected-no-diagnostics
#include "apis.hpp"

void fixed_weights_pattern(cuda::stream_ref stream)
{
  rmm::device_scalar<float> weight(cuopt::one_v<float>, stream);
}

void static_value(rmm::device_scalar<float>& weight, cuda::stream_ref stream)
{
  static const float one = 1.0f;
  weight.set_value_async(one, stream);
}

void reference_parameter(rmm::device_scalar<float>& weight,
                         const float& value,
                         cuda::stream_ref stream)
{
  weight.set_value_async(value, stream);
}

void synchronized_local(rmm::device_scalar<float>& weight, cuda::stream_ref stream)
{
  float value = 1.0f;
  weight.set_value_async(value, stream);
  stream.sync();
}

void copied_stream_keeps_identity(rmm::device_scalar<float>& weight, cuda::stream_ref stream)
{
  float value = 1.0f;
  auto alias  = stream;
  weight.set_value_async(value, alias);
  stream.sync();
}

void synchronous_scalar_getter(rmm::device_scalar<float>& weight, cuda::stream_ref stream)
{
  float value = 1.0f;
  weight.set_value_async(value, stream);
  (void)weight.value(stream);
}

void synchronous_element_setter(rmm::device_uvector<float>& weights, cuda::stream_ref stream)
{
  float value = 1.0f;
  weights.set_element(0, value, stream);
}

void synchronous_wrapper(float* destination, cuda::stream_ref stream)
{
  float value = 1.0f;
  raft::copy(destination, &value, 1, stream);
  stream.sync();
}

void mutate_after_sync(rmm::device_scalar<float>& weight, cuda::stream_ref stream)
{
  float value = 1.0f;
  weight.set_value_async(value, stream);
  stream.sync();
  value = 2.0f;
}

void memset_fill_is_passed_by_value(float* destination, cudaStream_t stream)
{
  int fill = 0;
  cudaMemsetAsync(destination, fill, sizeof(float), stream);
}

void scalar_zero_has_no_host_source(rmm::device_scalar<float>& weight, cuda::stream_ref stream)
{
  weight.set_value_to_zero_async(stream);
}
