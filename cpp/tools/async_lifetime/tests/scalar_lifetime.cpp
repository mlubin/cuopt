/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "apis.hpp"

void original_weights_pattern(rmm::device_scalar<float>& weight, cuda::stream_ref stream)
{  // expected-warning {{asynchronous}}
  float one = 1.0f;
  weight.set_value_async(one, stream);
}

void constructor_borrows_local(cuda::stream_ref stream)
{  // expected-warning {{asynchronous}}
  float one = 1.0f;
  rmm::device_scalar<float> weight(one, stream);
}

void const_is_not_static(rmm::device_scalar<float>& weight, cuda::stream_ref stream)
{  // expected-warning {{asynchronous}}
  const float one = 1.0f;
  weight.set_value_async(one, stream);
}

void constexpr_is_not_static(rmm::device_scalar<float>& weight, cuda::stream_ref stream)
{  // expected-warning {{asynchronous}}
  constexpr float one = 1.0f;
  weight.set_value_async(one, stream);
}

void by_value_parameter(rmm::device_scalar<float>& weight, float value, cuda::stream_ref stream)
{
  weight.set_value_async(value, stream);  // expected-warning {{asynchronous}}
}

void reference_alias(rmm::device_scalar<float>& weight, cuda::stream_ref stream)
{  // expected-warning {{asynchronous}}
  float value        = 1.0f;
  const float& alias = value;
  weight.set_value_async(alias, stream);
}

void element_borrows_local(rmm::device_uvector<float>& weights, cuda::stream_ref stream)
{  // expected-warning {{asynchronous}}
  float value = 1.0f;
  weights.set_element_async(0, value, stream);
}

inline void submit_scalar(rmm::device_scalar<float>& weight,
                          const float& value,
                          cuda::stream_ref stream)
{
  weight.set_value_async(value, stream);
}

void temporary_argument_through_wrapper(rmm::device_scalar<float>& weight, cuda::stream_ref stream)
{
  submit_scalar(weight, 1.0f, stream);  // expected-warning {{asynchronous}}
}

void nested_scope_expires_before_sync(rmm::device_scalar<float>& weight, cuda::stream_ref stream)
{
  {  // expected-warning {{asynchronous}}
    float value = 1.0f;
    weight.set_value_async(value, stream);
  }
  stream.sync();
}
