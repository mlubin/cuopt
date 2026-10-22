/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "apis.hpp"

// An arbitrary method name does not establish a stable RAFT/CUDA stream identity.
// expected-incomplete {{get_stream}}
namespace other {
class resource {
 public:
  cuda::stream_ref get_stream() const;
};
}  // namespace other

void unrelated_getter_is_not_a_handle(rmm::device_scalar<float>& weight,
                                      const other::resource& resource)
{  // expected-warning {{asynchronous}}
  float value = 1.0f;
  weight.set_value_async(value, resource.get_stream());
  resource.get_stream().sync();
}
