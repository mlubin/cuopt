/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "apis.hpp"

// A successful device-wide sync completes the current device, whose identity
// cannot yet be related to an arbitrary stream in this fixture.
// expected-incomplete {{device-wide completion requires current-device}}
void device_sync_requires_device_identity(rmm::device_scalar<float>& weight,
                                          cuda::stream_ref stream)
{  // expected-warning {{asynchronous}}
  float value = 1.0f;
  weight.set_value_async(value, stream);
  if (cudaDeviceSynchronize() != cudaSuccess) __builtin_unreachable();
}
