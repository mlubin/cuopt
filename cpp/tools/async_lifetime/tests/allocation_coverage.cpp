/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "apis.hpp"

// This constructor has no borrowed host value, but the owning device allocation
// still needs an identity/destruction model before analysis can claim completion.
// expected-no-diagnostics
// expected-incomplete {{allocation output identity is not modeled}}
void allocation_only_constructor(cuda::stream_ref stream)
{
  rmm::device_scalar<float> weight(stream);
}
