/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "apis.hpp"

// The caller must still be checked when the analyzer cannot inline this body.
// analyzer-config {{ipa=none}}
void deferred_copy_wrapper(float* destination, const float* source, cudaStream_t stream)
{
  cudaMemcpyAsync(destination, source, sizeof(float), cudaMemcpyHostToDevice, stream);
}

void local_passed_to_skipped_wrapper(float* destination, cudaStream_t stream)
{  // expected-warning {{asynchronous}}
  float value = 1.0f;
  deferred_copy_wrapper(destination, &value, stream);
}

void skipped_wrapper_completed_by_caller(float* destination, cudaStream_t stream)
{
  float value = 1.0f;
  deferred_copy_wrapper(destination, &value, stream);
  if (cudaStreamSynchronize(stream) != cudaSuccess) __builtin_unreachable();
}
