/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "apis.hpp"

void same_stream_deallocation(const float* source, cudaStream_t stream)
{
  void* destination;
  if (cudaMallocAsync(&destination, sizeof(float), stream) != cudaSuccess) return;
  cudaMemcpyAsync(destination, source, sizeof(float), cudaMemcpyDeviceToDevice, stream);
  cudaFreeAsync(destination, stream);
}

void wrong_stream_deallocation(const float* source, cudaStream_t stream, cudaStream_t other)
{
  void* destination;
  if (cudaMallocAsync(&destination, sizeof(float), stream) != cudaSuccess) return;
  cudaMemcpyAsync(destination, source, sizeof(float), cudaMemcpyDeviceToDevice, stream);
  cudaFreeAsync(destination, other);  // expected-warning {{asynchronous}}
}

void free_after_sync(const float* source, cudaStream_t stream, cudaStream_t other)
{
  void* destination;
  if (cudaMallocAsync(&destination, sizeof(float), stream) != cudaSuccess) return;
  cudaMemcpyAsync(destination, source, sizeof(float), cudaMemcpyDeviceToDevice, stream);
  if (cudaStreamSynchronize(stream) != cudaSuccess) __builtin_unreachable();
  cudaFreeAsync(destination, other);
}
