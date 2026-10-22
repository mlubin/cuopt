/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "apis.hpp"

// expected-no-diagnostics
// expected-incomplete {{opaque_mutation}}
// expected-incomplete {{opaque_global_mutation}}
// expected-incomplete {{cudaMemcpy3DAsync}}
// expected-incomplete {{cublasSaxpy_v2}}

void opaque_mutation(float&);
void opaque_global_mutation();
float* escaped_value;
struct cudaMemcpy3DParms;
cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms*, cudaStream_t);
int cublasSaxpy_v2(void*, int, const float*, const float*, int, float*, int);

void possible_mutation_while_copy_is_pending(float* destination, cuda::stream_ref stream)
{
  float value = 1.0f;
  raft::copy(destination, &value, 1, stream);
  opaque_mutation(value);
  stream.sync();
}

void unknown_cuda_copy(const cudaMemcpy3DParms* descriptor, cudaStream_t stream)
{
  cudaMemcpy3DAsync(descriptor, stream);
}

void global_alias_may_be_modified_while_copy_is_pending(float* destination, cuda::stream_ref stream)
{
  float value   = 1.0f;
  escaped_value = &value;
  raft::copy(destination, &value, 1, stream);
  opaque_global_mutation();
  stream.sync();
  escaped_value = nullptr;
}

void unknown_library_operation(void* handle, const float* alpha, const float* x, float* y)
{
  cublasSaxpy_v2(handle, 4, alpha, x, 1, y, 1);
}
