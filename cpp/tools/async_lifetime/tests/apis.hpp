/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// These declarations deliberately have no CUDA, GPU, or third-party dependency.
// Qualified names and argument positions match the APIs in contracts.json.
using size_t                      = __SIZE_TYPE__;
using cudaStream_t                = void*;
using cudaEvent_t                 = void*;
using cudaGraphExec_t             = void*;
using cudaError_t                 = int;
constexpr cudaError_t cudaSuccess = 0;
enum cudaMemcpyKind {
  cudaMemcpyHostToHost,
  cudaMemcpyHostToDevice,
  cudaMemcpyDeviceToHost,
  cudaMemcpyDeviceToDevice,
  cudaMemcpyDefault
};

cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
cudaError_t cudaMemsetAsync(void*, int, size_t, cudaStream_t);
cudaError_t cudaMallocAsync(void**, size_t, cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaStreamQuery(cudaStream_t);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t);
cudaError_t cudaEventSynchronize(cudaEvent_t);
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned);
cudaError_t cudaFreeAsync(void*, cudaStream_t);
cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t);

namespace cuda {
class stream_ref {
 public:
  explicit stream_ref(cudaStream_t value) : value_(value) {}
  cudaStream_t get() const { return value_; }
  void sync() const;

 private:
  cudaStream_t value_;
};
}  // namespace cuda

namespace rmm {
template <typename T>
class device_scalar {
 public:
  explicit device_scalar(cuda::stream_ref);
  device_scalar(const T&, cuda::stream_ref);
  device_scalar(const T&&, cuda::stream_ref) = delete;
  void set_value_async(const T&, cuda::stream_ref);
  void set_value_async(const T&&, cuda::stream_ref) = delete;
  void set_value_to_zero_async(cuda::stream_ref);
  T value(cuda::stream_ref) const;
};

template <typename T>
class device_uvector {
 public:
  void set_element_async(size_t, const T&, cuda::stream_ref);
  void set_element_async(size_t, const T&&, cuda::stream_ref) = delete;
  void set_element(size_t, const T&, cuda::stream_ref);
};
}  // namespace rmm

namespace raft {
template <typename T>
void copy(T*, const T*, size_t, cuda::stream_ref);
template <typename T>
void update_device(T*, const T*, size_t, cuda::stream_ref);
template <typename T>
void update_host(T*, const T*, size_t, cuda::stream_ref);
}  // namespace raft

namespace cuopt {
template <typename T>
inline constexpr T one_v = T{1};

namespace internals {
class base_solution_callback_t {
 public:
  void set_user_data(void*);
};
}  // namespace internals

namespace mathematical_optimization {
template <typename I, typename T>
class mip_solver_settings_t {
 public:
  void set_mip_callback(internals::base_solution_callback_t*, void*);
};

namespace io {
template <typename I, typename T>
class data_model_view_t {
 public:
  void set_objective_coefficients(const T*, I);
};
}  // namespace io
}  // namespace mathematical_optimization
}  // namespace cuopt
