#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

rapids-logger "Create checks conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-dependency-file-generator \
  --output conda \
  --file-key checks \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n checks
conda activate checks

# Run pre-commit checks
pre-commit run --all-files --show-diff-on-failure

rapids-logger "Create asynchronous lifetime analysis environment"
rapids-dependency-file-generator \
  --output conda \
  --file-key async_lifetime_checks \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env-async-lifetime.yaml

rapids-mamba-retry env create --yes -f env-async-lifetime.yaml -n async-lifetime-checks
conda activate async-lifetime-checks

rapids-logger "Build and test asynchronous lifetime checker"
cmake -S cpp/tools/async_lifetime -B cpp/build/async_lifetime/checker -G Ninja \
  -DCMAKE_CXX_COMPILER="${CONDA_PREFIX}/bin/clang++" \
  -DLLVM_DIR="${CONDA_PREFIX}/lib/cmake/llvm" \
  -DClang_DIR="${CONDA_PREFIX}/lib/cmake/clang"
cmake --build cpp/build/async_lifetime/checker
ctest --test-dir cpp/build/async_lifetime/checker --output-on-failure
