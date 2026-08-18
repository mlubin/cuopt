# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

add_executable(mps_presolve_structure mps_presolve_structure.cu)

target_include_directories(mps_presolve_structure
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../src"
    "${pslp_SOURCE_DIR}/include"
)

target_link_libraries(mps_presolve_structure
  PRIVATE
    cuopt_static
    ${CUOPT_PRIVATE_CUDA_LIBS}
)

if(NOT DEFINED INSTALL_TARGET OR "${INSTALL_TARGET}" STREQUAL "")
  target_link_options(mps_presolve_structure PRIVATE -Wl,--enable-new-dtags)
endif()
