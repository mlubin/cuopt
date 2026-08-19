# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

add_library(mps_structure_analysis STATIC
  mps_structure_analysis.cpp
  mps_structure_json.cpp
)

target_include_directories(mps_structure_analysis
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(mps_structure_analysis
  PUBLIC
    cuopt_static
)

ConfigureTest(MPS_STRUCTURE_ANALYSIS_TEST
  mps_structure_analysis_test.cpp
  LABELS numopt
  STATIC_LIB)

target_link_libraries(MPS_STRUCTURE_ANALYSIS_TEST
  PRIVATE
    mps_structure_analysis
)

ConfigureTest(MPS_LP_OVERLAY_TEST
  mps_lp_overlay.cpp
  mps_lp_overlay_test.cpp
  LABELS numopt
  STATIC_LIB)

target_include_directories(MPS_LP_OVERLAY_TEST
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../src"
)

target_link_libraries(MPS_LP_OVERLAY_TEST
  PRIVATE
    mps_structure_analysis
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # GCC diagnoses an existing dependent-expression parse in a transitive barrier header.
  target_compile_options(MPS_LP_OVERLAY_TEST PRIVATE -Wno-error=missing-template-keyword)
endif()

ConfigureTest(MPS_PRESOLVE_ROW_MAP_TEST
  mps_presolve_row_map_test.cpp
  LABELS numopt
  STATIC_LIB)

target_include_directories(MPS_PRESOLVE_ROW_MAP_TEST
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../src"
    "${CMAKE_CURRENT_SOURCE_DIR}/../.."
    "${pslp_SOURCE_DIR}/include"
)

add_executable(mps_presolve_structure
  mps_conflict_graph.cpp
  mps_lp_overlay.cpp
  mps_presolve_structure.cpp
)

target_include_directories(mps_presolve_structure
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../src"
    "${pslp_SOURCE_DIR}/include"
)

target_link_libraries(mps_presolve_structure
  PRIVATE
    mps_structure_analysis
    cuopt_static
    ${CUOPT_PRIVATE_CUDA_LIBS}
)

if(NOT DEFINED INSTALL_TARGET OR "${INSTALL_TARGET}" STREQUAL "")
  target_link_options(mps_presolve_structure PRIVATE -Wl,--enable-new-dtags)
endif()
