# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 3.30)
project(cuopt_async_lifetime LANGUAGES CXX)

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)
if(LLVM_VERSION_MAJOR LESS 23)
  message(FATAL_ERROR "The asynchronous lifetime checker requires Clang 23's LifetimeEnd callback")
endif()

add_library(CuoptAsyncLifetime MODULE AsyncLifetimeChecker.cpp)
target_compile_features(CuoptAsyncLifetime PRIVATE cxx_std_20)
target_include_directories(CuoptAsyncLifetime SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
separate_arguments(CUOPT_LLVM_DEFINITIONS NATIVE_COMMAND "${LLVM_DEFINITIONS}")
target_compile_options(CuoptAsyncLifetime PRIVATE ${CUOPT_LLVM_DEFINITIONS} -Wall -Wextra -Werror)
target_link_libraries(CuoptAsyncLifetime PRIVATE clang-cpp LLVM)
set_target_properties(CuoptAsyncLifetime PROPERTIES BUILD_RPATH "${LLVM_LIBRARY_DIRS}")

find_package(Python3 REQUIRED COMPONENTS Interpreter)
enable_testing()
add_test(NAME async_lifetime_contract_parser
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_contract_parser.py
    --clang ${LLVM_TOOLS_BINARY_DIR}/clang
    --plugin $<TARGET_FILE:CuoptAsyncLifetime>)
add_test(NAME async_lifetime_fixtures
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_tests.py
    --clang ${LLVM_TOOLS_BINARY_DIR}/clang
    --plugin $<TARGET_FILE:CuoptAsyncLifetime>)
get_filename_component(CUOPT_REPOSITORY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
add_test(NAME async_lifetime_driver
  COMMAND ${Python3_EXECUTABLE} -m unittest discover
    -s ${CUOPT_REPOSITORY_ROOT}/ci/tests -p test_async_lifetimes.py)
set_tests_properties(async_lifetime_contract_parser async_lifetime_fixtures async_lifetime_driver PROPERTIES TIMEOUT 120)
