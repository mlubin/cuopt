# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

if(NOT SKIP_ROUTING_BUILD)
  add_subdirectory(routing)
endif()
if(NOT BUILD_LP_ONLY)
  add_subdirectory(mip)
endif()
