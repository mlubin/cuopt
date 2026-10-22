/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "apis.hpp"

void view_outlives_input(cuopt::mathematical_optimization::io::data_model_view_t<int, float>& view)
{  // expected-warning {{asynchronous}}
  float coefficients[2] = {1.0f, 2.0f};
  view.set_objective_coefficients(coefficients, 2);
}

void synchronization_does_not_release_view(
  cuopt::mathematical_optimization::io::data_model_view_t<int, float>& view,
  cuda::stream_ref stream)
{  // expected-warning {{asynchronous}}
  float coefficients[2] = {1.0f, 2.0f};
  view.set_objective_coefficients(coefficients, 2);
  stream.sync();
}

void callback_outlives_user_data(cuopt::internals::base_solution_callback_t& callback)
{  // expected-warning {{asynchronous}}
  int user_data = 42;
  callback.set_user_data(&user_data);
}

void callback_outlives_registration(
  cuopt::mathematical_optimization::mip_solver_settings_t<int, float>& settings)
{  // expected-warning {{asynchronous}}
  cuopt::internals::base_solution_callback_t callback;
  settings.set_mip_callback(&callback, nullptr);
}

void view_and_input_share_scope()
{
  float coefficients[2] = {1.0f, 2.0f};
  cuopt::mathematical_optimization::io::data_model_view_t<int, float> view;
  view.set_objective_coefficients(coefficients, 2);
}

void view_allows_input_updates_before_use()
{
  float coefficients[2] = {1.0f, 2.0f};
  cuopt::mathematical_optimization::io::data_model_view_t<int, float> view;
  view.set_objective_coefficients(coefficients, 2);
  coefficients[0] = 3.0f;
}
