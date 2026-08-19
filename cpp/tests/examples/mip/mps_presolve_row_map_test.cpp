/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/mathematical_optimization/optimization_problem_interface.hpp>
#include <mip_heuristics/presolve/third_party_presolve.hpp>
#include <utilities/inline_lp_test_utils.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <unordered_set>

namespace cuopt::mathematical_optimization::examples {
namespace {

TEST(MpsPresolveRowMap, FollowsReducedOrderAndClearsStaleMappings)
{
  auto model = cuopt::test::parse_inline_lp(R"LP(
Minimize
  obj: x + y
Subject To
  c0: x + y = 1
  c1: 2 x + 2 y = 2
  c2: x - y <= 0
Binaries
  x
  y
End
)LP");

  mip::third_party_presolve_t<int, double> presolver;
  presolver.set_reduction_allowlist(std::unordered_set<std::string>{"parallelrows"});
  auto result = presolver.apply_presolve_from_mps_data(
    model, problem_category_t::MIP, presolver_t::Papilo, false, 1e-6, 1e-12, 20.0);

  ASSERT_EQ(result.status, mip::third_party_presolve_status_t::REDUCED);
  const auto& reduced_to_original = presolver.get_reduced_to_original_row_map();
  const auto& original_to_reduced = presolver.get_original_to_reduced_row_map();
  ASSERT_EQ(reduced_to_original.size(), result.reduced_problem.get_n_constraints());
  ASSERT_EQ(original_to_reduced.size(), model.get_n_constraints());
  EXPECT_EQ(
    static_cast<int>(original_to_reduced[0] < 0) + static_cast<int>(original_to_reduced[1] < 0), 1);
  EXPECT_GE(original_to_reduced[2], 0);
  for (std::size_t reduced_row = 0; reduced_row < reduced_to_original.size(); ++reduced_row) {
    ASSERT_GE(reduced_to_original[reduced_row], 0);
    EXPECT_EQ(original_to_reduced[static_cast<std::size_t>(reduced_to_original[reduced_row])],
              static_cast<int>(reduced_row));
  }

  auto infeasible_model = cuopt::test::parse_inline_lp(R"LP(
Minimize
  obj: x
Subject To
  c0: x = 0
  c1: 2 x = 2
Binaries
  x
End
)LP");
  result                = presolver.apply_presolve_from_mps_data(
    infeasible_model, problem_category_t::MIP, presolver_t::Papilo, false, 1e-6, 1e-12, 20.0);
  EXPECT_EQ(result.status, mip::third_party_presolve_status_t::INFEASIBLE);
  EXPECT_TRUE(presolver.get_reduced_to_original_row_map().empty());
  EXPECT_TRUE(presolver.get_original_to_reduced_row_map().empty());
}

}  // namespace
}  // namespace cuopt::mathematical_optimization::examples
