/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <dual_simplex/bound_flipping_ratio_test.hpp>

#include <math_optimization/tic_toc.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace cuopt::mathematical_optimization::simplex {

template <typename i_t, typename f_t>
i_t bound_flipping_ratio_test_t<i_t, f_t>::compute_breakpoints(std::vector<i_t>& indicies,
                                                               std::vector<f_t>& ratios,
                                                               std::vector<f_t>& harris_ratios)
{
  constexpr bool verbose = false;
  f_t pivot_tol          = settings_.pivot_tol;
  const f_t dual_tol     = settings_.dual_tol / 10;
  const i_t nz           = delta_z_indices_.size();

  // This path is normally a single pass. Relax the pivot threshold only when
  // every candidate is tiny, stopping before it becomes numerically meaningless.
  i_t count = 0;
  while (count == 0 && pivot_tol > f_t{1e-12}) {
    for (i_t h = 0; h < nz; ++h) {
      const i_t j = delta_z_indices_[h];
      const i_t k = nonbasic_mark_[j];
      if (vstatus_[j] == variable_status_t::NONBASIC_FIXED) { continue; }

      const f_t delta = delta_z_[j];
      const f_t zj    = z_[j];
      if (vstatus_[j] == variable_status_t::NONBASIC_LOWER && delta < -pivot_tol) {
        indicies[count]      = k;
        ratios[count]        = std::max(-zj / delta, f_t{0});
        harris_ratios[count] = std::max((-dual_tol - zj) / delta, f_t{0});
      } else if (vstatus_[j] == variable_status_t::NONBASIC_UPPER && delta > pivot_tol) {
        indicies[count]      = k;
        ratios[count]        = std::max(-zj / delta, f_t{0});
        harris_ratios[count] = std::max((dual_tol - zj) / delta, f_t{0});
      } else {
        continue;
      }
      if constexpr (verbose) { settings_.log.printf("ratios[%d] = %e\n", count, ratios[count]); }
      ++count;
    }
    work_estimate_ += 5 * nz + 5 * count;
    pivot_tol /= 10;
  }
  return count;
}

template <typename i_t, typename f_t>
i_t bound_flipping_ratio_test_t<i_t, f_t>::single_pass(i_t start,
                                                       i_t end,
                                                       const std::vector<i_t>& indicies,
                                                       const std::vector<f_t>& ratios,
                                                       f_t& step_length,
                                                       i_t& nonbasic_entering,
                                                       i_t& entering_index,
                                                       f_t& max_val)
{
  // Find the minimum ratio
  f_t min_val    = inf;
  entering_index = -1;
  i_t candidate  = -1;
  i_t k_idx      = -1;
  max_val        = 0.0;

  i_t min_found = 0;
  for (i_t k = start; k < end; ++k) {
    if (ratios[k] < min_val) {
      min_val   = ratios[k];
      candidate = indicies[k];
      k_idx     = k;
      min_found++;
    }
    if (ratios[k] > max_val) { max_val = ratios[k]; }
  }
  work_estimate_ += (end - start) + 2 * min_found;

  step_length       = min_val;
  nonbasic_entering = candidate;
  // this should be temporary, find root causes where the candidate is not filled
  if (nonbasic_entering == -1) { return RATIO_TEST_NUMERICAL_ISSUES; }
  const i_t j = entering_index = nonbasic_list_[nonbasic_entering];

  if (bounded_variables_[j]) { return k_idx; }
  return -1;  // we are done. do not increase the step-length further
}

template <typename i_t, typename f_t>
void bound_flipping_ratio_test_t<i_t, f_t>::determine_flips(f_t step_length,
                                                            i_t entering_index,
                                                            std::vector<i_t>& flip_indices)
{
  // The piecewise-linear model below assumes that a variable flips bounds as soon as
  // its reduced cost crosses zero. In practice, small changes between iterations can
  // make a reduced cost oscillate around zero, causing excessive bound flips and
  // cycling. We therefore flip only after the violation exceeds dual_tol / 10.
  // A bounded variable l_j <= x_j <= u_j contributes l_j*z_j to the dual objective
  // when z_j >= 0 and u_j*z_j when z_j < 0. If x_j = l_j and
  // -dual_tol/10 <= z_j < 0, the model uses u_j*z_j while the unflipped state uses
  // l_j*z_j. Their difference is (u_j - l_j)*|z_j|, bounded by
  // (u_j - l_j)*dual_tol/10. For multiple unflipped variables, the discrepancy is
  // bounded by sum_j (u_j - l_j)*dual_tol/10.
  const f_t flip_tol = settings_.dual_tol / 10;
  for (const i_t j : delta_z_indices_) {
    if (j == entering_index || !bounded_variables_[j]) { continue; }
    const f_t new_z = z_[j] + step_length * delta_z_[j];
    if ((vstatus_[j] == variable_status_t::NONBASIC_LOWER && new_z < -flip_tol) ||
        (vstatus_[j] == variable_status_t::NONBASIC_UPPER && new_z > flip_tol)) {
      flip_indices.push_back(j);
    }
  }
  work_estimate_ += 5 * delta_z_indices_.size() + flip_indices.size();
}

template <typename i_t, typename f_t>
i_t bound_flipping_ratio_test_t<i_t, f_t>::compute_step_length(f_t& step_length,
                                                               i_t& nonbasic_entering,
                                                               std::vector<i_t>& flip_indices)
{
  const i_t nz           = delta_z_indices_.size();
  constexpr bool verbose = false;
  flip_indices.clear();

  // Compute the initial set of breakpoints
  std::vector<i_t> indicies(nz);
  std::vector<f_t> ratios(nz);
  std::vector<f_t> harris_ratios(nz);
  work_estimate_ += 3 * nz;
  double t0           = tic();
  i_t num_breakpoints = compute_breakpoints(indicies, ratios, harris_ratios);
  time_compute_breakpoints_ += toc(t0);
  num_breakpoints_ = num_breakpoints;
  // Count zero ratios
  num_harris_zero_ = 0;
  num_exact_zero_  = 0;
  for (i_t k = 0; k < num_breakpoints; k++) {
    if (harris_ratios[k] == 0.0) num_harris_zero_++;
    if (ratios[k] == 0.0) num_exact_zero_++;
  }
  work_estimate_ += 2 * num_breakpoints;
  if constexpr (verbose) { settings_.log.printf("Initial breakpoints %d\n", num_breakpoints); }
  if (num_breakpoints == 0) {
    nonbasic_entering = -1;
    return RATIO_TEST_NO_ENTERING_VARIABLE;
  }

  f_t slope          = slope_;
  nonbasic_entering  = -1;
  i_t entering_index = RATIO_TEST_NO_ENTERING_VARIABLE;
  f_t max_step_length;

  t0        = tic();
  i_t k_idx = single_pass(0,
                          num_breakpoints,
                          indicies,
                          harris_ratios,
                          step_length,
                          nonbasic_entering,
                          entering_index,
                          max_step_length);
  time_single_pass_ += toc(t0);
  if (k_idx == RATIO_TEST_NUMERICAL_ISSUES) { return RATIO_TEST_NUMERICAL_ISSUES; }
  // The variable selected by single_pass is guaranteed to be in the first bucket: it
  // defines the minimum Harris ratio, and its exact ratio is no greater than its Harris
  // ratio. Its slope contribution is therefore applied by the bucket pass below.
  bool continue_search = k_idx >= 0 && num_breakpoints > 1;
  if (!continue_search) {
    if constexpr (verbose) {
      settings_.log.printf(
        "BFRT stopping. No bound flips. Step length %e Nonbasic entering %d Entering %d pivot %e\n",
        step_length,
        nonbasic_entering,
        entering_index,
        std::abs(delta_z_[entering_index]));
    }
    num_buckets_used_   = 0;
    step_length_result_ = step_length;
    determine_flips(step_length, entering_index, flip_indices);
    return entering_index;
  }

  if constexpr (verbose) {
    settings_.log.printf(
      "Continuing past initial step length %e entering index %d nonbasic entering %d slope %e\n",
      step_length,
      entering_index,
      nonbasic_entering,
      slope);
  }

  // This code combines bound flipping with Harris' ratio test. We want to take
  // the longest useful dual step while:
  //  1) remaining dual feasible up to a small tolerance,
  //  2) increasing the dual objective, and
  //  3) selecting a numerically stable pivot.
  //
  // Along a dual step alpha, each candidate reduced cost is
  //
  //   z_j(alpha) = z_j + alpha * delta_z_j.
  //
  // A nonbasic at its lower bound is dual feasible when z_j(alpha) >= 0, and
  // one at its upper bound is dual feasible when z_j(alpha) <= 0. Its exact
  // breakpoint is therefore
  //
  //   alpha_j = -z_j / delta_z_j.
  //
  // A boxed variable may cross this breakpoint by flipping to its opposite
  // bound. A one-sided variable cannot, so its breakpoint limits the step.
  //
  // The dual objective as a function of alpha is piecewise linear and concave.
  // When a boxed variable crosses its breakpoint, the slope decreases by
  //
  //   |delta_z_j| * (upper_j - lower_j).
  //
  // We may keep increasing alpha through boxed breakpoints while the cumulative
  // slope remains nonnegative. The first one-sided breakpoint, or the first
  // group of boxed breakpoints that makes the slope negative, ends the useful
  // range.
  //
  // Harris' ratio test relaxes dual feasibility by eps = dual_tol / 10:
  //
  //   z_j(alpha) >= -eps  for a variable at its lower bound,
  //   z_j(alpha) <=  eps  for a variable at its upper bound.
  //
  // These inequalities define the Harris ratios. They group nearly coincident
  // breakpoints so that we can prefer a larger |delta_z_j| instead of being
  // forced to use a weak pivot solely because its exact ratio is slightly
  // smaller.
  //
  // Sorting all breakpoints would cost O(num_breakpoints * log(num_breakpoints)).
  // The coarse pass instead expands its threshold geometrically to limit the
  // candidate set. This pass does not define the buckets. Bucket zero contains
  // every exact ratio no greater than the minimum Harris ratio. Each subsequent
  // bucket advances to the minimum Harris ratio among the remaining candidates
  // and collects every exact ratio up to that boundary. This continues until
  // the slope turns negative or a one-sided variable limits the step.
  //
  // Finally, search the admissible buckets from last to first. Within a bucket,
  // choose the largest |delta_z_j| strictly above
  //
  //   max(pivot_tol, min(0.1 * max_pivot, 1)).
  //
  // Break equal-pivot ties using the larger exact ratio. If no bucket contains
  // an acceptable pivot, retain the initial minimum-Harris-ratio candidate.
  // The final bound-flip set applies the dual_tol / 10 margin separately.

  // Use a coarse filter to find candidates
  f_t minimum_harris_ratio = step_length;
  f_t coarse_threshold     = (minimum_harris_ratio > 0.0)
                               ? std::min(10.0 * minimum_harris_ratio, max_step_length)
                               : max_step_length;
  f_t total_slope          = slope;
  bool found_unbounded     = false;
  std::vector<i_t> candidates(num_breakpoints);
  std::iota(candidates.begin(), candidates.end(), 0);
  work_estimate_ += 2 * num_breakpoints;
  i_t scan_start     = 0;
  i_t num_candidates = 0;

  // This is O( log10(max_step_length/min_step_length) * num_breakpoints)
  t0 = tic();
  while (total_slope >= 0.0 && coarse_threshold <= max_step_length &&
         scan_start < num_breakpoints && !found_unbounded) {
    for (i_t h = scan_start; h < num_breakpoints; ++h) {
      const i_t k = candidates[h];
      if (ratios[k] <= coarse_threshold) {
        // Candidate is less than coarse threshold, move it to the front of the candidate list
        std::swap(candidates[h], candidates[num_candidates]);
        num_candidates++;
        const i_t j = nonbasic_list_[indicies[k]];
        if (!bounded_variables_[j]) {
          found_unbounded = true;
        } else {
          total_slope -= std::abs(delta_z_[j]) * (upper_[j] - lower_[j]);
        }
      }
    }
    work_estimate_ += 2 * (num_breakpoints - scan_start) + 10 * (num_candidates - scan_start);
    scan_start = num_candidates;
    coarse_threshold *= 10.0;
  }
  time_coarse_filter_ += toc(t0);

  candidates.resize(num_candidates);

  // Check for variables with one sided bounds. These define the maximum step length.
  if (found_unbounded) {
    for (i_t h = 0; h < num_candidates; h++) {
      const i_t k = candidates[h];
      const i_t j = nonbasic_list_[indicies[k]];
      if (!bounded_variables_[j]) { max_step_length = std::min(max_step_length, harris_ratios[k]); }
    }
    work_estimate_ += 5 * num_candidates;

    // Remove candidates that are greater than the maximum step length
    const i_t candidates_before_removal = candidates.size();
    for (i_t h = candidates_before_removal - 1; h >= 0; h--) {
      const i_t k     = candidates[h];
      const f_t ratio = ratios[k];
      if (ratio > max_step_length) {
        // Swap with the last candidate and remove
        candidates[h] = candidates.back();
        candidates.pop_back();
      }
    }
    work_estimate_ +=
      2 * candidates_before_removal + 2 * (candidates_before_removal - candidates.size());
    num_candidates = candidates.size();
  }

  // Use a bucket sort to partition candidates into buckets by successive Harris breakpoints
  // bucket_start[k] = index in candidates[] where bucket k starts
  // Bucket k contains candidates[bucket_start[k]] .. candidates[bucket_start[k+1] - 1]
  f_t threshold   = minimum_harris_ratio;
  i_t num_buckets = 0;
  std::vector<i_t> bucket_start(num_candidates + 1, 0);
  f_t cumulative_slope = slope;
  scan_start           = 0;
  work_estimate_ += num_candidates + 1;

  // This is O(num_buckets * num_candidates)
  i_t slope_breaker_k = -1;  // the candidate k that made slope go negative
  t0                  = tic();
  while (cumulative_slope >= 0.0 && scan_start < num_candidates && threshold <= max_step_length) {
    f_t next_threshold = inf;
    i_t write          = scan_start;

    for (i_t h = scan_start; h < num_candidates; h++) {
      const i_t k     = candidates[h];
      const f_t ratio = ratios[k];

      if (ratio <= threshold) {
        const i_t j = nonbasic_list_[indicies[k]];
        if (bounded_variables_[j]) {
          cumulative_slope -= std::abs(delta_z_[j]) * (upper_[j] - lower_[j]);
          if (cumulative_slope < 0.0 && slope_breaker_k < 0) { slope_breaker_k = k; }
        }
        std::swap(candidates[h], candidates[write]);
        write++;
      } else {
        next_threshold = std::min(next_threshold, harris_ratios[k]);
      }
    }
    work_estimate_ += 3 * (num_candidates - scan_start) + 9 * (write - scan_start);

    bucket_start[++num_buckets] = write;
    if (write == scan_start) break;  // No progress — prevent infinite loop
    scan_start = write;
    threshold  = next_threshold;

    if (cumulative_slope < 0.0) break;
  }
  time_bucket_sort_ += toc(t0);
  bucket0_size_ = (num_buckets > 0) ? bucket_start[1] : 0;

  // Compute the maximum pivot
  // This is O(num_candidates)
  f_t max_pivot = 0.0;
  for (i_t h = 0; h < bucket_start[num_buckets]; h++) {
    const i_t k     = candidates[h];
    const i_t j     = nonbasic_list_[indicies[k]];
    const f_t pivot = std::abs(delta_z_[j]);
    if (pivot > max_pivot) { max_pivot = pivot; }
  }
  work_estimate_ += 4 * bucket_start[num_buckets];

  // Search the last admissible Harris group first and choose its strongest
  // pivot. Ratio-first selection causes weak, zero-length pivots under heavy
  // dual degeneracy.
  const f_t pivot_threshold =
    std::max(settings_.pivot_tol, std::min(f_t{0.1} * max_pivot, f_t{1.0}));
  i_t entering_k      = -1;
  i_t selected_bucket = -1;
  for (i_t b = num_buckets - 1; b >= 0; --b) {
    f_t best_pivot = -1.0;
    f_t best_ratio = -1.0;
    for (i_t h = bucket_start[b]; h < bucket_start[b + 1]; ++h) {
      const i_t k     = candidates[h];
      const i_t j     = nonbasic_list_[indicies[k]];
      const f_t pivot = std::abs(delta_z_[j]);
      if (pivot > pivot_threshold &&
          (pivot > best_pivot || (pivot == best_pivot && ratios[k] > best_ratio))) {
        best_pivot = pivot;
        best_ratio = ratios[k];
        entering_k = k;
      }
    }
    work_estimate_ += 2 + 5 * (bucket_start[b + 1] - bucket_start[b]);
    if (entering_k >= 0) {
      selected_bucket = b;
      break;
    }
  }

  // Step = entering variable's breakpoint ratio
  num_buckets_used_ = num_buckets;
  if (entering_k < 0) {
    // Fallback to single_pass result
    used_fallback_             = true;
    bucket_selected_           = -1;
    step_length_result_        = step_length;
    selected_is_slope_breaker_ = false;
    determine_flips(step_length, entering_index, flip_indices);
    return entering_index;
  }
  step_length       = ratios[entering_k];
  nonbasic_entering = indicies[entering_k];
  entering_index    = nonbasic_list_[nonbasic_entering];

  selected_is_slope_breaker_ = entering_k == slope_breaker_k;
  used_fallback_             = false;
  bucket_selected_           = selected_bucket;
  step_length_result_        = step_length;
  determine_flips(step_length, entering_index, flip_indices);

  return entering_index;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template class bound_flipping_ratio_test_t<int, double>;

#endif

}  // namespace cuopt::mathematical_optimization::simplex
