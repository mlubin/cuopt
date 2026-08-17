# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------------------------------
# Standalone hierarchical PDLP runner for a two-objective LP hierarchy.
#
# Inputs are two MPS models over the SAME constraint matrix, bounds, and
# variable/row ordering, differing only in objective: PRIMARY carries the
# primary objective c1; FULL carries the blended objective c1 + c2. The runner
# minimizes the effective secondary objective c2 over the (approximately)
# primary-optimal set:
#
#   reference solve/reuse → passes of
#     DESCEND — one Stable3 PDLP solve on the shifted objective
#               c1 − Aᵀy_ref + c2 (warm primal from the incumbent, zero dual,
#               fixed iteration budget). Pushes the secondary down along the
#               reference face; not expected to certify — its endpoint is an
#               intermediate point that only seeds the projection.
#     PROJECT — one Stable3 PDLP solve on the primary objective, warm from
#               the descent endpoint with a reference-dual start, at the gate
#               tolerance: an approximate projection back onto the
#               primary-optimal set. If it hits its iteration limit, ONE
#               fresh restart from its endpoint must certify.
#   → promotion gate: the candidate must be PDLP-Optimal (the certificate
#     gate) AND improve the independently audited secondary; a positive
#     improvement below the minimum threshold promotes and stops.
#
# Every phase is a stock Stable3 solve at the shared optimality tolerance;
# only the objective, warm start, and iteration budget vary between phases.
#
# Outputs (in OUTPUT_DIR): progress.csv (one independently audited row per
# phase), summary.txt, final_primal.f64, per-phase solver logs, the solved
# reference pair (reference_primal.f64 / reference_dual.f64), and optionally
# per-phase endpoint vectors (--dump-phase-vectors). Vector files are raw
# little-endian float64 arrays.
#
# Usage examples:
#
#   # From scratch (defaults):
#   python run_hierarchical_pdlp.py primary.mps.gz full.mps.gz OUT_DIR
#
#   # Reusing a previously dumped reference (skips the reference solve):
#   python run_hierarchical_pdlp.py primary.mps.gz full.mps.gz OUT_DIR \
#     --initial-reference-primal REF/reference_primal.f64 \
#     --initial-reference-dual   REF/reference_dual.f64
#
#   # Reference-only (produce a reusable reference, no passes):
#   python run_hierarchical_pdlp.py primary.mps.gz full.mps.gz OUT_DIR --max-passes 0
#
#   # Deterministic CPU-only checks of the pure functions and CLI:
#   python run_hierarchical_pdlp.py --self-test
# -----------------------------------------------------------------------------

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse

I_T_MAX = 2**31 - 1  # solver i_t is int32
LL_MAX = 2**63 - 1

# =============================================================================
# Small utilities
# =============================================================================


def fmt_double(v: float) -> str:
    # Up to 17 significant digits (round-trip precision for float64).
    return f"{v:.17g}"


def csv_field(s: str) -> str:
    if not any(c in s for c in ',"\r\n'):
        return s
    return '"' + s.replace('"', '""') + '"'


def all_finite(v: np.ndarray) -> bool:
    return bool(np.isfinite(v).all())


def checked_add_iterations(a: int, b: int) -> int:
    if a < 0 or b < 0:
        raise ValueError("checked_add_iterations: negative operand")
    s = a + b
    if s > LL_MAX:
        raise RuntimeError("cumulative iteration count overflow")
    return s


def checked_add_time(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError("checked_add_time: non-finite operand")
    s = a + b
    if not math.isfinite(s):
        raise RuntimeError("cumulative solve time overflow / non-finite")
    return s


def require_nonempty_file(path: str) -> None:
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        raise RuntimeError("expected a non-empty native log file: " + path)


# =============================================================================
# Raw f64 vector files (dumped phase endpoints / warm starts)
# =============================================================================


def write_f64(path: str, v: np.ndarray) -> None:
    np.ascontiguousarray(v, dtype=np.float64).tofile(path)


def read_f64(path: str, expected_size: int, what: str) -> np.ndarray:
    try:
        nbytes = os.path.getsize(path)
    except OSError:
        raise RuntimeError(f"{what}: cannot stat {path}")
    if nbytes != expected_size * 8:
        raise RuntimeError(
            f"{what}: {path} has {nbytes} bytes, expected {expected_size * 8}"
        )
    v = np.fromfile(path, dtype=np.float64)
    if v.size != expected_size:
        raise RuntimeError(f"{what}: failed while reading {path}")
    if not all_finite(v):
        raise RuntimeError(f"{what}: non-finite value in {path}")
    return v


PHASE_DESCENT = "descent"
PHASE_PROJECTION = "projection"
PHASE_PROJECTION_RESTART = "projection_restart"


# Owns dimensions, CSR arrays, row/variable bounds, effective objective vectors
# and offsets, the independently evaluated primary reference value, and the
# certified reference primal/dual vectors.
@dataclass
class HierarchyState:
    n_vars: int = 0
    n_cons: int = 0
    nnz: int = 0

    A_values: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    A_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    A_offsets: np.ndarray = field(default_factory=lambda: np.array([0], dtype=np.int32))

    rhs: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    row_lower: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    row_upper: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    var_lower: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    var_upper: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    c1: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    c2: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    offset1: float = 0.0
    offset2: float = 0.0

    primary_reference: float = 0.0
    reference_primal: Optional[np.ndarray] = None
    reference_dual: Optional[np.ndarray] = None

    _A_csr: Optional[scipy.sparse.csr_matrix] = None

    def A(self) -> scipy.sparse.csr_matrix:
        if self._A_csr is None:
            self._A_csr = scipy.sparse.csr_matrix(
                (self.A_values, self.A_indices, self.A_offsets),
                shape=(self.n_cons, self.n_vars),
            )
        return self._A_csr


# =============================================================================
# Pure hierarchy functions (exercised by --self-test)
# =============================================================================


@dataclass
class EffectiveObjectives:
    c1: np.ndarray
    c2: np.ndarray
    offset1: float
    offset2: float


# c1 = primary_scale * primary_coeffs; c_full = full_scale * full_coeffs;
# c2 = c_full - c1. Offsets analogously. Rejects any non-finite intermediate.
def compute_effective_objectives(
    primary_coeffs, primary_scale, primary_offset, full_coeffs, full_scale, full_offset
) -> EffectiveObjectives:
    primary_coeffs = np.asarray(primary_coeffs, dtype=np.float64)
    full_coeffs = np.asarray(full_coeffs, dtype=np.float64)
    if primary_coeffs.size != full_coeffs.size:
        raise ValueError("compute_effective_objectives: objective size mismatch")
    c1 = primary_scale * primary_coeffs
    cf = full_scale * full_coeffs
    c2 = cf - c1
    if not (all_finite(c1) and all_finite(cf) and all_finite(c2)):
        raise ValueError("compute_effective_objectives: non-finite coefficient")
    offset1 = primary_scale * primary_offset
    offset_full = full_scale * full_offset
    offset2 = offset_full - offset1
    if not (
        math.isfinite(offset1) and math.isfinite(offset_full) and math.isfinite(offset2)
    ):
        raise ValueError("compute_effective_objectives: non-finite offset")
    return EffectiveObjectives(c1=c1, c2=c2, offset1=offset1, offset2=offset2)


# c_shift = c1 - A^T * reference_dual + c2.
def build_shifted_objective(s: HierarchyState) -> np.ndarray:
    if s.c1.size != s.n_vars or s.c2.size != s.n_vars:
        raise ValueError("build_shifted_objective: objective size mismatch")
    if s.reference_dual is None or s.reference_dual.size != s.n_cons:
        raise ValueError("build_shifted_objective: reference dual size mismatch")
    aty = s.A().T.dot(s.reference_dual)
    if not all_finite(aty):
        raise ValueError("build_shifted_objective: non-finite A^T*y product")
    c_shift = s.c1 - aty + s.c2
    if not all_finite(c_shift):
        raise ValueError("build_shifted_objective: non-finite result")
    return c_shift


@dataclass
class EvaluationResult:
    primary: float = 0.0
    secondary: float = 0.0
    full: float = 0.0
    # Deviation from primary_reference — diagnostic only, never a gate (an
    # epsilon-optimality certificate leaves objective uncertainty well above
    # these deviations at scale, so value comparisons below that noise floor
    # are meaningless).
    primary_deviation: float = 0.0
    primary_abs_deviation: float = 0.0
    row_l2: float = 0.0
    row_linf: float = 0.0
    bound_l2: float = 0.0
    bound_linf: float = 0.0
    # Dual-corrected primary (reported when a reference dual is available):
    # primary minus the infeasibility-financed component as priced by y.
    has_dual_correction: bool = False
    dual_correction: float = 0.0
    corrected_primary: float = 0.0
    corrected_primary_deviation: float = 0.0


# Independent endpoint evaluation from the original model arrays.
def evaluate_point(s: HierarchyState, primary_reference: float, x: np.ndarray) -> EvaluationResult:
    x = np.asarray(x, dtype=np.float64)
    if x.size != s.n_vars:
        raise ValueError("evaluate_point: primal length mismatch")
    if not all_finite(x):
        raise ValueError("evaluate_point: non-finite primal")

    primary = s.offset1 + float(np.dot(s.c1, x))
    secondary = s.offset2 + float(np.dot(s.c2, x))
    full = primary + secondary

    a = s.A().dot(x)
    below = s.row_lower - a
    above = a - s.row_upper
    v = np.maximum(0.0, np.maximum(below, above))
    row_l2 = float(np.sqrt(np.dot(v, v)))
    row_linf = float(v.max()) if v.size else 0.0

    have_ref_dual = s.reference_dual is not None and s.reference_dual.size == s.n_cons
    if have_ref_dual:
        # Signed violation: negative below lo, positive above hi, zero inside.
        sv = np.where(a < s.row_lower, a - s.row_lower,
                      np.where(a > s.row_upper, a - s.row_upper, 0.0))
        dual_corr = float(np.dot(s.reference_dual, sv))

    bv = np.maximum(0.0, np.maximum(s.var_lower - x, x - s.var_upper))
    bound_l2 = float(np.sqrt(np.dot(bv, bv)))
    bound_linf = float(bv.max()) if bv.size else 0.0

    e = EvaluationResult(
        primary=primary,
        secondary=secondary,
        full=full,
        primary_deviation=primary - primary_reference,
        primary_abs_deviation=abs(primary - primary_reference),
        row_l2=row_l2,
        row_linf=row_linf,
        bound_l2=bound_l2,
        bound_linf=bound_linf,
    )
    if have_ref_dual:
        e.has_dual_correction = True
        e.dual_correction = dual_corr
        e.corrected_primary = primary - dual_corr
        e.corrected_primary_deviation = e.corrected_primary - primary_reference
        for m in (e.dual_correction, e.corrected_primary, e.corrected_primary_deviation):
            if not math.isfinite(m):
                raise ValueError("evaluate_point: non-finite metric")

    for m in (e.primary, e.secondary, e.full, e.primary_deviation,
              e.row_l2, e.row_linf, e.bound_l2, e.bound_linf):
        if not math.isfinite(m):
            raise ValueError("evaluate_point: non-finite metric")
    return e


# Clamp a warm-start primal into the variable box, reporting the correction.
# Solver endpoints can sit epsilon-outside their bounds (~1e-10 observed on
# descent endpoints), and an out-of-bounds vector is not a valid initial
# solution, so every warm start is clamped at every handoff. Audits and
# dumped endpoints stay raw; only warm starts are clamped.
def clamp_warm_start(s: HierarchyState, v: np.ndarray, what: str) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if v.size != s.n_vars:
        raise ValueError("clamp_warm_start: primal length mismatch")
    c = np.minimum(np.maximum(v, s.var_lower), s.var_upper)
    changed = c != v
    n_clamped = int(changed.sum())
    if n_clamped > 0:
        max_clamp = float(np.abs(c[changed] - v[changed]).max())
        print(
            f"note: clamped {n_clamped} {what} warm-start entries into variable "
            f"bounds (max correction {fmt_double(max_clamp)})",
            file=sys.stderr,
            flush=True,
        )
    return c


@dataclass
class PromotionDecision:
    promoted: bool = False
    stop: bool = False
    reason: str = ""
    relative_improvement: float = 0.0
    has_relative: bool = False


# Promotion gates in order (the pass candidate = the projection endpoint, or
# the projection_restart endpoint if the restart ran). Primary admissibility
# IS PDLP's Optimal certificate on the primary-objective solve at the gate
# tolerance — so gate 1 covers it and there is no reference-value comparison.
#   1. not Optimal                       -> projection_not_certified
#   2. candidate_secondary >= incumbent  -> secondary_not_improved
#   3. otherwise promote; improvement < threshold -> promote + stop, else continue
def decide_promotion(
    optimal: bool,
    candidate_secondary: float,
    incumbent_secondary: float,
    minimum_relative_improvement: float,
) -> PromotionDecision:
    d = PromotionDecision()
    if not optimal:
        d.reason = "projection_not_certified"
        return d
    if candidate_secondary >= incumbent_secondary:
        d.reason = "secondary_not_improved"
        return d
    improvement = incumbent_secondary - candidate_secondary
    denom = max(1.0, abs(incumbent_secondary))
    rel = improvement / denom
    if not (math.isfinite(improvement) and math.isfinite(rel)):
        raise ValueError("decide_promotion: non-finite improvement")
    d.promoted = True
    d.has_relative = True
    d.relative_improvement = rel
    if rel < minimum_relative_improvement:
        d.stop = True
        d.reason = "secondary_improvement_below_threshold"
    else:
        d.reason = "promoted"
    return d


# =============================================================================
# Input model validation and objective construction
# =============================================================================


def _quadratic_objective_present(m) -> bool:
    try:
        return np.asarray(m.get_quadratic_objective_values()).size > 0
    except Exception:
        return False


def _quadratic_constraints_present(m) -> bool:
    try:
        qc = m.get_quadratic_constraints()
        return qc is not None and len(qc) > 0
    except Exception:
        return False


# Validates that the model is a continuous minimization LP whose rows are all
# finite equalities (lower == upper). Raises ValueError otherwise.
def validate_model(m, label: str) -> None:
    def fail(why: str):
        raise ValueError(f"model '{label}' invalid: {why}")

    if m.get_sense():
        fail("objective must be minimization")
    if _quadratic_objective_present(m):
        fail("quadratic objective not supported")
    if _quadratic_constraints_present(m):
        fail("quadratic constraints not supported")

    A_values = np.asarray(m.get_constraint_matrix_values())
    A_indices = np.asarray(m.get_constraint_matrix_indices())
    A_offsets = np.asarray(m.get_constraint_matrix_offsets())
    c = np.asarray(m.get_objective_coefficients())
    var_lo = np.asarray(m.get_variable_lower_bounds())
    var_hi = np.asarray(m.get_variable_upper_bounds())
    var_types = np.asarray(m.get_variable_types())
    rhs = np.asarray(m.get_constraint_bounds())
    row_lo = np.asarray(m.get_constraint_lower_bounds())
    row_hi = np.asarray(m.get_constraint_upper_bounds())
    var_names = list(m.get_variable_names())
    row_names = list(m.get_row_names())

    n_vars = c.size
    n_cons = A_offsets.size - 1
    nnz = A_values.size
    if n_cons < 0:
        fail("negative dimensions")

    if A_indices.size != nnz:
        fail("matrix index array size inconsistent")
    if var_lo.size != n_vars:
        fail("variable lower bound size inconsistent")
    if var_hi.size != n_vars:
        fail("variable upper bound size inconsistent")
    if var_types.size != n_vars:
        fail("variable type array size inconsistent")
    if rhs.size != n_cons:
        fail("rhs array size inconsistent")
    if row_lo.size != n_cons:
        fail("row lower bound size inconsistent")
    if row_hi.size != n_cons:
        fail("row upper bound size inconsistent")
    if var_names and len(var_names) != n_vars:
        fail("variable name array size inconsistent")
    if row_names and len(row_names) != n_cons:
        fail("row name array size inconsistent")

    # CSR structural validity.
    if n_cons > 0 and A_offsets[0] != 0:
        fail("CSR offsets must begin at zero")
    if A_offsets.size == 0 or A_offsets[-1] != nnz:
        fail("CSR offsets must end at nnz")
    if np.any(np.diff(A_offsets) < 0):
        fail("CSR offsets must be nondecreasing")
    if nnz > 0 and (A_indices.min() < 0 or A_indices.max() >= n_vars):
        fail("CSR column index out of range")

    # Finiteness.
    if not all_finite(A_values):
        fail("non-finite matrix coefficient")
    if not all_finite(c):
        fail("non-finite objective coefficient")
    if not math.isfinite(m.get_objective_scaling_factor()):
        fail("non-finite objective scaling factor")
    if not math.isfinite(m.get_objective_offset()):
        fail("non-finite objective offset")

    # Variables: continuous, no NaN bounds, lower <= upper.
    vt = var_types.astype("U1")
    if not np.all(vt == "C"):
        fail("all variables must be continuous")
    if np.isnan(var_lo).any() or np.isnan(var_hi).any():
        fail("variable bound is NaN")
    if np.any(var_lo > var_hi):
        fail("variable lower bound exceeds upper bound")

    # Rows: every row a finite equality.
    if np.isnan(row_lo).any() or np.isnan(row_hi).any() or np.isnan(rhs).any():
        fail("row bound is NaN")
    if not (all_finite(row_lo) and all_finite(row_hi)):
        fail("row bound is not finite (inequality / free row not supported)")
    if np.any(row_lo != row_hi):
        fail("row is not an equality (lower != upper)")


def _require_equal_arrays(a, b, what: str) -> None:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size != b.size:
        raise ValueError(f"model mismatch: {what} size differs")
    if a.size and not np.array_equal(a, b):
        raise ValueError(f"model mismatch: {what} differs")


def _require_equal_name_lists(a, b, what: str) -> None:
    if len(a) != len(b):
        raise ValueError(f"model mismatch: {what} size differs")
    if list(a) != list(b):
        raise ValueError(f"model mismatch: {what} differs")


# The two models must describe exactly the same feasible region and ordering;
# only objective coefficients / scaling / offset / objective name may differ.
def validate_match(primary, full) -> None:
    _require_equal_arrays(
        primary.get_constraint_matrix_offsets(),
        full.get_constraint_matrix_offsets(),
        "CSR offsets",
    )
    _require_equal_arrays(
        primary.get_constraint_matrix_indices(),
        full.get_constraint_matrix_indices(),
        "CSR indices",
    )
    _require_equal_arrays(
        primary.get_constraint_matrix_values(),
        full.get_constraint_matrix_values(),
        "CSR values",
    )
    _require_equal_arrays(
        primary.get_constraint_lower_bounds(),
        full.get_constraint_lower_bounds(),
        "row lower bounds",
    )
    _require_equal_arrays(
        primary.get_constraint_upper_bounds(),
        full.get_constraint_upper_bounds(),
        "row upper bounds",
    )
    _require_equal_arrays(primary.get_constraint_bounds(), full.get_constraint_bounds(), "rhs")
    _require_equal_arrays(
        primary.get_variable_lower_bounds(),
        full.get_variable_lower_bounds(),
        "variable lower bounds",
    )
    _require_equal_arrays(
        primary.get_variable_upper_bounds(),
        full.get_variable_upper_bounds(),
        "variable upper bounds",
    )
    _require_equal_arrays(primary.get_variable_types(), full.get_variable_types(), "variable types")
    _require_equal_arrays(
        primary.get_ascii_row_types(), full.get_ascii_row_types(), "row types"
    )

    pv, fv = list(primary.get_variable_names()), list(full.get_variable_names())
    pr, fr = list(primary.get_row_names()), list(full.get_row_names())
    if (len(pv) == 0) != (len(fv) == 0):
        raise ValueError("model mismatch: variable name array emptiness differs")
    if (len(pr) == 0) != (len(fr) == 0):
        raise ValueError("model mismatch: row name array emptiness differs")
    _require_equal_name_lists(pv, fv, "variable names")
    _require_equal_name_lists(pr, fr, "row names")


# Build the hierarchy state (own copies of the shared structure) and derive the
# effective primary/secondary objectives from the two models.
def build_state(primary, full) -> HierarchyState:
    s = HierarchyState()
    s.A_values = np.array(primary.get_constraint_matrix_values(), dtype=np.float64)
    s.A_indices = np.array(primary.get_constraint_matrix_indices(), dtype=np.int32)
    s.A_offsets = np.array(primary.get_constraint_matrix_offsets(), dtype=np.int32)
    s.rhs = np.array(primary.get_constraint_bounds(), dtype=np.float64)
    s.row_lower = np.array(primary.get_constraint_lower_bounds(), dtype=np.float64)
    s.row_upper = np.array(primary.get_constraint_upper_bounds(), dtype=np.float64)
    s.var_lower = np.array(primary.get_variable_lower_bounds(), dtype=np.float64)
    s.var_upper = np.array(primary.get_variable_upper_bounds(), dtype=np.float64)

    s.n_vars = s.var_lower.size
    s.n_cons = s.A_offsets.size - 1
    s.nnz = s.A_values.size

    eff = compute_effective_objectives(
        primary.get_objective_coefficients(),
        primary.get_objective_scaling_factor(),
        primary.get_objective_offset(),
        full.get_objective_coefficients(),
        full.get_objective_scaling_factor(),
        full.get_objective_offset(),
    )
    s.c1 = eff.c1
    s.c2 = eff.c2
    s.offset1 = eff.offset1
    s.offset2 = eff.offset2
    return s


# =============================================================================
# CLI: parsing and options
# =============================================================================


def parse_ll(s: str):
    try:
        return int(s, 10)
    except ValueError:
        return None


def parse_dbl(s: str):
    try:
        return float(s)
    except ValueError:
        return None


# Options for the two-phase sequence. Defaults: descent 8k (a fixed budget —
# larger budgets over-descend and break projection), projection 60k with a
# 20k conditional fresh restart, every tolerance 1e-4.
@dataclass
class RunOptions:
    primary_path: str = ""
    full_path: str = ""
    output_dir: str = ""

    reference_iteration_limit: int = 150000
    reference_optimality_tolerance: float = 1e-4
    # Presolver for the reference solve only. The phases are always presolve-free
    # (they warm-start on the original variables); the reference's postsolved dual
    # is in the original row space and the A^T y shift is minimizer-preserving for
    # any dual, so presolve there is safe — and often necessary at large scale.
    reference_presolver: str = "pslp"
    # Reuse a previously computed reference instead of solving it (both files
    # required together; raw little-endian f64 vectors).
    initial_reference_primal: str = ""
    initial_reference_dual: str = ""

    max_passes: int = 5  # 0 = reference-only mode
    minimum_relative_secondary_improvement: float = 0.01

    # Phase budgets. Descent is a fixed budget by design (never solve-to-optimal).
    # The projection restart runs only when the projection solve hits its
    # iteration limit.
    descent_iterations: int = 8000
    projection_iterations: int = 60000
    projection_restart_iterations: int = 20000

    # Tolerance for every phase solve; the promotion gate is PDLP's Optimal
    # certificate at this tolerance.
    optimality_tolerance: float = 1e-4

    # Dump per-phase endpoint primal/dual vectors (validation / warm-start reuse).
    dump_phase_vectors: bool = False


VALUE_FLAGS = {
    "--reference-iteration-limit",
    "--reference-optimality-tolerance",
    "--reference-presolver",
    "--initial-reference-primal",
    "--initial-reference-dual",
    "--max-passes",
    "--minimum-relative-secondary-improvement",
    "--descent-iterations",
    "--projection-iterations",
    "--projection-restart-iterations",
    "--optimality-tolerance",
}
BOOL_FLAGS = {"--dump-phase-vectors"}


def parse_cli(args):
    """Returns (options, None) on success, (None, error_string) on failure."""
    opt = RunOptions()
    options = {}
    positional = []
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--"):
            if a in BOOL_FLAGS:
                if a in options:
                    return None, a + " given more than once"
                options[a] = "true"
                i += 1
                continue
            if a not in VALUE_FLAGS:
                return None, "unknown option: " + a
            if i + 1 >= len(args):
                return None, a + " requires a value"
            if a in options:
                return None, a + " given more than once"
            options[a] = args[i + 1]
            i += 2
        else:
            positional.append(a)
            i += 1

    if len(positional) != 3:
        return None, "expected exactly 3 positional arguments: PRIMARY.mps FULL.mps OUTPUT_DIR"
    opt.primary_path, opt.full_path, opt.output_dir = positional
    if opt.primary_path == opt.full_path:
        return None, "primary and full model paths must differ"

    # All integer options are narrowed to i_t before reaching the solver, so
    # reject anything above i_t's maximum here rather than overflowing later.
    def get_ll(key, current, min_value, max_value=I_T_MAX):
        if key not in options:
            return current, None
        v = parse_ll(options[key])
        if v is None or v < min_value or v > max_value:
            return None, f"{key}: invalid value '{options[key]}'"
        return v, None

    def get_pos_dbl(key, current):
        if key not in options:
            return current, None
        v = parse_dbl(options[key])
        if v is None or not math.isfinite(v) or v <= 0.0:
            return None, f"{key}: invalid value '{options[key]}'"
        return v, None

    err = None
    opt.reference_iteration_limit, err = get_ll(
        "--reference-iteration-limit", opt.reference_iteration_limit, 1)
    if err:
        return None, err
    opt.max_passes, err = get_ll("--max-passes", opt.max_passes, 0)
    if err:
        return None, err
    opt.descent_iterations, err = get_ll("--descent-iterations", opt.descent_iterations, 1)
    if err:
        return None, err
    opt.projection_iterations, err = get_ll(
        "--projection-iterations", opt.projection_iterations, 1)
    if err:
        return None, err
    opt.projection_restart_iterations, err = get_ll(
        "--projection-restart-iterations", opt.projection_restart_iterations, 1)
    if err:
        return None, err
    opt.reference_optimality_tolerance, err = get_pos_dbl(
        "--reference-optimality-tolerance", opt.reference_optimality_tolerance)
    if err:
        return None, err
    opt.optimality_tolerance, err = get_pos_dbl(
        "--optimality-tolerance", opt.optimality_tolerance)
    if err:
        return None, err
    opt.minimum_relative_secondary_improvement, err = get_pos_dbl(
        "--minimum-relative-secondary-improvement",
        opt.minimum_relative_secondary_improvement)
    if err:
        return None, err

    if "--reference-presolver" in options:
        v = options["--reference-presolver"]
        if v not in ("none", "pslp"):
            return None, "--reference-presolver must be 'none' or 'pslp'"
        opt.reference_presolver = v
    opt.initial_reference_primal = options.get("--initial-reference-primal", "")
    opt.initial_reference_dual = options.get("--initial-reference-dual", "")
    if bool(opt.initial_reference_primal) != bool(opt.initial_reference_dual):
        return None, ("--initial-reference-primal and --initial-reference-dual "
                      "must be given together")
    opt.dump_phase_vectors = "--dump-phase-vectors" in options
    return opt, None


def print_usage(f) -> None:
    f.write(
        "usage:\n"
        "  run_hierarchical_pdlp.py PRIMARY.mps[.gz] FULL.mps[.gz] OUTPUT_DIR \\\n"
        "    [--reference-presolver pslp|none]           (default pslp) \\\n"
        "    [--reference-optimality-tolerance 1e-4] [--reference-iteration-limit 150000] \\\n"
        "    [--initial-reference-primal F.f64 --initial-reference-dual F.f64]\n"
        "                    (reuse a dumped reference instead of solving it) \\\n"
        "    [--max-passes 5]                            (0 = reference-only mode) \\\n"
        "    [--minimum-relative-secondary-improvement 0.01] \\\n"
        "    [--descent-iterations 8000] \\\n"
        "    [--projection-iterations 60000] [--projection-restart-iterations 20000] \\\n"
        "    [--optimality-tolerance 1e-4]               (phase + certificate-gate tolerance) \\\n"
        "    [--dump-phase-vectors]                      (dump per-phase endpoint vectors)\n"
        "  run_hierarchical_pdlp.py --self-test\n"
    )


# =============================================================================
# Solve helpers (cuopt imports are deferred so --self-test stays CPU/GPU-free)
# =============================================================================


PRESOLVE_OFF = 0
PRESOLVE_PSLP = 2


@dataclass
class PhaseResult:
    kind: str = PHASE_DESCENT
    primal: Optional[np.ndarray] = None
    dual: Optional[np.ndarray] = None
    method_str: str = "PDLP"
    solver_mode_str: str = "Stable3"
    termination_str: str = ""
    error_str: str = "Success"
    iterations: int = 0
    solve_time: float = 0.0
    solver_primal_obj: float = 0.0
    solver_dual_obj: float = 0.0
    optimal: bool = False


def _make_settings(iteration_limit: int, opt_tol: float, presolve: int, log_path: str):
    from cuopt.linear_programming.solver_settings import SolverSettings
    from cuopt.linear_programming.solver_settings.solver_settings import (
        PDLPSolverMode,
        SolverMethod,
    )

    settings = SolverSettings()
    settings.set_parameter("method", int(SolverMethod.PDLP))
    settings.set_parameter("presolve", presolve)
    settings.set_parameter("pdlp_solver_mode", int(PDLPSolverMode.Stable3))
    settings.set_parameter("iteration_limit", int(iteration_limit))
    settings.set_parameter("log_to_console", True)
    settings.set_parameter("log_file", log_path)
    settings.set_optimality_tolerance(opt_tol)
    return settings


def _run_solve(model, settings):
    from cuopt.linear_programming.solver.solver import Solve
    from cuopt.linear_programming.solver.solver_wrapper import (
        ErrorStatus,
        LPTerminationStatus,
    )

    solution = Solve(model, settings)

    if solution.get_error_status() != ErrorStatus.Success:
        raise RuntimeError(
            "solve reported error: " + str(solution.get_error_message())
        )
    return solution, LPTerminationStatus


def _extract_endpoint(solution, n_vars: int, n_cons: int, what: str):
    primal = np.asarray(solution.get_primal_solution(), dtype=np.float64)
    dual = np.asarray(solution.get_dual_solution(), dtype=np.float64)
    if primal.size != n_vars:
        raise RuntimeError(what + ": unexpected primal vector size")
    if dual.size != n_cons:
        raise RuntimeError(what + ": unexpected dual vector size")
    if not (all_finite(primal) and all_finite(dual)):
        raise RuntimeError(what + ": non-finite value returned by solver")
    return primal, dual


def _solution_metadata(solution):
    st = float(solution.get_solve_time())
    iters = int(solution.get_lp_stats()["nb_iterations"])
    sp = float(solution.get_primal_objective())
    sd = float(solution.get_dual_objective())
    if not math.isfinite(st) or iters < 0 or not math.isfinite(sp) or not math.isfinite(sd):
        raise RuntimeError("solve returned invalid metadata")
    return st, iters, sp, sd


# Build a fresh model from the stored structure + requested objective, run a
# single fresh solve, validate every returned datum, and return the endpoint.
def run_phase(
    s: HierarchyState,
    kind: str,
    objective: np.ndarray,
    objective_offset: float,
    primal_start: np.ndarray,
    dual_start: np.ndarray,
    iteration_limit: int,
    opt_tol: float,
    log_path: str,
) -> PhaseResult:
    from cuopt.linear_programming.data_model import DataModel

    model = DataModel()
    model.set_csr_constraint_matrix(s.A_values, s.A_indices, s.A_offsets)
    model.set_constraint_bounds(s.rhs)
    model.set_row_types(np.full(s.n_cons, "E", dtype="U1"))
    model.set_constraint_lower_bounds(s.row_lower)
    model.set_constraint_upper_bounds(s.row_upper)
    model.set_variable_lower_bounds(s.var_lower)
    model.set_variable_upper_bounds(s.var_upper)
    model.set_variable_types(np.full(s.n_vars, "C", dtype="S1"))
    model.set_objective_coefficients(np.ascontiguousarray(objective, dtype=np.float64))
    model.set_objective_scaling_factor(1.0)
    model.set_objective_offset(objective_offset)
    model.set_maximize(False)
    # Transfer only explicit current primal and row dual vectors between phases
    # (the Solve wrapper plumbs these into the PDLP settings' initial solutions).
    model.set_initial_primal_solution(np.ascontiguousarray(primal_start, dtype=np.float64))
    model.set_initial_dual_solution(np.ascontiguousarray(dual_start, dtype=np.float64))

    settings = _make_settings(iteration_limit, opt_tol, PRESOLVE_OFF, log_path)
    solution, LPTerminationStatus = _run_solve(model, settings)

    term = solution.get_termination_status()
    if term not in (LPTerminationStatus.Optimal, LPTerminationStatus.IterationLimit):
        raise RuntimeError(
            "phase solve unexpected termination: " + solution.get_termination_reason()
        )

    st, iters, sp, sd = _solution_metadata(solution)
    primal, dual = _extract_endpoint(solution, s.n_vars, s.n_cons, "phase endpoint")

    pr = PhaseResult(
        kind=kind,
        primal=primal,
        dual=dual,
        iterations=iters,
        solve_time=st,
        solver_primal_obj=sp,
        solver_dual_obj=sd,
        optimal=(term == LPTerminationStatus.Optimal),
        termination_str=solution.get_termination_reason(),
    )
    require_nonempty_file(log_path)
    return pr


# Solve the parsed primary model to optimality, establish the independently
# evaluated primary reference value, and store the reference primal/dual.
# Returns (iterations, solve_time); reference work is recorded separately
# from pass work.
def solve_reference(primary_model, state: HierarchyState, opt: RunOptions):
    log_path = os.path.join(opt.output_dir, "reference.log")
    presolve = PRESOLVE_PSLP if opt.reference_presolver == "pslp" else PRESOLVE_OFF
    settings = _make_settings(
        opt.reference_iteration_limit,
        opt.reference_optimality_tolerance,
        presolve,
        log_path,
    )
    solution, LPTerminationStatus = _run_solve(primary_model, settings)

    term = solution.get_termination_status()
    if term != LPTerminationStatus.Optimal:
        raise RuntimeError(
            "reference solve did not reach Optimal (got '"
            + solution.get_termination_reason()
            + "'); consider raising --reference-iteration-limit"
        )

    st, iters, sp, sd = _solution_metadata(solution)
    primal, dual = _extract_endpoint(solution, state.n_vars, state.n_cons, "reference")

    # A presolved reference solve returns a postsolved primal, whose substituted
    # entries can land marginally outside variable bounds.
    primal = clamp_warm_start(state, primal, "reference")

    require_nonempty_file(log_path)

    # primary_reference from an independent evaluation (temporary reference 0);
    # never the solver-reported objective.
    e0 = evaluate_point(state, 0.0, primal)
    state.primary_reference = e0.primary

    # Self-consistency: re-evaluating against the stored reference yields ~0
    # deviation.
    e1 = evaluate_point(state, state.primary_reference, primal)
    tol = 1e-9 * max(1.0, abs(state.primary_reference))
    if abs(e1.primary_deviation) > tol:
        raise RuntimeError("reference primary self-consistency check failed")

    state.reference_primal = primal
    state.reference_dual = dual
    return iters, st


# Load a previously computed reference pair from raw f64 files instead of
# solving (mirrors solve_reference's post-processing: clamp into the box,
# independent primary_reference evaluation). Returns (0, 0.0) to mirror
# solve_reference's (iterations, solve_time).
def load_reference(state: HierarchyState, opt: RunOptions):
    primal = read_f64(opt.initial_reference_primal, state.n_vars, "--initial-reference-primal")
    dual = read_f64(opt.initial_reference_dual, state.n_cons, "--initial-reference-dual")

    primal = clamp_warm_start(state, primal, "loaded reference")

    e0 = evaluate_point(state, 0.0, primal)
    state.primary_reference = e0.primary
    state.reference_primal = primal
    state.reference_dual = dual
    print(
        "note: reference loaded from files; primary_reference="
        + fmt_double(state.primary_reference),
        file=sys.stderr,
        flush=True,
    )
    return 0, 0.0


# =============================================================================
# Output helpers
# =============================================================================


PROGRESS_HEADER = (
    "pass,phase,method,solver_mode,termination,error_status,iterations,solve_time,"
    "cumulative_iterations,cumulative_solve_time,solver_primal_objective,solver_dual_objective,"
    "primary,secondary,full,primary_deviation,primary_absolute_deviation,"
    "row_l2,row_linf,bound_l2,bound_linf,promoted,"
    "relative_secondary_improvement,reason"
)


def write_progress_row(
    csv_f,
    pass_no: int,
    pr: PhaseResult,
    e: EvaluationResult,
    cum_iters: int,
    cum_time: float,
    promoted: bool,
    relative_improvement: Optional[float],
    reason: str,
) -> None:
    fields = [
        str(pass_no),
        csv_field(pr.kind),
        csv_field(pr.method_str),
        csv_field(pr.solver_mode_str),
        csv_field(pr.termination_str),
        csv_field(pr.error_str),
        str(pr.iterations),
        fmt_double(pr.solve_time),
        str(cum_iters),
        fmt_double(cum_time),
        fmt_double(pr.solver_primal_obj),
        fmt_double(pr.solver_dual_obj),
        fmt_double(e.primary),
        fmt_double(e.secondary),
        fmt_double(e.full),
        fmt_double(e.primary_deviation),
        fmt_double(e.primary_abs_deviation),
        fmt_double(e.row_l2),
        fmt_double(e.row_linf),
        fmt_double(e.bound_l2),
        fmt_double(e.bound_linf),
        "true" if promoted else "false",
        fmt_double(relative_improvement) if relative_improvement is not None else "",
        csv_field(reason),
    ]
    csv_f.write(",".join(fields) + "\n")
    csv_f.flush()


@dataclass
class RunResult:
    stop_reason: str = ""
    accepted_passes: int = 0
    final_eval: EvaluationResult = field(default_factory=EvaluationResult)
    cumulative_iterations: int = 0
    cumulative_solve_time: float = 0.0
    final_primal: Optional[np.ndarray] = None


# One line of the end-of-run solve history: a solve (reference or phase) with
# its audited objectives and, for gate candidates, the signed relative change
# of the secondary vs the incumbent (negative = improvement).
@dataclass
class HistoryRow:
    label: str
    termination: str
    iterations: int
    solve_time: float
    primary: float
    primary_deviation: float
    secondary: float
    secondary_change: Optional[float]  # (candidate - incumbent) / max(1, |incumbent|)
    decision: str


def render_history(history, res: RunResult):
    lines = ["solve history:"]
    lines.append(
        f"  {'solve':<26}{'termination':<16}{'iterations':>10}{'time_s':>9}"
        f"{'primary':>17}{'primary_dev':>13}{'secondary':>17}"
        f"{'secondary_change':>18}  decision"
    )
    for r in history:
        if r.secondary_change is None:
            change = "-"
        else:
            pct = 100.0 * r.secondary_change
            # Scientific notation for changes that would round to +/-0.00%.
            change = f"{pct:+.2f}%" if pct == 0.0 or abs(pct) >= 0.005 else f"{pct:+.2e}%"
        lines.append(
            f"  {r.label:<26}{r.termination:<16}{r.iterations:>10}{r.solve_time:>9.1f}"
            f"{r.primary:>17.8e}{r.primary_deviation:>13.3e}{r.secondary:>17.8e}"
            f"{change:>18}  {r.decision}"
        )
    lines.append(
        f"totals: accepted_passes={res.accepted_passes}"
        f" cumulative_phase_iterations={res.cumulative_iterations}"
        f" cumulative_phase_solve_time={res.cumulative_solve_time:.1f}s"
    )
    return lines


def write_summary(output_dir: str, state: HierarchyState, res: RunResult,
                  reference_iterations: int) -> None:
    e = res.final_eval
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(
            "status=success\n"
            f"stop_reason={res.stop_reason}\n"
            f"accepted_passes={res.accepted_passes}\n"
            f"n_variables={state.n_vars}\n"
            f"n_constraints={state.n_cons}\n"
            f"reference_iterations={reference_iterations}\n"
            f"cumulative_phase_iterations={res.cumulative_iterations}\n"
            f"cumulative_phase_solve_time={fmt_double(res.cumulative_solve_time)}\n"
            f"primary_reference={fmt_double(state.primary_reference)}\n"
            f"final_primary={fmt_double(e.primary)}\n"
            f"final_secondary={fmt_double(e.secondary)}\n"
            f"final_full={fmt_double(e.full)}\n"
            f"final_primary_deviation={fmt_double(e.primary_deviation)}\n"
            f"final_row_l2={fmt_double(e.row_l2)}\n"
            f"final_row_linf={fmt_double(e.row_linf)}\n"
            f"final_bound_l2={fmt_double(e.bound_l2)}\n"
            f"final_bound_linf={fmt_double(e.bound_linf)}\n"
        )


def write_final_primal(output_dir: str, primal: np.ndarray) -> None:
    write_f64(os.path.join(output_dir, "final_primal.f64"), primal)


def pass_file_path(directory: str, pass_no: int, suffix: str) -> str:
    return os.path.join(directory, f"pass_{pass_no:03d}.{suffix}")


# =============================================================================
# Iterative procedure
# =============================================================================


def run_passes(state: HierarchyState, opt: RunOptions, csv_f, history) -> RunResult:
    res = RunResult()
    incumbent_primal = state.reference_primal.copy()
    incumbent_eval = evaluate_point(state, state.primary_reference, incumbent_primal)

    cum_iters = 0
    cum_time = 0.0
    accepted = 0

    zero_dual = np.zeros(state.n_cons, dtype=np.float64)
    # The shift depends only on the (fixed) reference dual and c1/c2, so it is
    # constant across passes.
    c_shift = build_shifted_objective(state)

    def finalize(reason: str) -> RunResult:
        res.stop_reason = reason
        res.accepted_passes = accepted
        res.final_eval = incumbent_eval
        res.final_primal = incumbent_primal
        res.cumulative_iterations = cum_iters
        res.cumulative_solve_time = cum_time
        return res

    def dump_endpoint(pass_no: int, pr: PhaseResult) -> None:
        if not opt.dump_phase_vectors:
            return
        write_f64(pass_file_path(opt.output_dir, pass_no, pr.kind + ".primal.f64"), pr.primal)
        write_f64(pass_file_path(opt.output_dir, pass_no, pr.kind + ".dual.f64"), pr.dual)

    while accepted < opt.max_passes:
        pass_no = accepted + 1

        # --- DESCEND: shifted objective, incumbent primal, zero dual (the zero
        # dual encodes y_full ≈ y_ref, the shifted problem's principled cold
        # start). Fixed budget; the endpoint is an intermediate point that only
        # seeds the projection — never promoted. ---
        descent_start = clamp_warm_start(state, incumbent_primal, "descent")
        r_d = run_phase(
            state,
            PHASE_DESCENT,
            c_shift,
            0.0,
            descent_start,
            zero_dual,
            opt.descent_iterations,
            opt.optimality_tolerance,
            pass_file_path(opt.output_dir, pass_no, "descent.log"),
        )
        e_d = evaluate_point(state, state.primary_reference, r_d.primal)
        cum_iters = checked_add_iterations(cum_iters, r_d.iterations)
        cum_time = checked_add_time(cum_time, r_d.solve_time)
        write_progress_row(csv_f, pass_no, r_d, e_d, cum_iters, cum_time, False, None,
                           "intermediate")
        history.append(HistoryRow(
            f"pass {pass_no} descent", r_d.termination_str, r_d.iterations,
            r_d.solve_time, e_d.primary, e_d.primary_deviation, e_d.secondary,
            None, "intermediate"))
        dump_endpoint(pass_no, r_d)

        # --- PROJECT: primary objective, warm from the descent endpoint,
        # reference-dual start. The pass candidate is this solve's endpoint —
        # unless it hits the iteration limit, in which case ONE fresh restart
        # from that endpoint must certify. ---
        projection_start = clamp_warm_start(state, r_d.primal, "projection")
        r_p = run_phase(
            state,
            PHASE_PROJECTION,
            state.c1,
            state.offset1,
            projection_start,
            state.reference_dual,
            opt.projection_iterations,
            opt.optimality_tolerance,
            pass_file_path(opt.output_dir, pass_no, "projection.log"),
        )
        e_p = evaluate_point(state, state.primary_reference, r_p.primal)
        cum_iters = checked_add_iterations(cum_iters, r_p.iterations)
        cum_time = checked_add_time(cum_time, r_p.solve_time)
        dump_endpoint(pass_no, r_p)

        if not r_p.optimal:
            write_progress_row(csv_f, pass_no, r_p, e_p, cum_iters, cum_time, False, None,
                               "projection_stalled")
            history.append(HistoryRow(
                f"pass {pass_no} projection", r_p.termination_str, r_p.iterations,
                r_p.solve_time, e_p.primary, e_p.primary_deviation, e_p.secondary,
                None, "projection_stalled"))

            restart_start = clamp_warm_start(state, r_p.primal, "projection_restart")
            r_r = run_phase(
                state,
                PHASE_PROJECTION_RESTART,
                state.c1,
                state.offset1,
                restart_start,
                state.reference_dual,
                opt.projection_restart_iterations,
                opt.optimality_tolerance,
                pass_file_path(opt.output_dir, pass_no, "projection_restart.log"),
            )
            e_p = evaluate_point(state, state.primary_reference, r_r.primal)
            cum_iters = checked_add_iterations(cum_iters, r_r.iterations)
            cum_time = checked_add_time(cum_time, r_r.solve_time)
            dump_endpoint(pass_no, r_r)
            r_p = r_r
            candidate_label = f"pass {pass_no} projection_restart"
        else:
            candidate_label = f"pass {pass_no} projection"

        # --- Promotion (the certified projection endpoint is the candidate) ---
        d = decide_promotion(
            r_p.optimal,
            e_p.secondary,
            incumbent_eval.secondary,
            opt.minimum_relative_secondary_improvement,
        )
        write_progress_row(
            csv_f, pass_no, r_p, e_p, cum_iters, cum_time, d.promoted,
            d.relative_improvement if d.has_relative else None, d.reason,
        )
        history.append(HistoryRow(
            candidate_label, r_p.termination_str, r_p.iterations, r_p.solve_time,
            e_p.primary, e_p.primary_deviation, e_p.secondary,
            (e_p.secondary - incumbent_eval.secondary)
            / max(1.0, abs(incumbent_eval.secondary)),
            d.reason))
        if not d.promoted:
            return finalize(d.reason)
        incumbent_primal = r_p.primal
        incumbent_eval = e_p
        accepted += 1
        if d.stop:
            return finalize("secondary_improvement_below_threshold")

    return finalize("maximum_passes")


# Set when run() creates the output directory, so the failure.txt handler in
# main() only writes into a directory this run owns.
g_created_output_dir = False


def run(opt: RunOptions) -> int:
    global g_created_output_dir
    from cuopt.linear_programming.io.parser import Read

    # Parse and validate both models before touching the filesystem / GPU.
    primary_model = Read(opt.primary_path)
    full_model = Read(opt.full_path)
    validate_model(primary_model, "primary")
    validate_model(full_model, "full")
    validate_match(primary_model, full_model)
    state = build_state(primary_model, full_model)
    del full_model  # free the full model before the solves

    if os.path.exists(opt.output_dir):
        raise RuntimeError("output path already exists: " + opt.output_dir)
    os.mkdir(opt.output_dir)
    g_created_output_dir = True

    # Reference: solve (initialization work, recorded separately) or reuse a
    # previously computed pair. A solved reference is always dumped so any later
    # run can reuse it via --initial-reference-primal/-dual.
    solved_reference = not opt.initial_reference_primal
    if solved_reference:
        reference_iterations, reference_time = solve_reference(primary_model, state, opt)
        write_f64(os.path.join(opt.output_dir, "reference_primal.f64"),
                  state.reference_primal)
        write_f64(os.path.join(opt.output_dir, "reference_dual.f64"),
                  state.reference_dual)
    else:
        reference_iterations, reference_time = load_reference(state, opt)
    del primary_model

    e_ref = evaluate_point(state, state.primary_reference, state.reference_primal)
    history = [HistoryRow(
        "reference" if solved_reference else "reference (loaded)",
        "Optimal" if solved_reference else "-",
        reference_iterations, reference_time,
        e_ref.primary, e_ref.primary_deviation, e_ref.secondary,
        None, "reference")]

    with open(os.path.join(opt.output_dir, "progress.csv"), "w") as csv_f:
        csv_f.write(PROGRESS_HEADER + "\n")
        csv_f.flush()

        if opt.max_passes == 0:
            # Reference-only mode: the deliverable is the audited reference itself.
            res = RunResult()
            res.stop_reason = "reference_only"
            res.accepted_passes = 0
            res.final_eval = e_ref
            res.final_primal = state.reference_primal
        else:
            res = run_passes(state, opt, csv_f, history)

    write_final_primal(opt.output_dir, res.final_primal)
    write_summary(opt.output_dir, state, res, reference_iterations)

    print(flush=True)
    for line in render_history(history, res):
        print(line, flush=True)
    print(flush=True)
    print(
        f"status=success stop_reason={res.stop_reason}"
        f" accepted_passes={res.accepted_passes}"
        f" reference_iterations={reference_iterations}"
        f" cumulative_phase_iterations={res.cumulative_iterations}"
        f" final_secondary={fmt_double(res.final_eval.secondary)}",
        flush=True,
    )
    return 0


# =============================================================================
# Self-test (deterministic, CPU-only; pure hierarchy functions + CLI)
# =============================================================================


g_self_test_failures = 0


def st_check(name: str, ok: bool) -> None:
    global g_self_test_failures
    if not ok:
        print("SELF_TEST FAIL: " + name, file=sys.stderr)
        g_self_test_failures += 1


def st_expect_throws(name: str, fn) -> None:
    threw = False
    try:
        fn()
    except Exception:
        threw = True
    st_check(name, threw)


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def make_test_state() -> HierarchyState:
    # 2 rows, 3 vars.
    # Row 0: cols {0,1} vals {1,2}; Row 1: cols {1,2} vals {3,4}.
    s = HierarchyState()
    s.n_vars = 3
    s.n_cons = 2
    s.A_offsets = np.array([0, 2, 4], dtype=np.int32)
    s.A_indices = np.array([0, 1, 1, 2], dtype=np.int32)
    s.A_values = np.array([1.0, 2.0, 3.0, 4.0])
    s.nnz = 4
    s.c1 = np.array([10.0, 20.0, 30.0])
    s.c2 = np.array([1.0, 1.0, 1.0])
    s.offset1 = 0.0
    s.offset2 = 0.0
    s.rhs = np.array([5.0, 20.0])
    s.row_lower = np.array([5.0, 20.0])
    s.row_upper = np.array([5.0, 20.0])
    s.var_lower = np.array([0.0, 0.0, 0.0])
    s.var_upper = np.array([10.0, 10.0, 2.0])
    return s


def run_self_test() -> bool:
    global g_self_test_failures
    g_self_test_failures = 0

    # 1. Effective objective derivation, including scale and offset.
    eff = compute_effective_objectives([1.0, 2.0], 2.0, 3.0, [4.0, 6.0], 0.5, 10.0)
    st_check("effective_c1", np.array_equal(eff.c1, [2.0, 4.0]))
    st_check("effective_c2", np.array_equal(eff.c2, [0.0, -1.0]))
    st_check("effective_offset1", approx(eff.offset1, 6.0))
    st_check("effective_offset2", approx(eff.offset2, -1.0))  # 0.5*10 - 6 = -1
    st_expect_throws(
        "effective_size_mismatch",
        lambda: compute_effective_objectives([1.0], 1.0, 0.0, [1.0, 2.0], 1.0, 0.0),
    )

    # 2. Shifted objective on a hand-computed equality system.
    s = make_test_state()
    s.reference_dual = np.array([5.0, 7.0])
    # A^T y: col0 = 1*5 = 5; col1 = 2*5 + 3*7 = 31; col2 = 4*7 = 28.
    # c_shift = c1 - A^T y + c2 = [10-5+1, 20-31+1, 30-28+1] = [6, -10, 3].
    cs = build_shifted_objective(s)
    st_check("shifted_objective", np.array_equal(cs, [6.0, -10.0, 3.0]))

    # 3. Independent primary / secondary / full evaluation.
    s = make_test_state()
    s.offset1 = 100.0
    s.offset2 = 5.0
    e = evaluate_point(s, 0.0, np.array([1.0, 2.0, 3.0]))
    st_check("eval_primary", approx(e.primary, 240.0))
    st_check("eval_secondary", approx(e.secondary, 11.0))
    st_check("eval_full", approx(e.full, 251.0))

    # 4. Equality-row and variable-bound violation L2 / L-infinity.
    s = make_test_state()  # rhs {5,20}, var_upper {10,10,2}
    # x = {1,2,3}: row0 activity = 5 (ok); row1 activity = 18, |18-20| = 2.
    # var2 = 3 > upper 2 -> bound violation 1.
    e = evaluate_point(s, 0.0, np.array([1.0, 2.0, 3.0]))
    st_check("row_l2", approx(e.row_l2, 2.0))
    st_check("row_linf", approx(e.row_linf, 2.0))
    st_check("bound_l2", approx(e.bound_l2, 1.0))
    st_check("bound_linf", approx(e.bound_linf, 1.0))
    st_check("no_dual_correction_without_ref_dual", not e.has_dual_correction)

    # 4b. Dual-corrected primary. Row0 activity 5 = rhs (signed violation 0);
    # row1 activity 18 vs rhs 20 (signed violation -2). y_ref = {5, 7} ->
    # correction = 5*0 + 7*(-2) = -14. c1 = {1,0,0}, offset1 = 0, x = {1,2,3}
    # -> primary = 1; corrected = 1 - (-14) = 15.
    s = make_test_state()
    s.c1 = np.array([1.0, 0.0, 0.0])
    s.offset1 = 0.0
    s.reference_dual = np.array([5.0, 7.0])
    e = evaluate_point(s, 15.0, np.array([1.0, 2.0, 3.0]))
    st_check("dual_correction_flag", e.has_dual_correction)
    st_check("dual_correction_value", approx(e.dual_correction, -14.0))
    st_check("corrected_primary_value", approx(e.corrected_primary, 15.0))
    st_check("corrected_deviation_zero", approx(e.corrected_primary_deviation, 0.0))
    st_check("raw_deviation", approx(e.primary_deviation, -14.0))
    # Feasible point: correction is zero, corrected == raw.
    ef = evaluate_point(s, 0.0, np.array([5.0, 0.0, 5.0]))
    st_check("feasible_zero_correction",
             approx(ef.dual_correction, 0.0) and approx(ef.corrected_primary, ef.primary))

    # 4c. Warm-start clamping: out-of-box entries land exactly on their bound,
    #     in-box entries are untouched, and the length is validated.
    s = make_test_state()  # var box: [0,10] [0,10] [0,2]
    v = clamp_warm_start(s, np.array([-1.0e-10, 5.0, 2.0 + 1.0e-10]), "test")
    st_check("clamp_below", v[0] == 0.0)
    st_check("clamp_inside_untouched", v[1] == 5.0)
    st_check("clamp_above", v[2] == 2.0)
    w = clamp_warm_start(s, np.array([0.0, 10.0, 2.0]), "test")
    st_check("clamp_noop_on_bounds", np.array_equal(w, [0.0, 10.0, 2.0]))
    st_expect_throws("clamp_rejects_wrong_length",
                     lambda: clamp_warm_start(s, np.array([1.0, 2.0]), "test"))

    # 5. Promotion gates: rejection for uncertified and nonimproving candidates.
    d_nonopt = decide_promotion(False, 0.0, 10.0, 0.01)
    st_check("reject_uncertified",
             not d_nonopt.promoted and d_nonopt.reason == "projection_not_certified")
    d_equal = decide_promotion(True, 10.0, 10.0, 0.01)
    st_check("reject_not_improved_equal",
             not d_equal.promoted and d_equal.reason == "secondary_not_improved")
    d_worse = decide_promotion(True, 11.0, 10.0, 0.01)
    st_check("reject_not_improved_worse", not d_worse.promoted)

    # 6. Promotion + continuation for relative improvement >= threshold.
    d_above = decide_promotion(True, 98.0, 100.0, 0.01)  # rel 0.02
    st_check("promote_continue_above",
             d_above.promoted and not d_above.stop and d_above.reason == "promoted")
    d_exact = decide_promotion(True, 99.0, 100.0, 0.01)  # rel exactly 0.01
    st_check("promote_continue_exact",
             d_exact.promoted and not d_exact.stop and d_exact.reason == "promoted")

    # 7. Promotion + stop for a positive improvement strictly below threshold.
    d = decide_promotion(True, 99.5, 100.0, 0.01)  # rel 0.005
    st_check("promote_stop_below",
             d.promoted and d.stop and d.reason == "secondary_improvement_below_threshold")

    # 8. Cumulative iteration / time overflow rejection.
    st_check("checked_add_normal", checked_add_iterations(5, 7) == 12)
    st_expect_throws("checked_add_iterations_overflow",
                     lambda: checked_add_iterations(LL_MAX, 1))
    st_check("checked_add_time_normal", approx(checked_add_time(1.0, 2.0), 3.0))
    st_expect_throws("checked_add_time_overflow",
                     lambda: checked_add_time(1.7e308, 1.7e308))

    # 8b. Solve-history rendering: header, change formatting, totals.
    hist = [
        HistoryRow("reference", "Optimal", 100, 1.5, -10.0, 0.0, 50.0,
                   None, "reference"),
        HistoryRow("pass 1 projection", "Optimal", 200, 2.5, -10.5, -0.5, 40.0,
                   (40.0 - 50.0) / max(1.0, abs(50.0)), "promoted"),
    ]
    rr = RunResult(stop_reason="x", accepted_passes=1,
                   cumulative_iterations=300, cumulative_solve_time=4.0)
    lines = render_history(hist, rr)
    st_check("history_line_count", len(lines) == 5)  # title, header, 2 rows, totals
    st_check("history_header",
             "secondary_change" in lines[1] and "decision" in lines[1])
    st_check("history_reference_row",
             "reference" in lines[2] and lines[2].rstrip().endswith("reference"))
    st_check("history_change_pct", "-20.00%" in lines[3] and "promoted" in lines[3])
    st_check("history_totals",
             "accepted_passes=1" in lines[-1]
             and "cumulative_phase_iterations=300" in lines[-1])

    # 9. CLI: documented defaults parse; unknown flags rejected.
    def parse(extra):
        return parse_cli(["a.mps", "b.mps", "outdir"] + extra)

    o, err = parse([])
    st_check("cli_defaults_parse", err is None)
    st_check("cli_positionals", o.primary_path == "a.mps" and o.full_path == "b.mps"
             and o.output_dir == "outdir")
    st_check("cli_default_ref_tol", approx(o.reference_optimality_tolerance, 1e-4))
    st_check("cli_default_ref_presolver", o.reference_presolver == "pslp")
    st_check("cli_default_ref_limit", o.reference_iteration_limit == 150000)
    st_check("cli_default_max_passes", o.max_passes == 5)
    st_check("cli_default_descent", o.descent_iterations == 8000)
    st_check("cli_default_projection", o.projection_iterations == 60000)
    st_check("cli_default_projection_restart", o.projection_restart_iterations == 20000)
    st_check("cli_default_tolerance", approx(o.optimality_tolerance, 1e-4))
    st_check("cli_default_min_improvement",
             approx(o.minimum_relative_secondary_improvement, 0.01))
    st_check("cli_default_no_dump", not o.dump_phase_vectors)

    o2, err = parse(["--descent-iterations", "4000", "--projection-iterations", "100000",
                     "--projection-restart-iterations", "10000", "--optimality-tolerance",
                     "1e-6", "--max-passes", "0", "--reference-presolver", "none",
                     "--dump-phase-vectors"])
    st_check("cli_parse_values",
             err is None and o2.descent_iterations == 4000
             and o2.projection_iterations == 100000
             and o2.projection_restart_iterations == 10000
             and approx(o2.optimality_tolerance, 1e-6) and o2.max_passes == 0
             and o2.reference_presolver == "none" and o2.dump_phase_vectors)
    o3, err = parse(["--initial-reference-primal", "p.f64",
                     "--initial-reference-dual", "d.f64"])
    st_check("cli_ref_pair",
             err is None and o3.initial_reference_primal == "p.f64"
             and o3.initial_reference_dual == "d.f64")

    def rejects(extra):
        _, err = parse(extra)
        return err is not None

    st_check("cli_reject_single_phase", rejects(["--single-phase"]))
    st_check("cli_reject_hp", rejects(["--hp", "restart_k_p=0"]))
    st_check("cli_reject_max_cycles", rejects(["--max-cycles", "5"]))
    st_check("cli_reject_phase_a", rejects(["--phase-a-iterations", "1000"]))
    st_check("cli_reject_bootstrap", rejects(["--primary-bootstrap-iterations", "50000"]))
    st_check("cli_reject_zero_descent", rejects(["--descent-iterations", "0"]))
    st_check("cli_reject_zero_projection", rejects(["--projection-iterations", "0"]))
    st_check("cli_reject_negative_passes", rejects(["--max-passes", "-1"]))
    st_check("cli_reject_overflow_ref_limit",
             rejects(["--reference-iteration-limit", "5000000000"]))
    st_check("cli_reject_overflow_max_passes", rejects(["--max-passes", "5000000000"]))
    st_check("cli_reject_overflow_projection",
             rejects(["--projection-iterations", "5000000000"]))
    o4, err = parse(["--reference-iteration-limit", "2147483647"])
    st_check("cli_accept_int_max_ref_limit",
             err is None and o4.reference_iteration_limit == 2147483647)
    st_check("cli_reject_zero_tol", rejects(["--optimality-tolerance", "0"]))
    st_check("cli_reject_bad_presolver", rejects(["--reference-presolver", "papilo"]))
    st_check("cli_reject_primal_alone", rejects(["--initial-reference-primal", "p.f64"]))
    st_check("cli_reject_dual_alone", rejects(["--initial-reference-dual", "d.f64"]))
    st_check("cli_reject_repeated_flag", rejects(["--max-passes", "2", "--max-passes", "3"]))
    st_check("cli_reject_repeated_bool",
             rejects(["--dump-phase-vectors", "--dump-phase-vectors"]))
    st_check("cli_reject_missing_value", rejects(["--max-passes"]))

    _, err = parse_cli(["a.mps", "b.mps"])
    st_check("cli_reject_two_positionals", err is not None)
    _, err = parse_cli(["a.mps", "b.mps", "c", "d"])
    st_check("cli_reject_four_positionals", err is not None)
    _, err = parse_cli(["a.mps", "a.mps", "out"])
    st_check("cli_reject_same_paths", err is not None)

    if g_self_test_failures == 0:
        print("SELF_TEST PASS")
        return True
    print(f"SELF_TEST FAILED: {g_self_test_failures} case(s)", file=sys.stderr)
    return False


def setup_rmm_pool() -> None:
    # An RMM pool (over an async upstream) sized to a fraction of free device
    # memory, installed as the process-wide current device resource.
    import rmm

    free_mem, _total = rmm.mr.available_device_memory()
    granularity = 256
    initial = (int(free_mem * 0.4) // granularity) * granularity
    pool = rmm.mr.PoolMemoryResource(rmm.mr.CudaAsyncMemoryResource(),
                                     initial_pool_size=initial)
    rmm.mr.set_current_device_resource(pool)


def main(argv) -> int:
    # --self-test: deterministic CPU-only checks, no MPS files, no solver.
    if "--self-test" in argv[1:]:
        if len(argv) != 2:
            print("FATAL: --self-test takes no other arguments", file=sys.stderr)
            return 2
        try:
            return 0 if run_self_test() else 1
        except Exception as e:
            print("FATAL: " + str(e), file=sys.stderr)
            return 1

    opt, error = parse_cli(argv[1:])
    if opt is None:
        sys.stderr.write("error: " + error + "\n")
        print_usage(sys.stderr)
        return 2

    try:
        setup_rmm_pool()
        return run(opt)
    except Exception as e:
        print("FATAL: " + str(e), file=sys.stderr)
        # Retain evidence: if the output directory was created, record the failure.
        if g_created_output_dir and opt.output_dir and os.path.isdir(opt.output_dir):
            try:
                with open(os.path.join(opt.output_dir, "failure.txt"), "w") as f:
                    f.write(str(e) + "\n")
            except OSError:
                pass
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
