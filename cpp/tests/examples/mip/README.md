# Inspecting MPS structure before and after presolve

`mps_presolve_structure` is a small internal C++ driver that:

1. reads a free- or fixed-format MPS file;
2. detects structural roles in the original formulation;
3. applies cuOpt's host-side PaPILO presolver; and
4. repeats the analysis on the reduced formulation and reports structural deltas.

The summary includes variable and constraint classes, bound classes, row and
column degree statistics, the densest named rows and columns, connected
components in the row-column bipartite graph, and the row-derived conflict
graph for binary literals. Conflict information includes base cliques, clique
extensions, edges, degrees, components, and the most-conflicted literals.
Connected matrix components expose independent blocks in the constraint
matrix.

The special-structure analysis scans the original rows directly so every
relation retains its source row. It reports exact-one, set-packing,
set-covering, GUB, binary-cardinality, binary-knapsack, two-binary
implication, variable-bound/activation, affine-definition, and
singleton-equality roles.
Row-family tags are non-exclusive where appropriate. Exact-one records retain
literal polarity, member coefficients, coverage, overlap components, and
stable stage-local row IDs.

Binary-domain counts include integral columns fixed at 0 or 1, while the
conflict-table summary labels its count as *unfixed binary* because only those
columns contribute two live literal states. This distinction matters for
formulations such as `30n20b8`, whose source exact-one rows retain many
fixed-zero members.

Exact-one groups are re-detected independently at each model stage and define
stage-local block-graph projections:

- direct co-occurrence in a row;
- coupling through one ungrouped integer variable;
- coupling through one continuous variable;
- either mediator; and
- the union of direct and mediated coupling.

Each projection reports weighted edges, connected-component sizes, isolated
blocks, and the most coupled blocks. Additional views decompose integer-only
rows, continuous equalities, mixed-response rows, and the continuous
projection. The report includes interface widths, repeated component
signatures, deterministic coefficient-labelled refinement, numerical scale
diagnostics, and continuous-repair candidates.

Pair-level block provenance is capped at 100,000 witness occurrences per
projection kind to avoid quadratic memory growth around global rows or
mediators. Reports mark capped projections incomplete and retain candidate and
materialized counts. Mediator hyperedges, including every block-to-mediator
source-row path, remain complete. Original blocks are not transported through
presolve; the presolved report describes newly detected groups, not a claim
that the original block system survived unchanged.

The conflict section uses cuOpt's clique-table builder, so it reports exact
conflicts implied by individual knapsack, set-packing, and set-partitioning
rows. It includes each binary literal's conflict with its complement, but does
not include conflicts discovered later by probing or branch-and-bound search.

This example deliberately uses cuOpt's internal C++ presolve API. That API is
not stable or installed as part of the supported public interface, so build the
driver from this source tree.

## Build and run

Configure cuOpt with tests enabled, then build only this example:

```bash
cmake --build cpp/build --target mps_presolve_structure --parallel "$PARALLEL_LEVEL"
cpp/build/tests/examples/mip/mps_presolve_structure path/to/model.mps
```

Compressed `.mps.gz` and `.mps.bz2` inputs work when the corresponding parser
support was enabled in the cuOpt build.

Use `--fixed` when the input uses fixed-column MPS formatting:

```bash
cpp/build/tests/examples/mip/mps_presolve_structure path/to/model.mps --fixed
```

Analysis switches are orthogonal and may be repeated:

```bash
# Static row provenance, blocks, decompositions, and diagnostics are always run.
cpp/build/tests/examples/mip/mps_presolve_structure model.mps --level basic

# Add deterministic three-round coefficient-labelled refinement.
cpp/build/tests/examples/mip/mps_presolve_structure model.mps --level symmetry

# Add a source-objective LP relaxation overlay (integer types relaxed, no cuts).
cpp/build/tests/examples/mip/mps_presolve_structure model.mps --level lp

# Also solve an explicitly labelled zero-objective A/B relaxation.
cpp/build/tests/examples/mip/mps_presolve_structure model.mps \
  --level lp --objective-erased-lp

# Write complete relation membership and provenance to a JSON sidecar.
cpp/build/tests/examples/mip/mps_presolve_structure model.mps \
  --level lp --level symmetry --json structure.json
```

The LP overlay uses finite time and iteration limits. It reports integer
fractionality, exact-one entropy/effective support/margins, row slack and dual
summaries, and reduced costs by variable family. The source-objective solve is
the default because erasing the objective can select a different degenerate LP
face.

`--level probing` is intentionally rejected. The current internal probing API
mutates solver state and does not retain enough infeasible-literal,
binary-to-continuous, or source-row provenance to produce the promised report
safely. Likewise, PaPILO currently exposes surviving column mappings but not a
complete row-reduction trace; the driver labels structural changes as net
deltas and does not guess whether a missing column was fixed, substituted, or
aggregated. Implied-integer records carry both their reduced and original
column IDs. The reduced-to-original row map identifies the surviving source-row
slot, but does not prove that the row's coefficients and bounds were unchanged
or name every source involved in a parallel-row reduction. The `symmetry` level
reports deterministic refinement-equivalent classes, not proof of automorphism
or isomorphism.

PaPILO presolve does not accept quadratic objectives or quadratic constraints,
and the host-only path does not perform cuOpt's GPU-side semi-continuous
reformulation. The driver reports a clear error for these inputs rather than
silently inspecting a different model.
