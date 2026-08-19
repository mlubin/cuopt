# Inspecting MPS structure before and after presolve

`mps_presolve_structure` is a small internal C++ driver that:

1. reads a free- or fixed-format MPS file;
2. detects structural roles in the original formulation;
3. applies cuOpt's host-side PaPILO presolver; and
4. repeats the analysis on the reduced formulation and reports structural deltas.

Agents extending the example should start with the
[research breadcrumbs](#research-breadcrumbs), which map each research question
to a code pickup point and a common provenance/test contract.

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

## Research breadcrumbs

Potential extensions are marked in the source as
`RESEARCH-BREADCRUMB(mps-structure/<slug>)`. To find every pickup point, run:

```bash
rg 'RESEARCH-BREADCRUMB\(mps-structure/' cpp/tests/examples/mip
```

The bracketed tier on each marker describes its expected scope:

- `driver-local`: implementable entirely in this example;
- `internal-api`: needs a typed cuOpt-internal result or additional provenance;
- `solver-invasive`: needs solver instrumentation or an isolated execution mode.

The breadcrumbs are research questions, not promises that a detector is sound
for every formulation. An implementation is complete only when it:

1. returns a typed record rather than printing while it detects;
2. feeds bounded human output and the JSON sidecar from that same record;
3. identifies the model stage and preserves stage-local plus available original
   row/column IDs, row side, detector, and source witnesses;
4. records tolerances, limits, elapsed work, and `complete: false` whenever a
   cap or timeout can change the result;
5. distinguishes a candidate heuristic from a checked certificate;
6. adds deterministic tests covering a positive case, a near miss, relevant
   polarity/scaling, provenance, and any budget boundary; and
7. documents the exact mathematical definition without inferring intent from
   MPS names.

When pair expansion can be quadratic, retain the complete relation as a
hyperedge and cap only its pairwise projection. When an analysis invokes a
mutating solver subsystem, run it on an isolated copy and return an explicit
status instead of changing the model used by other detectors.

| Breadcrumb | Tier | Research question and natural pickup point |
|---|---|---|
| `safe-json-sidecar` | driver-local | Reject an output equivalent to the MPS input and commit a completed sidecar atomically. Start at `write_structure_json()` and the option handling in `mps_presolve_structure.cpp`. |
| `run-manifest` | driver-local | Record input identity, effective detector/presolve/LP settings, statuses, timings, caps, and the cuOpt build identity. Extend the top-level JSON document rather than individual stage detectors. |
| `report-detail-levels` | driver-local | Add `summary`, `relations`, and `full` serialization tiers plus a collection-oriented JSONL/CSV summary. Keep detection independent of presentation. |
| `typed-conflict-report` | driver-local | Replace the print-only conflict inspection with a typed, budgeted result shared by terminal and JSON renderers. Preserve clique hyperedges and cap only pair materialization. |
| `conflict-row-provenance` | internal-api | Retain originating row, row side, and extraction rule while `build_clique_table()` creates a clique. Do not reconstruct provenance from the final adjacency. |
| `family-hypergraphs` | driver-local | Give packing, covering, GUB, cardinality, and knapsack groups the overlap/component/interface summaries currently available only for exact-one groups. |
| `activation-graph` | driver-local | Build the binary-controller-to-target graph from `variable_bound_t`, including fan-in/out, shared targets, encoded bounds, and bound-quality ratios. Do not assign semantic labels to controllers. |
| `implication-closure` | driver-local | Analyze the row-certified literal implication graph: SCCs, equivalence classes, forced/contradictory literals, and bounded closure or transitive reduction. Keep static and future probing arcs separate. |
| `matrix-separators` | driver-local | Add articulation rows/columns, bridges, biconnected components, high-degree hubs, and explicit block-angular interface scores to `decomposition_t`. |
| `variable-role-taxonomy` | driver-local | Record non-exclusive roles such as group member, controller, controlled response, mediator, interface, affine pivot, and repair candidate, with coverage and overlap summaries. |
| `network-certificate` | driver-local | Replace the current flow-like candidate with a checked signed-incidence/network certificate, and expose affine reconstruction expressions and dependency order separately. |
| `numerical-certificates` | driver-local | Retain the rows/columns behind aggregate numerical warnings, including normalized residuals and decimal-scaling admission evidence. |
| `refinement-stabilization` | driver-local | Refine until stable or a configured cap, report whether stabilization occurred, and expose the cap. Rename the user-facing mode from symmetry to refinement while retaining a compatibility alias if needed. |
| `automorphism-orbits` | internal-api | Add guarded DejaVu orbit diagnostics only after graph construction and unfiltered diagnostic results are exposed separately from solver exploitability thresholds. Validate exported generators. |
| `lp-structure-overlays` | driver-local | Analyze original and presolved LP relaxations, fractional-block subgraphs, active interfaces, objective localization, activation slack, and solution-dependent dual-weighted coupling. |
| `presolve-lineage` | internal-api | Expose ordered fixed/substituted/parallel-column records from PaPILO postsolve storage. Survivor maps alone cannot identify a reduction operation or all source rows. |
| `cut-aware-root-state` | solver-invasive | Add a dedicated root-cut observer and an explicit stop after the cut loop. A tree node limit is not a root-only diagnostic mode. |
| `probing-overlay` | solver-invasive | First create an immutable, copy-safe probing snapshot that retains infeasible probes, continuous and integer implications, reasons, budgets, and completeness. The current mutating cache is not a reporting API. |

Do not add a detector/plugin registry until multiple independently implemented
extensions demonstrate a common lifecycle that the existing
`model_analysis_t` pipeline cannot express cleanly.
