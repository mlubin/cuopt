# Inspecting MPS structure before and after presolve

`mps_presolve_structure` is a small internal C++ driver that:

1. reads a free- or fixed-format MPS file;
2. summarizes the original constraint matrix;
3. applies cuOpt's host-side PaPILO presolver; and
4. summarizes the reduced matrix and reports the reductions.

The summary includes variable and constraint classes, bound classes, row and
column degree statistics, the densest named rows and columns, connected
components in the row-column bipartite graph, and the row-derived conflict
graph for binary literals. Conflict information includes base cliques, clique
extensions, edges, degrees, components, and the most-conflicted literals.
Connected matrix components expose independent blocks in the constraint
matrix.

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

Use `--fixed` when the input uses fixed-column MPS formatting:

```bash
cpp/build/tests/examples/mip/mps_presolve_structure path/to/model.mps --fixed
```

PaPILO presolve does not accept quadratic objectives or quadratic constraints,
and the host-only path does not perform cuOpt's GPU-side semi-continuous
reformulation. The driver reports a clear error for these inputs rather than
silently inspecting a different model.
