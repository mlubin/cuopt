# Asynchronous storage lifetime checker

This Clang Static Analyzer plugin tracks storage borrowed by asynchronous operations. API contracts identify the borrowed argument, access mode, stream, and completion rules. A stack value passed to `rmm::device_scalar::set_value_async`, for example, must remain alive and unchanged until the copy completes. The source's lifetime is independent of the device scalar that receives it.

The checker reports supported lifetime violations and separately records incomplete analysis. Unsupported contracts, unresolved stream/ownership relationships, and exploration cutoffs cannot establish safety. The contract registry is versioned in [contracts.json](contracts.json); its `unsupported` entries describe API families that still need models. Presence in that inventory does not mean an API is implemented by the checker.

Plain forwarding wrappers receive inferred summaries from their function bodies, so disabling inlining must not hide a borrowed argument. The regression fixtures also cover heap deallocation and default-allocator `std::vector` storage, including ownership transfers. Graph replay, owning RMM device-allocation identities, and device-wide synchronization without a current-device/stream-device relationship remain explicit coverage gaps.

## Build an analysis toolchain

Clang 23 or newer is required for the `check::LifetimeEnd` callback. Build and load the plugin with the same Clang release: analyzer plugin APIs are version-specific. This toolchain is separate from cuOpt's production compiler and pinned formatting tools.

The permanent toolchain pins live in the `async_lifetime_checks` group in [dependencies.yaml](../../../dependencies.yaml). The existing `ci/check_style.sh` job creates that separate environment and runs all three CTest gates: the analyzer fixtures, compilation-database driver tests, and contract-parser tests. This CPU regression gate tests the checker; analyzing cuOpt translation units also requires a configured cuOpt build as described below.

The following user-space environment was verified with Clang 23.1.4:

```bash
analysis_prefix="$PWD/cpp/build/async_lifetime/toolchain"
mamba create -y -p "$analysis_prefix" -c conda-forge \
  clang=23.1.4 clangdev=23.1.4 llvmdev=23.1.4 cmake ninja
"$analysis_prefix/bin/cmake" \
  -S cpp/tools/async_lifetime -B cpp/build/async_lifetime/checker \
  -G Ninja \
  -DCMAKE_CXX_COMPILER="$analysis_prefix/bin/clang++" \
  -DLLVM_DIR="$analysis_prefix/lib/cmake/llvm" \
  -DClang_DIR="$analysis_prefix/lib/cmake/clang"
"$analysis_prefix/bin/cmake" --build cpp/build/async_lifetime/checker
```

No GPU is needed to build the plugin or run the hermetic regression fixtures:

```bash
ctest --test-dir cpp/build/async_lifetime/checker --output-on-failure
```

Each fixture checks required diagnostics and the coverage JSON. Unexpected coverage gaps, missing reports, and analyzer exploration cutoffs fail the tests. The original stack-local weight initialization must warn, and the `cuopt::one_v<f_t>` constructor initialization must pass.

The analyzer must use `cfg-lifetime=true`; the driver and fixture runner enable it explicitly because Clang's default is false. Keep the core analyzer checkers enabled: `core.BuiltinFunctions` models path termination such as `__builtin_unreachable`, which affects lifetime analysis on CUDA error paths.

## Analyze a configured cuOpt build

Generate compilation commands from the existing build configuration:

```bash
.cuopt_env/bin/cmake -S cpp -B cpp/build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

Inspect the database before relying on it: a database containing only dependency sources provides no project coverage. The driver defaults to translation units under `cpp/src` and returns incomplete if none are present. Positional arguments select exact source paths and require corresponding database entries. To inspect a `.cuh` header, select a translation unit that includes and instantiates it.

```bash
python ci/check_async_lifetimes.py \
  --clang "$analysis_prefix/bin/clang++" \
  --plugin cpp/build/async_lifetime/checker/libCuoptAsyncLifetime.so \
  --contracts cpp/tools/async_lifetime/contracts.json \
  --compilation-database cpp/build/compile_commands.json \
  --output-dir cpp/build/async_lifetime/reports \
  cpp/src/mip_heuristics/diversity/population.cu
```

The driver preserves macro definitions, include ordering, C++ standard, and host GCC search paths. C++ commands without an explicit standard are incomplete because different compiler releases can have different defaults. It maps CUDA architecture/default-stream settings for Clang host analysis, records removed compilation-output/NVCC backend options, and rejects unmapped NVCC controls. Clang errors and ignored compilation options produce incomplete results. It never runs the database's shell command.

Use the same fetched CCCL headers as the cuOpt build. On the verified installation, those headers under `cpp/build/_deps/cccl-src` parsed successfully; the environment's different CCCL copy had incompatible deduction-guide annotations. Changing include order to suppress this error would analyze a different configuration.

Clang 23.1.4 warns that CUDA 13.3 is newer than its supported CUDA 13.2. The driver appends `-Wno-error=unknown-cuda-version` so the build's general `-Werror` setting does not turn that specific compatibility warning into an error; the warning remains visible. Host parsing and analyzer execution succeeded on the actual `weights.cuh` instantiations with the build's CCCL headers; this does not establish compatibility for every CUDA translation unit. The driver records each parse result and fails if any selected unit cannot be analyzed.

## Interpret results

The output directory contains a toolchain/plugin compatibility log, each translation unit's normalized command and diagnostics, checker coverage JSON, and an aggregate `summary.json`.

Exit statuses are:

| Status | Meaning |
| --- | --- |
| `0` | Every selected translation unit completed the supported analysis with no lifetime diagnostics. |
| `1` | Analysis completed and found lifetime violations. |
| `2` | Coverage is incomplete, including an empty selection/database, missing source, parse/plugin failure, missing report, unresolved operation, or exploration cutoff. Known violations remain in the summary. |

Completion applies to the selected translation units and configured API models. It is not a proof that arbitrary external code, uninstantiated templates, other build variants, or concurrent host threads are safe. Review the recorded scope and incomplete reasons alongside diagnostics.

The checker distinguishes copied values from retained storage: a kernel's scalar argument, memset fill value, or copied parameter descriptor does not have the same lifetime as memory referenced by a pointer inside it. Completion also requires the relevant stream/event relationship; finding an unrelated synchronization call is insufficient. In particular, `RAFT_CHECK_CUDA` synchronizes only in debug builds and must not be treated as a release-build completion barrier.
