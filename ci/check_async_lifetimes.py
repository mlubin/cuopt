#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the cuOpt async lifetime checker with explicit coverage accounting."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


CHECKER = "cuopt.AsyncLifetime"
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".cu"}
DROP_FLAGS = {
    "-c": "compile output",
    "-MD": "dependency output",
    "-MMD": "dependency output",
    "-MP": "dependency output",
    "-forward-unknown-to-host-compiler": "NVCC driver forwarding",
    "--expt-relaxed-constexpr": "Clang native CUDA constexpr support",
    "--expt-extended-lambda": "Clang native CUDA lambda support",
    "-lineinfo": "device debug output",
    "--generate-line-info": "device debug output",
    "-G": "device debug output",
    "--Werror=all-warnings": "NVCC diagnostic policy",
    "--diag-suppress=177": "NVCC diagnostic identifier",
    "-Werror=cross-execution-space-call": "NVCC diagnostic identifier",
    "-Wno-error=non-template-friend": "GCC-only diagnostic identifier",
    "-Xfatbin=-compress-all": "device binary output",
    "-Xfatbin=--compress-level=3": "device binary output",
    "--jobserver": "NVCC compile scheduling",
    "--threads=0": "NVCC compile scheduling",
    "--cudart=shared": "link runtime selection",
    "--cudart=static": "link runtime selection",
}
DROP_PAIRS = {"-o", "-MF", "-MT", "-MQ", "--output-file"}
PASS_PAIRS = {
    "-I",
    "-isystem",
    "-iquote",
    "-idirafter",
    "-include",
    "-imacros",
    "-D",
    "-U",
    "-std",
    "--std",
    "-stdlib",
    "--sysroot",
    "-isysroot",
    "--target",
    "-target",
    "-B",
    "--gcc-toolchain",
    "--cuda-path",
}


class IncompleteAnalysis(ValueError):
    """The input cannot be analyzed without losing coverage or semantics."""


def absolute(path, directory):
    path = Path(path)
    return (path if path.is_absolute() else directory / path).resolve()


def expand_responses(arguments, directory, active=()):
    expanded = []
    for argument in arguments:
        if not argument.startswith("@"):
            expanded.append(argument)
            continue
        path = absolute(argument[1:], directory)
        if path in active or len(active) >= 16:
            raise IncompleteAnalysis(f"cyclic/nested response file: {path}")
        try:
            contents = shlex.split(path.read_text())
        except (OSError, ValueError) as error:
            raise IncompleteAnalysis(
                f"cannot read response file {path}: {error}"
            ) from error
        expanded.extend(expand_responses(contents, directory, (*active, path)))
    return expanded


def normalize_entry(entry):
    """Keep frontend semantics and record every discarded backend argument."""
    directory = Path(entry["directory"]).resolve()
    source = absolute(entry["file"], directory)
    arguments = entry.get("arguments")
    if arguments is None:
        arguments = shlex.split(entry["command"])
    if not isinstance(arguments, list) or not all(
        isinstance(x, str) for x in arguments
    ):
        raise IncompleteAnalysis(
            "compilation arguments must be a string array"
        )
    arguments = expand_responses(arguments, directory)
    while arguments and Path(arguments[0]).name in {"ccache", "sccache"}:
        arguments.pop(0)
    if not arguments:
        raise IncompleteAnalysis("empty compiler command")
    compiler = arguments.pop(0)
    nvcc = Path(compiler).name == "nvcc"
    located = (
        shutil.which(compiler)
        if not Path(compiler).is_absolute()
        else compiler
    )
    compiler_path = absolute(located or compiler, directory)
    prefix = compiler_path.parent.parent
    kept, removed = [], []
    language = None
    host_compiler = None
    found_source = False
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        index += 1

        def operand():
            nonlocal index
            if index == len(arguments):
                raise IncompleteAnalysis(f"missing operand for {argument}")
            value = arguments[index]
            index += 1
            return value

        if argument in DROP_FLAGS:
            removed.append(
                {"argument": argument, "reason": DROP_FLAGS[argument]}
            )
        elif argument in DROP_PAIRS:
            removed.append(
                {
                    "argument": [argument, operand()],
                    "reason": "compile/dependency output",
                }
            )
        elif argument in {"-x", "--x"}:
            language = operand()
            if language == "cu":
                language = "cuda"
            if language not in {"cuda", "c", "c++"}:
                raise IncompleteAnalysis(
                    f"unsupported source language: {language}"
                )
        elif argument in {
            "-ccbin",
            "--compiler-bindir",
        } or argument.startswith(("-ccbin=", "--compiler-bindir=")):
            host_compiler = (
                argument.split("=", 1)[1] if "=" in argument else operand()
            )
            removed.append(
                {
                    "argument": [argument, host_compiler],
                    "reason": "mapped host compiler search path",
                }
            )
        elif argument == "-Xcompiler" or argument.startswith("-Xcompiler="):
            forwarded = (
                argument.split("=", 1)[1] if "=" in argument else operand()
            )
            # NVCC accepts comma-separated host options; feed each back through
            # this loop so GCC-only diagnostics and output options are handled.
            arguments[index:index] = [
                part
                for token in shlex.split(forwarded)
                for part in token.split(",")
            ]
        elif argument in {
            "-gencode",
            "--generate-code",
        } or argument.startswith(("-gencode=", "--generate-code=")):
            specification = (
                argument.split("=", 1)[1] if "=" in argument else operand()
            )
            match = re.fullmatch(
                r"arch=compute_([0-9]+[af]?),code=(?:\[(?:sm|compute)_[0-9]+[af]?(?:,(?:sm|compute)_[0-9]+[af]?)*\]|(?:sm|compute)_[0-9]+[af]?)",
                specification,
            )
            if not match:
                raise IncompleteAnalysis(
                    f"unsupported NVCC architecture: {specification}"
                )
            kept.append(f"--cuda-gpu-arch=sm_{match[1]}")
            removed.append(
                {
                    "argument": [argument, specification],
                    "reason": "mapped CUDA target architecture",
                }
            )
        elif argument.startswith("--default-stream="):
            mode = argument.split("=", 1)[1]
            if mode not in {"legacy", "per-thread"}:
                raise IncompleteAnalysis(f"unsupported default stream: {mode}")
            # --analyze does not claim the driver's codegen option. Pass it
            # to cc1, where it also defines CUDA_API_PER_THREAD_DEFAULT_STREAM.
            kept.extend(["-Xclang", f"-fgpu-default-stream={mode}"])
        elif argument in PASS_PAIRS:
            kept.extend([argument, operand()])
        elif argument.startswith("-"):
            # Reject unknown NVCC driver controls instead of silently removing
            # them. Other options reach Clang, whose errors remain fatal.
            if (
                nvcc
                and argument.startswith(("--", "-X", "-arch", "-code", "-rdc"))
                and not argument.startswith(
                    (
                        "--sysroot=",
                        "--target=",
                        "--std=",
                        "--gcc-toolchain=",
                        "--cuda-path=",
                        "--cuda-gpu-arch=",
                    )
                )
            ):
                raise IncompleteAnalysis(f"unmapped NVCC option: {argument}")
            kept.append(argument)
        elif absolute(argument, directory) == source and not found_source:
            found_source = True
        else:
            raise IncompleteAnalysis(
                f"unrecognized compiler input: {argument}"
            )
    if not found_source:
        raise IncompleteAnalysis("source is absent from compiler command")
    if language is None:
        language = (
            "cuda"
            if source.suffix == ".cu"
            else "c"
            if source.suffix == ".c"
            else "c++"
        )
    if language in {"c++", "cuda"} and not any(
        x in {"-std", "--std"} or x.startswith(("-std=", "--std="))
        for x in kept
    ):
        raise IncompleteAnalysis(
            "C++ standard is implicit in the original compiler command"
        )
    if language == "cuda":
        if not nvcc and not any(x.startswith("--cuda-path") for x in kept):
            raise IncompleteAnalysis(
                "CUDA command has no explicit toolkit path"
            )
        if nvcc:
            kept.extend(
                [f"--cuda-path={prefix}", "-isystem", str(prefix / "include")]
            )
        kept.append("--cuda-host-only")
    if not any(x.startswith("--gcc-toolchain") for x in kept):
        gcc_prefix = prefix
        if host_compiler:
            host_path = absolute(host_compiler, directory)
            gcc_prefix = (
                host_path.parent
                if host_path.is_dir()
                else host_path.parent.parent
            )
        if (gcc_prefix / "lib" / "gcc").is_dir():
            kept.append(f"--gcc-toolchain={gcc_prefix}")
    return directory, source, ["-x", language, *kept], removed


def inspect_coverage(path):
    """Validate checker output before interpreting an absence of diagnostics."""
    try:
        report = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise IncompleteAnalysis(
            f"missing/invalid coverage report: {error}"
        ) from error
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise IncompleteAnalysis("unsupported coverage schema")
    for field in ("analyses", "diagnostics", "cutoffs"):
        if type(report.get(field)) is not int or report[field] < 0:
            raise IncompleteAnalysis(f"invalid coverage field: {field}")
    for field in ("modeled_calls", "incomplete"):
        counts = report.get(field)
        if not isinstance(counts, dict) or any(
            type(x) is not int or x < 0 for x in counts.values()
        ):
            raise IncompleteAnalysis(f"invalid coverage field: {field}")
    reasons = []
    if report["analyses"] == 0:
        reasons.append("no function bodies analyzed")
    if report["cutoffs"]:
        reasons.append("analyzer exploration cutoff")
    reasons.extend(
        reason for reason, count in report["incomplete"].items() if count
    )
    return report, reasons


def check_toolchain(clang, plugin, output_dir):
    version = subprocess.run(
        [str(clang), "--version"], capture_output=True, text=True, check=False
    )
    match = re.search(r"clang version (\d+)\.\d+(?:\.\d+)?", version.stdout)
    if version.returncode or not match or int(match[1]) < 23:
        raise IncompleteAnalysis("Clang 23 or newer is required")
    probe = subprocess.run(
        [str(clang), "-cc1", "-load", str(plugin), "-analyzer-checker-help"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = probe.stdout + probe.stderr
    (output_dir / "toolchain.log").write_text(
        version.stdout + version.stderr + output
    )
    if probe.returncode or not re.search(rf"\b{re.escape(CHECKER)}\b", output):
        raise IncompleteAnalysis(
            "checker plugin cannot load with this Clang ABI; see toolchain.log"
        )
    return version.stdout.splitlines()[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "clang",
        "plugin",
        "contracts",
        "compilation-database",
        "output-dir",
    ):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument(
        "sources",
        nargs="*",
        type=Path,
        help="exact translation-unit paths; defaults to cpp/src entries",
    )
    options = parser.parse_args(argv)
    output_dir = options.output_dir.resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"cannot create report directory: {error}", file=sys.stderr)
        return 2
    summary = {
        "schema_version": 1,
        "status": "incomplete",
        "errors": [],
        "translation_units": [],
    }
    summary["compilation_database"] = str(
        options.compilation_database.resolve()
    )
    summary["scope"] = (
        "explicit translation units"
        if options.sources
        else "cpp/src entries in the supplied build configuration"
    )
    try:
        clang = Path(
            shutil.which(str(options.clang)) or options.clang
        ).resolve()
        plugin = options.plugin.resolve()
        contracts = options.contracts.resolve()
        for path in (clang, plugin, contracts):
            if not path.is_file():
                raise IncompleteAnalysis(
                    f"required file does not exist: {path}"
                )
        if "," in str(contracts) or "," in str(output_dir):
            raise IncompleteAnalysis(
                "analyzer-config paths cannot contain commas"
            )
        summary["clang_path"] = str(clang)
        summary["plugin"] = str(plugin)
        summary["contracts"] = str(contracts)
        summary["clang"] = check_toolchain(clang, plugin, output_dir)
        entries = json.loads(options.compilation_database.read_text())
        if not isinstance(entries, list) or not entries:
            raise IncompleteAnalysis("empty/invalid compilation database")
        requested = {path.resolve() for path in options.sources}
        selected = []
        found = set()
        for entry in entries:
            entry = {
                **entry,
                "directory": str(
                    absolute(
                        entry["directory"],
                        options.compilation_database.resolve().parent,
                    )
                ),
            }
            source = absolute(entry["file"], Path(entry["directory"]))
            choose = (
                source in requested
                if requested
                else source.is_relative_to(REPO_ROOT / "cpp" / "src")
            )
            if choose and source.suffix in SOURCE_SUFFIXES:
                selected.append(entry)
                found.add(source)
        missing = requested - found
        if missing:
            summary["errors"].append(
                "sources missing from compilation database: "
                + ", ".join(str(x) for x in sorted(missing))
            )
        if not selected:
            raise IncompleteAnalysis("no project translation units selected")
        for index, entry in enumerate(selected):
            unit = {
                "source": entry["file"],
                "status": "incomplete",
                "reasons": [],
            }
            summary["translation_units"].append(unit)
            try:
                directory, source, arguments, removed = normalize_entry(entry)
                tag = f"{index:04d}-{source.stem}-{hashlib.sha256(str(source).encode()).hexdigest()[:12]}"
                coverage = output_dir / f"{tag}.coverage.json"
                diagnostics = output_dir / f"{tag}.log"
                # A report left by a prior invocation must never stand in for a
                # checker that failed to run during this invocation.
                coverage.unlink(missing_ok=True)
                command = [
                    str(clang),
                    "--analyze",
                    *arguments,
                    "-Wno-error=unknown-cuda-version",
                    "-Xclang",
                    "-analyzer-output",
                    "-Xclang",
                    "plist-multi-file",
                    "-Xclang",
                    "-load",
                    "-Xclang",
                    str(plugin),
                    "-Xclang",
                    f"-analyzer-checker={CHECKER}",
                    "-Xclang",
                    "-analyzer-config",
                    "-Xclang",
                    f"cfg-lifetime=true,{CHECKER}:Contracts={contracts},{CHECKER}:Coverage={coverage}",
                    "-o",
                    str(output_dir / f"{tag}.plist"),
                    str(source),
                ]
                unit.update(
                    {
                        "source": str(source),
                        "directory": str(directory),
                        "command": command,
                        "removed_arguments": removed,
                        "coverage_file": str(coverage),
                        "diagnostics_file": str(diagnostics),
                    }
                )
                print(f"Analyzing {source}", flush=True)
                result = subprocess.run(
                    command,
                    cwd=directory,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                    check=False,
                )
                diagnostics.write_text(result.stdout)
                unit["exit_code"] = result.returncode
                if result.stdout:
                    print(result.stdout, end="", file=sys.stderr)
                if result.returncode:
                    unit["reasons"].append(
                        f"Clang failed with exit status {result.returncode}"
                    )
                if re.search(
                    r"unknown warning option|argument unused during compilation|ignoring unknown|unsupported option",
                    result.stdout,
                ):
                    unit["reasons"].append(
                        "compiler ignored/does not support a compilation option"
                    )
                report, reasons = inspect_coverage(coverage)
                unit["coverage"] = report
                unit["reasons"].extend(reasons)
                if not unit["reasons"]:
                    unit["status"] = (
                        "violations" if report["diagnostics"] else "complete"
                    )
            except (
                IncompleteAnalysis,
                OSError,
                ValueError,
                KeyError,
                TypeError,
            ) as error:
                unit["reasons"].append(str(error))
    except (
        IncompleteAnalysis,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        summary["errors"].append(str(error))
    units = summary["translation_units"]
    incomplete = (
        bool(summary["errors"])
        or not units
        or any(unit["status"] == "incomplete" for unit in units)
    )
    violations = sum(
        unit.get("coverage", {}).get("diagnostics", 0) for unit in units
    )
    summary["diagnostics"] = violations
    summary["status"] = (
        "incomplete"
        if incomplete
        else "violations"
        if violations
        else "complete"
    )
    try:
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
    except OSError as error:
        print(f"cannot write analysis summary: {error}", file=sys.stderr)
        return 2
    print(
        f"Async lifetime analysis: {summary['status']}; {len(units)} translation units; {violations} diagnostics. Report: {output_dir / 'summary.json'}"
    )
    for error in summary["errors"]:
        print(error, file=sys.stderr)
    for unit in units:
        for reason in unit["reasons"]:
            print(f"{unit['source']}: incomplete: {reason}", file=sys.stderr)
    return 2 if incomplete else 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
