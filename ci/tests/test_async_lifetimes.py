#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise compilation fidelity and fail-closed analyzer coverage."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_async_lifetimes as driver  # noqa: E402


class NormalizeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = self.directory / "source.cu"
        self.source.write_text("void kernel();\n")

    def entry(self, *arguments):
        return {
            "directory": str(self.directory),
            "file": str(self.source),
            "arguments": ["/opt/cuda/bin/nvcc", *arguments, str(self.source)],
        }

    def test_preserves_defines_includes_standard_and_maps_stream_semantics(
        self,
    ):
        entry = self.entry(
            "-DFEATURE=1",
            "-I",
            "first",
            "-isystem",
            "second",
            "-UFEATURE",
            "-std=c++20",
            "--default-stream=per-thread",
            "--generate-code=arch=compute_120a,code=[sm_120a]",
            "-Xcompiler=-fPIC,-Wall",
            "--expt-extended-lambda",
            "-o",
            "out.o",
        )
        _, _, arguments, removed = driver.normalize_entry(entry)
        self.assertEqual(
            arguments[:12],
            [
                "-x",
                "cuda",
                "-DFEATURE=1",
                "-I",
                "first",
                "-isystem",
                "second",
                "-UFEATURE",
                "-std=c++20",
                "-Xclang",
                "-fgpu-default-stream=per-thread",
                "--cuda-gpu-arch=sm_120a",
            ],
        )
        self.assertIn("-fPIC", arguments)
        self.assertIn("-Wall", arguments)
        self.assertIn("--cuda-host-only", arguments)
        self.assertTrue(removed)

    def test_unknown_nvcc_control_is_not_discarded(self):
        with self.assertRaisesRegex(
            driver.IncompleteAnalysis, "unmapped NVCC"
        ):
            driver.normalize_entry(self.entry("--use_fast_math"))

    def test_malformed_architecture_is_not_guessed(self):
        for value in ("bogus", "arch=compute_120a,code=[bogus]"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(
                    driver.IncompleteAnalysis, "architecture"
                ),
            ):
                driver.normalize_entry(self.entry(f"--generate-code={value}"))

    def test_original_gcc_search_path_is_preserved(self):
        prefix = self.directory / "original-environment"
        (prefix / "lib" / "gcc").mkdir(parents=True)
        entry = self.entry("-std=c++20")
        entry["arguments"][0] = str(prefix / "bin" / "nvcc")
        _, _, arguments, _ = driver.normalize_entry(entry)
        self.assertIn(f"--gcc-toolchain={prefix}", arguments)
        entry = self.entry("-std=c++20", "--gcc-toolchain=/another/compiler")
        _, _, arguments, _ = driver.normalize_entry(entry)
        self.assertEqual(
            [x for x in arguments if x.startswith("--gcc-toolchain")],
            ["--gcc-toolchain=/another/compiler"],
        )

    def test_response_file_and_quoted_command(self):
        response = self.directory / "options file.rsp"
        response.write_text('-DNAME="hello world" -I "include directory"')
        entry = self.entry("-std=c++20", f"@{response}")
        _, _, arguments, _ = driver.normalize_entry(entry)
        self.assertIn("-DNAME=hello world", arguments)
        self.assertIn("include directory", arguments)
        entry["command"] = '/usr/bin/c++ -std=c++20 -c "source.cu"'
        del entry["arguments"]
        with self.assertRaisesRegex(driver.IncompleteAnalysis, "toolkit path"):
            driver.normalize_entry(entry)

    def test_response_cycle_is_incomplete(self):
        response = self.directory / "cycle.rsp"
        response.write_text("@cycle.rsp")
        with self.assertRaisesRegex(driver.IncompleteAnalysis, "cyclic"):
            driver.normalize_entry(self.entry("@cycle.rsp"))

    def test_missing_source_operand_is_rejected(self):
        entry = self.entry()
        entry["arguments"].pop()
        with self.assertRaisesRegex(driver.IncompleteAnalysis, "absent"):
            driver.normalize_entry(entry)

    def test_implicit_cpp_standard_is_incomplete(self):
        with self.assertRaisesRegex(
            driver.IncompleteAnalysis, r"C\+\+ standard is implicit"
        ):
            driver.normalize_entry(self.entry())


class DriverStatusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = self.directory / "sample.cpp"
        self.source.write_text("void example() {}\n")
        self.compiler = self.directory / "clang"
        self.plugin = self.directory / "plugin.so"
        self.contracts = self.directory / "contracts.json"
        for path in (self.compiler, self.plugin, self.contracts):
            path.write_text("")
        self.database = self.directory / "compile_commands.json"
        self.database.write_text(
            json.dumps(
                [
                    {
                        "directory": str(self.directory),
                        "file": str(self.source),
                        "arguments": [
                            "/usr/bin/c++",
                            "-std=c++20",
                            "-c",
                            str(self.source),
                        ],
                    }
                ]
            )
        )
        self.report = {
            "schema_version": 1,
            "analyses": 1,
            "diagnostics": 0,
            "cutoffs": 0,
            "modeled_calls": {},
            "incomplete": {},
        }
        self.produce_report = True
        self.returncode = 0
        self.log = ""
        self.help_output = driver.CHECKER

    def fake_run(self, command, **kwargs):
        if "--version" in command:
            return subprocess.CompletedProcess(
                command, 0, "clang version 23.1.4\n", ""
            )
        if "-analyzer-checker-help" in command:
            return subprocess.CompletedProcess(
                command, 0, self.help_output, ""
            )
        for argument in command:
            if argument.startswith("cfg-lifetime=true,"):
                path = argument.split(f"{driver.CHECKER}:Coverage=", 1)[1]
                if self.produce_report:
                    Path(path).write_text(json.dumps(self.report))
        return subprocess.CompletedProcess(command, self.returncode, self.log)

    def run_driver(self, sources=None):
        argv = [
            "--clang",
            str(self.compiler),
            "--plugin",
            str(self.plugin),
            "--contracts",
            str(self.contracts),
            "--compilation-database",
            str(self.database),
            "--output-dir",
            str(self.directory / "reports"),
        ]
        argv.extend(
            str(x) for x in ([self.source] if sources is None else sources)
        )
        with (
            patch.object(driver.subprocess, "run", self.fake_run),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            status = driver.main(argv)
        report = json.loads(
            (self.directory / "reports" / "summary.json").read_text()
        )
        return status, report

    def test_complete_empty_diagnostics(self):
        status, report = self.run_driver()
        self.assertEqual(status, 0)
        self.assertEqual(report["status"], "complete")

    def test_lifetime_violation_has_distinct_exit_status(self):
        self.report["diagnostics"] = 1
        self.assertEqual(self.run_driver()[0], 1)

    def test_uncovered_calls_are_incomplete_even_with_violations(self):
        self.report["diagnostics"] = 1
        self.report["incomplete"] = {"unsupported contract": 1}
        self.assertEqual(self.run_driver()[0], 2)

    def test_cutoff_and_zero_analysis_are_incomplete(self):
        for field in ("cutoffs", "analyses"):
            with self.subTest(field=field):
                self.report[field] = 1 if field == "cutoffs" else 0
                self.assertEqual(self.run_driver()[0], 2)
                self.report[field] = 0 if field == "cutoffs" else 1

    def test_parse_error_is_incomplete_even_if_report_exists(self):
        self.returncode = 1
        self.assertEqual(self.run_driver()[0], 2)

    def test_ignored_compiler_argument_is_incomplete(self):
        self.log = (
            "warning: argument unused during compilation: '-some-flag'\n"
        )
        self.assertEqual(self.run_driver()[0], 2)

    def test_missing_report_cannot_reuse_previous_report(self):
        self.assertEqual(self.run_driver()[0], 0)
        self.produce_report = False
        self.assertEqual(self.run_driver()[0], 2)

    def test_empty_database_no_project_units_and_missing_selection(self):
        self.assertEqual(self.run_driver(sources=[])[0], 2)
        self.assertEqual(
            self.run_driver(sources=[self.directory / "missing.cu"])[0], 2
        )
        self.database.write_text("[]")
        self.assertEqual(self.run_driver()[0], 2)

    def test_incompatible_plugin_is_incomplete(self):
        self.help_output = "plugin is incompatible with this version of Clang"
        self.assertEqual(self.run_driver()[0], 2)


if __name__ == "__main__":
    unittest.main()
