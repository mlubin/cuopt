#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run hermetic analyzer regression fixtures; no CUDA toolkit or GPU is needed."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


FIXTURES = Path(__file__).resolve().parent
DEFAULT_CONTRACTS = FIXTURES.parent / "contracts.json"


def analyzer_command(clang, plugin, contracts, source, coverage):
    command = [
        str(clang),
        "--analyze",
        "-std=c++20",
        "-Xclang",
        "-load",
        "-Xclang",
        str(plugin),
        "-Xclang",
        "-analyzer-disable-checker=deadcode.DeadStores",
        "-Xclang",
        "-analyzer-checker=cuopt.AsyncLifetime",
        "-Xclang",
        "-analyzer-output=text",
        "-Xclang",
        "-analyzer-config",
        "-Xclang",
        f"cuopt.AsyncLifetime:Contracts={contracts}",
        "-Xclang",
        "-analyzer-config",
        "-Xclang",
        f"cfg-lifetime=true,cuopt.AsyncLifetime:Coverage={coverage}",
        "-Xclang",
        "-verify",
        "-Xclang",
        "-verify-ignore-unexpected=note",
        "-o",
        "/dev/null",
        str(source),
    ]
    for config in re.findall(
        r"analyzer-config\s*\{\{(.*?)\}\}", source.read_text()
    ):
        command[1:1] = ["-Xclang", "-analyzer-config", "-Xclang", config]
    return command


class AnalyzerFixtures(unittest.TestCase):
    """Each source specifies both required warnings and clean examples."""

    def run_fixture(self, source):
        with tempfile.TemporaryDirectory(
            prefix="cuopt-lifetime-test-"
        ) as directory:
            coverage = Path(directory) / "coverage.json"
            result = subprocess.run(
                analyzer_command(
                    OPTIONS.clang,
                    OPTIONS.plugin,
                    OPTIONS.contracts,
                    source,
                    coverage,
                ),
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            self.assertEqual(
                result.returncode, 0, result.stdout + result.stderr
            )
            self.assertTrue(
                coverage.is_file(),
                "Analyzer did not produce its coverage report",
            )
            report = json.loads(coverage.read_text())
            self.assertEqual(report["schema_version"], 1)
            self.assertGreater(report["analyses"], 0)
            self.assertEqual(report["cutoffs"], 0, report)
            expected = re.findall(
                r"expected-incomplete\s*\{\{(.*?)\}\}", source.read_text()
            )
            reasons = report["incomplete"]
            for pattern in expected:
                self.assertTrue(
                    any(re.search(pattern, reason) for reason in reasons),
                    report,
                )
            for reason in reasons:
                self.assertTrue(
                    any(re.search(pattern, reason) for pattern in expected),
                    report,
                )


def add_fixture_test(source):
    def test_fixture(self):
        self.run_fixture(source)

    setattr(AnalyzerFixtures, f"test_{source.stem}", test_fixture)


for fixture in sorted(FIXTURES.glob("*.cpp")):
    add_fixture_test(fixture)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clang", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    OPTIONS, unittest_args = parser.parse_known_args()
    for name in ("clang", "plugin", "contracts"):
        path = getattr(OPTIONS, name).resolve()
        if not path.is_file():
            parser.error(f"--{name} is not a file: {path}")
        setattr(OPTIONS, name, path)
    unittest.main(argv=[sys.argv[0], *unittest_args])
