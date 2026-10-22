#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify malformed and unsupported contracts cannot silently become models."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from run_tests import analyzer_command


FIXTURES = Path(__file__).resolve().parent
API = "rmm::device_scalar::set_value_async"


class ContractParser(unittest.TestCase):
    def analyze(self, mutate):
        document = copy.deepcopy(self.document)
        model = next(c for c in document["contracts"] if c["name"] == API)
        mutate(model)
        with tempfile.TemporaryDirectory(
            prefix="cuopt-contract-parser-"
        ) as directory:
            work = Path(directory)
            registry = work / "contracts.json"
            registry.write_text(json.dumps(document))
            source = work / "contract.cpp"
            source.write_text(
                "// expected-no-diagnostics\n"
                f'#include "{(FIXTURES / "apis.hpp").as_posix()}"\n'
                "void submit(rmm::device_scalar<float>& output, const float& input, "
                "cuda::stream_ref stream) { output.set_value_async(input, stream); }\n"
            )
            coverage = work / "coverage.json"
            command = analyzer_command(
                OPTIONS.clang, OPTIONS.plugin, registry, source, coverage
            )
            # Fatal schema diagnostics must reach stderr; Clang's -verify
            # consumer suppresses them while checking source annotations.
            for option in ("-verify", "-verify-ignore-unexpected=note"):
                index = command.index(option)
                del command[index - 1 : index + 1]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            report = (
                json.loads(coverage.read_text()) if coverage.exists() else None
            )
            return result, report

    def test_invalid_schema_values_fail_closed(self):
        cases = (
            lambda m: m.update(effect="unknown_effect"),
            lambda m: m.update(kind="unknown_kind"),
            lambda m: m.update(stream={"unknown_selector": 1}),
            lambda m: m.update(ignored_semantic_field=True),
            lambda m: m["borrows"][0].update(arg=-1),
            lambda m: m["borrows"][0].update(storage="unknown_storage"),
            lambda m: m["borrows"][0].update(access="unknown_access"),
        )
        for number, mutate in enumerate(cases):
            with self.subTest(case=number):
                result, _ = self.analyze(mutate)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "Invalid asynchronous lifetime contract", result.stderr
                )

    def test_unsupported_semantics_are_coverage_gaps(self):
        cases = (
            (
                lambda m: m.update(when={"arg_types": {"0": "new_selector"}}),
                "argument type selector",
            ),
            (
                lambda m: m.update(
                    return_borrow={"storage": "pointee", "access": "write"}
                ),
                "returned allocation borrow",
            ),
            (
                lambda m: m.update(ownership={"receiver": True}),
                "ownership",
            ),
            (
                lambda m: m["borrows"][0].update(
                    storage="container_data", device_only=True
                ),
                "device container allocation",
            ),
        )
        for number, (mutate, expected) in enumerate(cases):
            with self.subTest(case=number):
                result, report = self.analyze(mutate)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIsNotNone(report)
                self.assertTrue(
                    any(expected in reason for reason in report["incomplete"]),
                    report,
                )
                self.assertNotIn(API, report["modeled_calls"], report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clang", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument(
        "--contracts", type=Path, default=FIXTURES.parent / "contracts.json"
    )
    OPTIONS, remaining = parser.parse_known_args()
    ContractParser.document = json.loads(OPTIONS.contracts.read_text())
    unittest.main(argv=[sys.argv[0], *remaining])
