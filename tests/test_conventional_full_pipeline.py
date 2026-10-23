"""Offline input tests for the conventional full-pipeline runner."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiment" / "rq1_detection"))
import run_conventional_full_pipeline as runner


class ConventionalRunnerTests(unittest.TestCase):
    def test_workbook_has_source_and_labels_but_runner_uses_only_source(self):
        cases = runner.read_cases(runner.DEFAULT_WORKBOOK)
        self.assertEqual(len(cases), 173)
        self.assertEqual(sum(case["label"] == "safe" for case in cases), 17)
        self.assertTrue(all(case["code"].strip() for case in cases))

    def test_explicit_gold_comments_are_removed_without_changing_code(self):
        source = """/* @source: SWC-101
@vulnerable_at_lines: 7 */
pragma solidity ^0.4.0;
// <yes> <report> ARITHMETIC
contract Example { uint x; }
"""
        clean = runner.strip_explicit_gold_comments(source)
        self.assertIn("pragma solidity ^0.4.0;", clean)
        self.assertIn("contract Example { uint x; }", clean)
        self.assertNotIn("SWC-101", clean)
        self.assertNotIn("ARITHMETIC", clean)


if __name__ == "__main__":
    unittest.main()
