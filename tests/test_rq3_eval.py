"""Offline checks for the RQ3 adapter evaluation runner."""

import json
import tempfile
import unittest
from pathlib import Path

from experiment.rq3_slm_rag.eval_rq3 import (
    load_jsonl, load_smartbugs, make_prompt, parse_label, score, select_balanced,
)


class Rq3EvalTests(unittest.TestCase):
    def test_exact_json_label_only(self):
        labels = ["arithmetic", "reentrancy"]
        self.assertEqual(parse_label('{"category":"arithmetic"}', labels), "arithmetic")
        self.assertEqual(parse_label('```json\n{"category":"reentrancy"}\n```', labels), "reentrancy")
        self.assertEqual(parse_label("I think it is arithmetic", labels), "__invalid__")
        self.assertEqual(parse_label('{"category":"other"}', labels), "__invalid__")

    def test_smartbugs_excludes_multilabel_contracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "single.sol").write_text("contract A {}", encoding="utf-8")
            (root / "multi.sol").write_text("contract B {}", encoding="utf-8")
            entries = [
                {"path": "single.sol", "vulnerabilities": [{"category": "arithmetic"}]},
                {"path": "multi.sol", "vulnerabilities": [
                    {"category": "arithmetic"}, {"category": "reentrancy"}
                ]},
            ]
            (root / "vulnerabilities.json").write_text(json.dumps(entries), encoding="utf-8")
            self.assertEqual([item["id"] for item in load_smartbugs(root)], ["single.sol"])

    def test_jsonl_requires_unique_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "eval.jsonl"
            line = json.dumps({"id": "1", "code": "contract A {}", "label": "arithmetic"})
            source.write_text(line + "\n" + line + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_jsonl(source)

    def test_prompt_does_not_take_gold_label(self):
        prompt = make_prompt("contract A {}", ["arithmetic", "reentrancy"], "")
        self.assertIn("contract A {}", prompt)
        self.assertIn("(none)", prompt)

    def test_invalid_output_is_counted_as_wrong(self):
        rows = [
            {"gold": "arithmetic", "predicted": "arithmetic"},
            {"gold": "reentrancy", "predicted": "__invalid__"},
        ]
        result = score(rows, ["arithmetic", "reentrancy"])
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["invalid_outputs"], 1)

    def test_small_limit_spans_labels(self):
        samples = [
            {"id": "a1", "label": "arithmetic"},
            {"id": "a2", "label": "arithmetic"},
            {"id": "r1", "label": "reentrancy"},
        ]
        selected = select_balanced(samples, 2)
        self.assertEqual({row["label"] for row in selected}, {"arithmetic", "reentrancy"})


if __name__ == "__main__":
    unittest.main()
