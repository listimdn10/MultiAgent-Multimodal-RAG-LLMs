"""Offline contract tests for the RQ5 judge runner (no provider calls)."""

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "experiment/rq5_report_quality/run_llm_judge.py"
spec = importlib.util.spec_from_file_location("rq5_llm_judge", SCRIPT)
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


def row(case_id, method, report="report"):
    return {"case_id": case_id, "method": method, "contract_context": "contract C {}",
            "identified_vulnerability": "unauthorized withdrawal", "report": report}


class RQ5JudgeTests(unittest.TestCase):
    def write_manifest(self, rows):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "reports.jsonl"
        path.write_text("\n".join(json.dumps(item) for item in rows) + "\n", encoding="utf-8")
        return path

    def test_manifest_requires_paired_identical_context(self):
        path = self.write_manifest([row("1", "CONVERA")])
        with self.assertRaisesRegex(ValueError, "exactly CONVERA, GPTLens"):
            judge.load_manifest(path)
        other = row("1", "GPTLens")
        other["contract_context"] = "different"
        path = self.write_manifest([row("1", "CONVERA"), other])
        with self.assertRaisesRegex(ValueError, "different contract_context"):
            judge.load_manifest(path)

    def test_prompt_does_not_expose_method_label(self):
        prompt = judge.render_prompt(row("1", "CONVERA", "Specific finding"))
        self.assertIn("Specific finding", prompt)
        self.assertNotIn("CONVERA", prompt)
        self.assertIn("Clarity (1-10)", prompt)

    def test_score_validation_and_overall_mean(self):
        payload = {criterion: value for criterion, value in
                   zip(judge.CRITERIA, (8, "7", 9))}
        for criterion in judge.CRITERIA:
            payload[f"{criterion}_justification"] = "Specific reason"
        scored = judge.parse_score(json.dumps(payload))
        self.assertEqual(scored["overall"], 8)
        payload["clarity"] = 11
        with self.assertRaisesRegex(ValueError, "between 1 and 10"):
            judge.parse_score(json.dumps(payload))

    def test_summary_excludes_incomplete_pairs(self):
        rows = [row("A", "CONVERA"), row("A", "GPTLens"),
                row("B", "CONVERA"), row("B", "GPTLens")]
        records = []
        for repeat in (1, 2):
            for method in judge.METHODS:
                records.append({"case_id": "A", "method": method, "repeat": repeat,
                                "status": "ok", "scores": {"clarity": 8, "completeness": 7,
                                                            "readability": 9, "overall": 8}})
        records.append({"case_id": "B", "method": "CONVERA", "repeat": 1,
                        "status": "ok", "scores": {"clarity": 10, "completeness": 10,
                                                    "readability": 10, "overall": 10}})
        summary = judge.summarize(records, rows, 2)
        self.assertEqual(summary["complete_case_ids"], ["A"])
        self.assertEqual(summary["scores_on_complete_method_intersection"]["CONVERA"]["mean"]["overall"], 8)
        pair = summary["pairwise_comparisons_on_matched_cases"]["CONVERA__vs__GPTLens"]
        self.assertEqual(pair["paired_case_count"], 1)
        self.assertEqual(pair["std_population_across_repeats"]["CONVERA"]["overall"], 0)

    def test_dry_run_writes_nothing(self):
        path = self.write_manifest([row("1", "CONVERA"), row("1", "GPTLens")])
        out = path.parent / "results"
        self.assertEqual(judge.main(["--input", str(path), "--output-dir", str(out), "--dry-run"]), 0)
        self.assertFalse(out.exists())

    def test_partial_four_method_and_single_method_inputs(self):
        path = self.write_manifest([row("1", "CONVERA"), row("1", "GPTLens"),
                                    row("2", "CONVERA")])
        methods = ("CONVERA", "GPTLens", "VulnHuntGPT", "GPTScan")
        with self.assertRaisesRegex(ValueError, "must contain exactly"):
            judge.load_manifest(path, methods)
        self.assertEqual(len(judge.load_manifest(path, methods, allow_missing=True)), 3)
        single = self.write_manifest([row("1", "CONVERA")])
        self.assertEqual(len(judge.load_manifest(single, ("CONVERA",))), 1)
        summary = judge.summarize([], judge.load_manifest(single, ("CONVERA",)), 3,
                                  ("CONVERA",))
        self.assertEqual(summary["complete_requested_method_cases"], 0)
        self.assertEqual(summary["pairwise_comparisons_on_matched_cases"], {})

    def test_openai_request_uses_structured_output_without_storing_response(self):
        response_body = {"status": "completed", "id": "resp_test", "usage": {"input_tokens": 10},
                         "output": [{"type": "message", "content": [
                             {"type": "output_text", "text": '{"clarity":8}'}]}]}
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["headers"] = request.headers
            captured["payload"] = json.loads(request.data)
            return io.BytesIO(json.dumps(response_body).encode("utf-8"))

        with patch.object(judge.urllib.request, "urlopen", side_effect=fake_urlopen):
            answer, metadata = judge.openai_generate("Judge this report", "gpt-6-luna",
                                                     "test-key", "low", 1200, 10, 0)
        self.assertEqual(answer, '{"clarity":8}')
        self.assertEqual(metadata["response_id"], "resp_test")
        self.assertEqual(captured["url"], "https://api.openai.com/v1/responses")
        self.assertFalse(captured["payload"]["store"])
        self.assertTrue(captured["payload"]["text"]["format"]["strict"])
        self.assertNotIn("test-key", json.dumps(captured["payload"]))


if __name__ == "__main__":
    unittest.main()
