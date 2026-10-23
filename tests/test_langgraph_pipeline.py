"""Offline smoke tests for LangGraph orchestration (no model/API calls)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import langgraph_pipeline as pipeline


class AuditGraphTests(unittest.TestCase):
    def test_all_stages_run_in_order_and_share_state(self):
        seen = []

        def node(stage):
            def run(state):
                seen.append(stage)
                if stage != "semantic":
                    previous = pipeline.STAGES[pipeline.STAGES.index(stage) - 1]
                    self.assertIn(previous, state)
                return {stage: {"stage": stage}}

            return run

        with patch.dict(pipeline.NODES, {stage: node(stage) for stage in pipeline.STAGES}):
            result = pipeline.build_audit_graph().invoke({})

        self.assertEqual(seen, list(pipeline.STAGES))
        self.assertEqual(result["consensus"], {"stage": "consensus"})

    def test_subgraph_and_invalid_stage_order(self):
        seen = []

        def node(stage):
            def run(state):
                seen.append(stage)
                return {stage: {}}

            return run

        with patch.dict(pipeline.NODES, {stage: node(stage) for stage in pipeline.STAGES}):
            pipeline.build_audit_graph(("fusion", "reason")).invoke({})
        self.assertEqual(seen, ["fusion", "reason"])
        for stages in [(), ("reason", "fusion"), ("reason", "reason"), ("unknown",)]:
            with self.subTest(stages=stages), self.assertRaises(ValueError):
                pipeline.build_audit_graph(stages)

    def test_artifact_must_be_fresh_and_valid(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rag_output.json"
            path.write_text('{"old": true}', encoding="utf-8")
            with patch.object(pipeline, "_artifact", return_value=path):
                with self.assertRaisesRegex(RuntimeError, "did not write a fresh"):
                    pipeline._run_and_read("rag", lambda: "ok")
                with self.assertRaisesRegex(RuntimeError, "rag failed"):
                    pipeline._run_and_read("rag", lambda: {"error": "missing input"})
                payload = pipeline._run_and_read(
                    "rag", lambda: path.write_text(json.dumps({"new": True}), encoding="utf-8")
                )
            self.assertEqual(payload, {"new": True})


if __name__ == "__main__":
    unittest.main()
