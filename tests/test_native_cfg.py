"""Offline smoke tests for the native Solidity-to-CFG path on Windows."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from embedding_agent import compile_runtime_bytecode, extract_cfg_embedding


ROOT = Path(__file__).resolve().parents[1]
SOURCE = """pragma solidity ^0.4.24;
contract CFGSmoke {
    uint public value;
    function set(uint x) public { value = x; }
}
"""


class NativeCFGTests(unittest.TestCase):
    def test_native_compiler_emits_runtime(self):
        bytecode, chosen = compile_runtime_bytecode(SOURCE)
        self.assertIn("CFGSmoke", chosen)
        self.assertGreater(len(bytecode), 100)

    def test_ethersolve_cfg_is_not_fallback(self):
        previous_cwd = Path.cwd()
        previous_setting = os.environ.get("CONVERA_REQUIRE_REAL_CFG")
        with tempfile.TemporaryDirectory(prefix="convera-cfg-test-") as directory:
            folder = Path(directory)
            shutil.copy2(ROOT / "EtherSolve.jar", folder / "EtherSolve.jar")
            (folder / "rag_output.json").write_text(
                json.dumps({"code": SOURCE, "Audit_report": "No issue."}),
                encoding="utf-8",
            )
            try:
                os.chdir(folder)
                os.environ["CONVERA_REQUIRE_REAL_CFG"] = "1"
                embedding = extract_cfg_embedding("rag_output.json")
                self.assertEqual(len(embedding), 64)
                self.assertTrue((folder / "report.json").is_file())
            finally:
                os.chdir(previous_cwd)
                if previous_setting is None:
                    os.environ.pop("CONVERA_REQUIRE_REAL_CFG", None)
                else:
                    os.environ["CONVERA_REQUIRE_REAL_CFG"] = previous_setting


if __name__ == "__main__":
    unittest.main()
