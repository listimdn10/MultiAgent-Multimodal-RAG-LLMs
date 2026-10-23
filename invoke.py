"""Run the Bi-modal semantic -> RAG -> embedding LangGraph flow."""

from __future__ import annotations

import os
from pathlib import Path

from langgraph_pipeline import ROOT, build_audit_graph


def main() -> None:
    os.chdir(ROOT)  # Existing analysis tools use project-relative artifact paths.
    sample = Path("contracts/sample.sol")
    if not sample.is_file():
        raise FileNotFoundError(f"Missing contract source: {sample}")
    result = build_audit_graph(("semantic", "rag", "embedding")).invoke({})
    print("Bi-modal flow completed:", {name: list(data) for name, data in result.items()})


if __name__ == "__main__":
    main()
