"""Run the complete CONVERA LangGraph pipeline from the repository root."""

from __future__ import annotations

import os
from time import perf_counter

from langgraph_pipeline import ROOT, build_audit_graph


def main() -> None:
    os.chdir(ROOT)
    started = perf_counter()
    result = build_audit_graph().invoke({})
    print(f"Full audit completed in {perf_counter() - started:.2f}s")
    print("Consensus:", result["consensus"])


if __name__ == "__main__":
    main()
