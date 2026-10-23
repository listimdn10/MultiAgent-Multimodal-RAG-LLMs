"""Deterministic LangGraph orchestration for the CONVERA audit stages.

Nodes call the existing analysis implementations exactly once. Artifacts keep
their historical names so benchmark readers and downstream scripts continue
to work. Model loading is deferred until a node actually runs.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
STAGES = ("semantic", "rag", "embedding", "fusion", "reason", "consensus")
ARTIFACTS = {
    "semantic": "input.json",
    "rag": "rag_output.json",
    "embedding": "parser_output.json",
    "fusion": "fusion_output_agent.json",
    "reason": "explainer_output.json",
    "consensus": "consensus_output.json",
}


class AuditState(TypedDict, total=False):
    semantic: dict[str, Any]
    rag: dict[str, Any]
    embedding: dict[str, Any]
    fusion: dict[str, Any]
    reason: dict[str, Any]
    consensus: dict[str, Any]


def _artifact(stage: str) -> Path:
    return ROOT / ARTIFACTS[stage]


def _run_and_read(stage: str, operation: Any) -> dict[str, Any]:
    path = _artifact(stage)
    old_mtime = path.stat().st_mtime_ns if path.exists() else None
    old_digest = hashlib.sha256(path.read_bytes()).digest() if path.exists() else None
    result = operation()
    if isinstance(result, str) and result.lower().startswith("error"):
        raise RuntimeError(f"{stage} failed: {result}")
    if isinstance(result, dict) and result.get("error"):
        raise RuntimeError(f"{stage} failed: {result['error']}")
    if not path.exists():
        raise RuntimeError(f"{stage} did not write a fresh {path.name}")
    if path.stat().st_mtime_ns == old_mtime and hashlib.sha256(path.read_bytes()).digest() == old_digest:
        raise RuntimeError(f"{stage} did not write a fresh {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return payload


def semantic_node(state: AuditState) -> AuditState:
    from semantic_agent import analyze_functional_semantics

    sample = ROOT / "contracts" / "sample.sol"
    if not sample.exists():
        sample = ROOT / "sample.sol"
    code = sample.read_text(encoding="utf-8")
    if not code.strip():
        raise ValueError(f"Empty Solidity source: {sample}")
    semantic = analyze_functional_semantics(code)
    payload = {"code": code, "functional_semantic": semantic}
    _artifact("semantic").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"semantic": payload}


def rag_node(state: AuditState) -> AuditState:
    from rag_agent import rag_tool

    payload = _run_and_read("rag", rag_tool._run)
    if not str(payload.get("Audit_report", "")).strip():
        raise RuntimeError("RAG stage produced no audit report")
    return {"rag": payload}


def embedding_node(state: AuditState) -> AuditState:
    from embedding_agent import embedding_tool

    return {"embedding": _run_and_read("embedding", embedding_tool._run)}


def fusion_node(state: AuditState) -> AuditState:
    from agent_fusion import build_fusion_tool

    return {"fusion": _run_and_read("fusion", build_fusion_tool()._run)}


def reason_node(state: AuditState) -> AuditState:
    from explainer import build_explainer_tool

    return {"reason": _run_and_read("reason", build_explainer_tool()._run)}


def consensus_node(state: AuditState) -> AuditState:
    from consensus_agent import build_consensus_tool

    tool = build_consensus_tool()
    payload = _run_and_read(
        "consensus",
        lambda: tool._run("contracts/sample.sol", "rag_output.json", "explainer_output.json"),
    )
    if payload.get("decision") == "FAILED":
        raise RuntimeError(payload.get("reasoning", "Consensus model failed"))
    return {"consensus": payload}


NODES = {
    "semantic": semantic_node,
    "rag": rag_node,
    "embedding": embedding_node,
    "fusion": fusion_node,
    "reason": reason_node,
    "consensus": consensus_node,
}


def build_audit_graph(stages: tuple[str, ...] = STAGES):
    """Compile a sequential graph for the requested audit stages."""
    if not stages or len(set(stages)) != len(stages) or any(stage not in NODES for stage in stages):
        raise ValueError(f"Invalid audit stages: {stages!r}")
    if tuple(sorted(stages, key=STAGES.index)) != tuple(stages):
        raise ValueError("Audit stages must follow pipeline order")
    graph = StateGraph(AuditState)
    for stage in stages:
        graph.add_node(stage, NODES[stage])
    graph.add_edge(START, stages[0])
    for earlier, later in zip(stages, stages[1:]):
        graph.add_edge(earlier, later)
    graph.add_edge(stages[-1], END)
    return graph.compile()
