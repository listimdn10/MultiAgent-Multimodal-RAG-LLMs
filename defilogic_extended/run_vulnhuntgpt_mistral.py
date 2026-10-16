"""Reproduce the legacy VulnHunt-GPT workflow with Mistral and local retrieval.

The upstream repository (listimdn10/Vulnhunt_GPT, commit 592242f) expects a
private Pinecone index, OpenAI embeddings, and a Gemini preprocessor.  Those
external artifacts are not distributed with the code.  This runner retains the
repository's Markdown knowledge corpus and audit prompt, replaces the missing
Pinecone dependency with deterministic local lexical retrieval, and uses the
configured Mistral endpoint for the audit response.  Results must therefore be
reported as a legacy-code reproduction, not as the paper's original numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
BENCH = Path(__file__).resolve().parent
LEGACY = BENCH / "third_party" / "VulnHuntGPT"
KNOWLEDGE_ROOT = LEGACY / "data"
DEFI_MANIFEST = BENCH / "manifest.json"
VERITE_MANIFEST = ROOT / "rw_dataset" / "manifests" / "rw_manifest_business_logic.json"
RESULTS = BENCH / "results"

MAX_SOURCE_CHARS = 60_000
MAX_KNOWLEDGE_CHARS = 10_000
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def tokens(text: str) -> Counter[str]:
    return Counter(token.lower() for token in TOKEN_RE.findall(text))


def strip_comments(source: str) -> str:
    """Use deterministic stripping in place of the unavailable Gemini cleaner."""
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", source)


def read_knowledge() -> list[tuple[str, str, Counter[str]]]:
    docs: list[tuple[str, str, Counter[str]]] = []
    for path in sorted(KNOWLEDGE_ROOT.rglob("*.md")):
        if ".ipynb_checkpoints" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            docs.append((str(path.relative_to(LEGACY)), text, tokens(text)))
    if not docs:
        raise RuntimeError("No VulnHunt-GPT Markdown knowledge documents were found.")
    return docs


def retrieve(source: str, docs: list[tuple[str, str, Counter[str]]], top_k: int = 4) -> list[dict[str, str]]:
    query = tokens(source)
    ranked: list[tuple[float, str, str]] = []
    for name, text, document_tokens in docs:
        overlap = sum(min(query[token], document_tokens[token]) for token in query)
        norm = (sum(query.values()) * sum(document_tokens.values())) ** 0.5
        score = overlap / norm if norm else 0.0
        ranked.append((score, name, text))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [
        {"path": name, "text": text[:MAX_KNOWLEDGE_CHARS]}
        for _, name, text in ranked[:top_k]
    ]


def call_mistral(prompt: str) -> str:
    key = os.environ.get("MISTRAL_API_KEY") or os.environ.get("mistral")
    if not key:
        raise RuntimeError("MISTRAL_API_KEY (or legacy mistral) is not set.")
    response = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": os.environ.get("MISTRAL_MODEL", "ministral-8b-latest"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 2500,
        },
        timeout=600,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def extract_json(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass
    return {}


def build_prompt(source: str, retrieved: list[dict[str, str]]) -> str:
    knowledge = "\n\n".join(
        f"SOURCE: {item['path']}\n{item['text']}" for item in retrieved
    )
    return f"""You are VulnHunt-GPT. Analyze the Solidity smart contract below for concrete vulnerabilities.

Use the retrieved VulnHunt-GPT knowledge only as background. Do not report a vulnerability merely because a generic pattern appears: connect a reachable code mechanism to a realistic impact. The contract is a known real-world incident candidate, but no exploit description or label is supplied.

Return valid JSON only with exactly this schema:
{{
  "verdict": "vulnerable" | "no_confirmed_vulnerability",
  "findings": [
    {{"title": "...", "function": "...", "cause": "...", "evidence": "specific code-level evidence", "impact": "...", "remediation": "..."}}
  ]
}}

A vulnerable verdict requires at least one finding with non-empty function, cause, and code-level evidence. Preserve the original VulnHunt-GPT intent: describe vulnerabilities and remediation.

RETRIEVED VULNHUNT-GPT KNOWLEDGE:
{knowledge}

SOLIDITY SOURCE:
{source}
"""


def source_text(paths: list[Path]) -> str:
    chunks = []
    for path in paths:
        content = path.read_text(encoding="utf-8", errors="ignore")
        chunks.append(f"// FILE: {path.name}\n{content}")
    return "\n\n".join(chunks)


def defi_cases(requested: set[str]) -> list[dict[str, Any]]:
    manifest = read_json(DEFI_MANIFEST)
    cases = []
    for item in manifest["cases"]:
        if requested and item["id"] not in requested:
            continue
        source_dir = BENCH / "sources" / item["id"] / "source"
        files = sorted(source_dir.rglob("*.sol"))
        status = BENCH / "sources" / item["id"] / "source_status.json"
        if files and status.exists() and read_json(status).get("ready"):
            cases.append(
                {
                    "id": item["id"],
                    "incident_name": item["name"],
                    "source_files": files,
                    "gold_bug_family": item["gold_bug_family"],
                    "gold_property_family": item["gold_property_family"],
                }
            )
    return cases


def verite_cases(requested: set[str]) -> list[dict[str, Any]]:
    manifest = read_json(VERITE_MANIFEST)
    cases = []
    for incident in manifest["incidents"]:
        case_id = incident["incident_name"]
        if requested and case_id not in requested:
            continue
        selected = set(incident.get("selected_addresses", []))
        address = next((entry for entry in incident["addresses"] if entry["address"] in selected), None)
        if not address:
            continue
        source_dir = ROOT / address["address_dir"] / "sources"
        files = sorted(source_dir.rglob("*.sol"))
        if files:
            cases.append(
                {
                    "id": case_id,
                    "incident_name": case_id,
                    "source_files": files,
                    "gold_bug_family": incident["gold_bug_family"],
                    "address": address["address"],
                }
            )
    return cases


def evaluate(case: dict[str, Any], docs: list[tuple[str, str, Counter[str]]], output_dir: Path) -> dict[str, Any]:
    raw_source = source_text(case["source_files"])
    cleaned = strip_comments(raw_source)
    truncated = len(cleaned) > MAX_SOURCE_CHARS
    source = cleaned[:MAX_SOURCE_CHARS]
    retrieved = retrieve(source, docs)
    raw = call_mistral(build_prompt(source, retrieved))
    parsed = extract_json(raw)
    findings = parsed.get("findings") if isinstance(parsed.get("findings"), list) else []
    valid_findings = [
        item for item in findings
        if isinstance(item, dict)
        and all(str(item.get(field, "")).strip() for field in ("function", "cause", "evidence"))
    ]
    detected = parsed.get("verdict") == "vulnerable" and bool(valid_findings)
    payload = {
        "protocol": "legacy VulnHunt-GPT code/corpus with Mistral and local lexical retrieval",
        "input_truncated": truncated,
        "source_chars": len(source),
        "retrieved_documents": [item["path"] for item in retrieved],
        "raw_response": raw,
        "parsed_response": parsed,
    }
    write_json(output_dir / "cases" / case["id"] / "output.json", payload)
    return {
        "case_id": case["id"],
        "incident_name": case["incident_name"],
        "status": "ok",
        "detected": detected,
        "finding_count": len(valid_findings),
        "input_truncated": truncated,
        "gold_bug_family": case.get("gold_bug_family", ""),
        "gold_property_family": case.get("gold_property_family", ""),
    }


def main() -> None:
    global MAX_SOURCE_CHARS
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("defi", "verite"), required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-source-chars", type=int, default=MAX_SOURCE_CHARS)
    parser.add_argument("--run-dir", default="")
    args = parser.parse_args()
    MAX_SOURCE_CHARS = args.max_source_chars
    load_dotenv(ROOT / ".env")
    selected = set(args.case)
    cases = defi_cases(selected) if args.dataset == "defi" else verite_cases(selected)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise RuntimeError("No eligible source-ready cases selected.")
    docs = read_knowledge()
    output_dir = Path(args.run_dir).resolve() if args.run_dir else RESULTS / datetime.now().strftime("vulnhuntgpt_mistral_%Y%m%d_%H%M%S")
    rows = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']}", flush=True)
        try:
            rows.append(evaluate(case, docs, output_dir))
        except Exception as exc:
            rows.append({"case_id": case["id"], "incident_name": case["incident_name"], "status": "failed", "error": f"{type(exc).__name__}: {exc}", "detected": False})
    hits = sum(bool(row.get("detected")) for row in rows if row["status"] == "ok")
    summary = {
        "method": "VulnHunt-GPT legacy-code reproduction with Mistral/local retrieval",
        "legacy_repository": "https://github.com/listimdn10/Vulnhunt_GPT.git",
        "legacy_commit": "592242f",
        "adaptation": "The undistributed Pinecone index, OpenAI embeddings, and Gemini preprocessor are replaced by local lexical retrieval over the bundled Markdown corpus and a Mistral audit call.",
        "model": os.environ.get("MISTRAL_MODEL", "ministral-8b-latest"),
        "dataset": args.dataset,
        "eligible_incidents": len(cases),
        "failed_cases": sum(row["status"] != "ok" for row in rows),
        "idr": {"hits": hits, "rate": hits / len(cases)},
        "rows": rows,
    }
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
