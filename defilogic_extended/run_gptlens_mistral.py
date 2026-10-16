"""Run the official GPTLens auditor--critic--ranker protocol with Mistral.

The prompts and ranker formula are imported from the GPTLens ``release`` branch.
Only the transport is changed from the retired OpenAI ChatCompletion client to
Mistral's OpenAI-compatible endpoint.  This is consequently labelled a
*GPTLens-protocol reproduction with Mistral*, not a reproduction of the
published GPT-4 numbers.  As GPTLens has no independent property validator,
this runner intentionally does not report CONVERA's RCVR metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
BENCH = Path(__file__).resolve().parent
GPTLENS_SRC = BENCH / "third_party" / "GPTLens" / "src"
MANIFEST = BENCH / "manifest.json"
RESULTS = BENCH / "results"
sys.path.insert(0, str(GPTLENS_SRC))
from prompts import (  # noqa: E402
    auditor_format_constrain,
    auditor_prompt,
    critic_format_constrain,
    critic_zero_shot_prompt,
    topk_prompt1,
    topk_prompt2,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_object(raw: str) -> dict[str, Any]:
    """Recover the outer JSON object without accepting non-JSON model content."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start < 0:
        return {}
    depth, quoted, escaped = 0, False, False
    for index, char in enumerate(cleaned[start:], start=start):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(cleaned[start : index + 1])
                    return parsed if isinstance(parsed, dict) else {}
                except json.JSONDecodeError:
                    return {}
    return {}


def recover_output_list(raw: str, kind: str) -> list[dict[str, Any]]:
    """Recover official GPTLens fields when a model emits literal code newlines.

    GPTLens asks the auditor to place an entire Solidity function in JSON.  Some
    models return literal line breaks inside that string, which makes the outer
    JSON invalid.  This recovery accepts only the named fields from the
    prescribed GPTLens schema and saves the raw response alongside the result.
    """
    outer = extract_object(raw)
    parsed = outer.get("output_list")
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]

    entries: list[dict[str, Any]] = []
    starts = list(re.finditer(r'"function_name"\s*:\s*"([^"]+)"', raw))
    for index, start in enumerate(starts):
        segment = raw[start.start() : starts[index + 1].start() if index + 1 < len(starts) else len(raw)]
        vulnerability = re.search(r'"vulnerability"\s*:\s*"(.*?)"\s*,', segment, re.DOTALL)
        if not vulnerability:
            continue
        item: dict[str, Any] = {
            "function_name": start.group(1),
            "vulnerability": vulnerability.group(1).strip(),
        }
        if kind == "auditor":
            code = re.search(r'"code"\s*:\s*"(.*?)"\s*,\s*"vulnerability"', segment, re.DOTALL)
            reason = re.search(r'"reason"\s*:\s*"(.*?)(?=\n\s*}\s*(?:,|\]))', segment, re.DOTALL)
            item["code"] = code.group(1).strip() if code else ""
            item["reason"] = reason.group(1).strip() if reason else ""
        else:
            criticism = re.search(r'"criticism"\s*:\s*"(.*?)(?=\n\s*"(?:correctness|severity|profitability)"|\n\s*})', segment, re.DOTALL)
            item["criticism"] = criticism.group(1).strip() if criticism else ""
            for field in ("correctness", "severity", "profitability"):
                score = re.search(rf'"{field}"\s*:\s*([0-9]+(?:\.[0-9]+)?)', segment)
                item[field] = float(score.group(1)) if score else 0.0
        entries.append(item)
    return entries


def call_mistral(prompt: str, temperature: float, max_tokens: int = 4000) -> str:
    key = os.environ.get("MISTRAL_API_KEY") or os.environ.get("mistral")
    if not key:
        raise RuntimeError("MISTRAL_API_KEY (or legacy mistral) is not set.")
    response = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": os.environ.get("MISTRAL_MODEL", "ministral-8b-latest"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=600,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def normalize_code(code: str) -> str:
    return " ".join(code.split())


def source_ready_cases(manifest: dict[str, Any], requested: set[str]) -> list[dict[str, Any]]:
    selected = [case for case in manifest["cases"] if not requested or case["id"] in requested]
    found = {case["id"] for case in selected}
    missing = requested - found
    if missing:
        raise ValueError(f"Unknown case ids: {', '.join(sorted(missing))}")
    ready = []
    for case in selected:
        status_path = BENCH / "sources" / case["id"] / "source_status.json"
        if status_path.exists() and load_json(status_path).get("ready"):
            ready.append(case)
    return ready


def case_code(case_id: str) -> str:
    parts = []
    for path in sorted((BENCH / "sources" / case_id / "source").rglob("*.sol")):
        parts.extend((f"// FILE: {path.name}", path.read_text(encoding="utf-8", errors="ignore")))
    if not parts:
        raise FileNotFoundError(f"No collected source for {case_id}")
    return "\n".join(parts)


def safe_score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def property_family(text: str) -> str:
    text = text.lower()
    if any(term in text for term in ("collateral", "solvency", "liquidation", "borrow")):
        return "collateral_solvency"
    if any(term in text for term in ("entitlement", "single-use", "single use", "repeated claim", "claim twice")):
        return "entitlement_consumption"
    if any(term in text for term in ("reward", "payout", "threshold", "accrued fee", "fee accrual")):
        return "reward_payout_boundedness"
    if any(term in text for term in ("rounding", "accounting", "conservation", "share supply")):
        return "value_accounting_conservation"
    if any(term in text for term in ("asset mismatch", "wrong asset", "token identity")):
        return "asset_identity_consistency"
    if any(term in text for term in ("state transition", "phase", "ordering")):
        return "state_transition_correctness"
    if any(term in text for term in ("onlyowner", "access control", "unauthor", "permission")):
        return "authorization_governance"
    return "unknown"


def evaluate(case: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    code = normalize_code(case_code(case["id"]))
    auditor_input = auditor_prompt + code + auditor_format_constrain + topk_prompt1.format(topk=3) + topk_prompt2
    auditor_raw = call_mistral(auditor_input, temperature=0.7)
    auditor = recover_output_list(auditor_raw, "auditor")

    listing = ""
    for item in auditor:
        listing += (
            f"function_name: {item.get('function_name', '')}\n"
            f"code: {item.get('code', '')}\n"
            f"vulnerability: {item.get('vulnerability', '')}\n"
            f"reason: {item.get('reason', '')}\n------------------\n"
        )
    critic_raw, critic = "", []
    if listing:
        critic_raw = call_mistral(critic_zero_shot_prompt + listing + critic_format_constrain, temperature=0.0)
        critic = recover_output_list(critic_raw, "critic")

    ranked = []
    for index, item in enumerate(critic):
        candidate = dict(item)
        if index < len(auditor):
            candidate.update({key: auditor[index].get(key, "") for key in ("reason", "code", "file_name")})
        candidate["final_score"] = 0.5 * safe_score(candidate.get("correctness")) + 0.25 * safe_score(candidate.get("severity")) + 0.25 * safe_score(candidate.get("profitability"))
        ranked.append(candidate)
    ranked.sort(key=lambda item: item["final_score"], reverse=True)
    save_json(output_dir / case["id"] / "auditor.json", {"raw": auditor_raw, "parsed": auditor})
    save_json(output_dir / case["id"] / "critic.json", {"raw": critic_raw, "parsed": critic})
    save_json(output_dir / case["id"] / "ranked.json", ranked)

    top = ranked[0] if ranked else {}
    text = " ".join(str(top.get(key, "")) for key in ("vulnerability", "reason", "criticism"))
    return {
        "case_id": case["id"],
        "incident_name": case["name"],
        "status": "ok",
        "finding_count": len(ranked),
        "top_score": top.get("final_score", 0.0),
        "detected": bool(ranked and safe_score(top.get("final_score")) >= 5.0),
        "predicted_property_family": property_family(text),
        "gold_property_family": case["gold_property_family"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--run-dir", default="")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    manifest = load_json(MANIFEST)
    cases = source_ready_cases(manifest, set(args.case))
    if not cases:
        raise RuntimeError("No source-ready cases selected.")
    output_dir = Path(args.run_dir).resolve() if args.run_dir else RESULTS / datetime.now().strftime("gptlens_mistral_%Y%m%d_%H%M%S")
    rows = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']}", flush=True)
        try:
            rows.append(evaluate(case, output_dir / "cases"))
        except Exception as exc:
            rows.append({"case_id": case["id"], "incident_name": case["name"], "status": "failed", "error": f"{type(exc).__name__}: {exc}", "detected": False})
    hits = sum(bool(row.get("detected")) for row in rows if row["status"] == "ok")
    summary = {
        "method": "GPTLens protocol reproduction with Mistral",
        "source_protocol": "official GPTLens release prompts and 0.5/0.25/0.25 ranker",
        "model": os.environ.get("MISTRAL_MODEL", "ministral-8b-latest"),
        "eligible_incidents": len(cases),
        "failed_cases": sum(row["status"] != "ok" for row in rows),
        "idr": {"hits": hits, "rate": hits / len(cases)},
        "rcvr": "N/A: GPTLens supplies critic scores but no independent validation evidence.",
        "rows": rows,
    }
    save_json(output_dir / "summary.json", summary)
    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
