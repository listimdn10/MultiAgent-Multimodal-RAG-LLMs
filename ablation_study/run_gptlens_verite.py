"""Reproduce the GPTLens auditor--critic--ranker protocol on VERITE.

The runner sends only the frozen Solidity source to the model.  Incident labels
are loaded solely after inference to aggregate address-level findings into the
30 incident-level IDR score.  This is a Mistral-backed protocol reproduction,
not the original GPT-4 implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = Path(__file__).resolve().parent / "inputs"
RESULTS_ROOT = Path(__file__).resolve().parent / "results"
sys.path.insert(0, str(ROOT / "defilogic_extended"))

from run_gptlens_mistral import (  # noqa: E402
    auditor_format_constrain,
    auditor_prompt,
    call_mistral,
    critic_format_constrain,
    critic_zero_shot_prompt,
    normalize_code,
    recover_output_list,
    safe_score,
    topk_prompt1,
    topk_prompt2,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_cases(requested: set[str]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    available: set[str] = set()
    for case_path in sorted(INPUT_ROOT.glob("*/case.json")):
        case = read_json(case_path)
        available.update({case["case_id"], case["incident_name"]})
        if requested and case["case_id"] not in requested and case["incident_name"] not in requested:
            continue
        rag_path = case_path.with_name("rag_output.json")
        if not rag_path.exists():
            continue
        # Keep the gold family out of the model-facing object.
        cases.append(
            {
                "case_id": case["case_id"],
                "incident_name": case["incident_name"],
                "gold_bug_family": case["gold_bug_family"],
                "code": read_json(rag_path)["code"],
            }
        )
    missing = requested - available
    if missing:
        raise ValueError(f"Unknown VERITE case or incident: {', '.join(sorted(missing))}")
    return cases


def evaluate(case: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    code = normalize_code(case["code"])
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

    ranked: list[dict[str, Any]] = []
    for index, item in enumerate(critic):
        candidate = dict(item)
        if index < len(auditor):
            candidate.update({key: auditor[index].get(key, "") for key in ("reason", "code", "file_name")})
        candidate["final_score"] = (
            0.5 * safe_score(candidate.get("correctness"))
            + 0.25 * safe_score(candidate.get("severity"))
            + 0.25 * safe_score(candidate.get("profitability"))
        )
        ranked.append(candidate)
    ranked.sort(key=lambda item: item["final_score"], reverse=True)

    case_dir = output_dir / case["case_id"]
    write_json(case_dir / "auditor.json", {"raw": auditor_raw, "parsed": auditor})
    write_json(case_dir / "critic.json", {"raw": critic_raw, "parsed": critic})
    write_json(case_dir / "ranked.json", ranked)

    top = ranked[0] if ranked else {}
    return {
        "case_id": case["case_id"],
        "incident_name": case["incident_name"],
        "gold_bug_family": case["gold_bug_family"],
        "status": "ok",
        "finding_count": len(ranked),
        "top_score": top.get("final_score", 0.0),
        "detected": bool(ranked and safe_score(top.get("final_score")) >= 5.0),
        "top_vulnerability": top.get("vulnerability", ""),
        "top_reason": top.get("reason", ""),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["incident_name"]].append(row)
    incident_rows = []
    for name, members in sorted(grouped.items()):
        successful = [row for row in members if row["status"] == "ok"]
        incident_rows.append(
            {
                "incident_name": name,
                "gold_bug_family": members[0]["gold_bug_family"],
                "num_address_cases": len(members),
                "failed_address_cases": len(members) - len(successful),
                "idr_hit": any(bool(row.get("detected")) for row in successful),
            }
        )
    hits = sum(row["idr_hit"] for row in incident_rows)
    return {
        "method": "GPTLens protocol reproduction with Mistral",
        "source_protocol": "official GPTLens prompts and 0.5/0.25/0.25 auditor--critic ranker",
        "metric_definition": {
            "IDR": "an incident has at least one address-level GPTLens finding with ranker score >= 5.0",
            "RCVR": "not automatically reported: GPTLens has no validator, and textual root-cause matching requires separate manual annotation",
        },
        "num_incidents": len(incident_rows),
        "num_address_cases": len(rows),
        "failed_address_cases": sum(row["status"] != "ok" for row in rows),
        "idr": {"hits": hits, "rate": hits / len(incident_rows) if incident_rows else 0.0},
        "rcvr": None,
        "incident_rows": incident_rows,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPTLens protocol reproduction on frozen VERITE inputs.")
    parser.add_argument("--case", action="append", default=[], help="VERITE case id or incident name; omit for all cases.")
    parser.add_argument("--run-dir", default="", help="Output directory; a timestamped directory is used when omitted.")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    cases = load_cases(set(args.case))
    if not cases:
        raise RuntimeError("No frozen VERITE cases were found.")
    output_dir = Path(args.run_dir).resolve() if args.run_dir else RESULTS_ROOT / datetime.now().strftime("gptlens_verite_mistral_%Y%m%d_%H%M%S")
    rows = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['case_id']}", flush=True)
        try:
            rows.append(evaluate(case, output_dir / "cases"))
        except Exception as exc:  # preserve partial runs for diagnosis and resumption
            rows.append(
                {
                    "case_id": case["case_id"],
                    "incident_name": case["incident_name"],
                    "gold_bug_family": case["gold_bug_family"],
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "detected": False,
                }
            )
    summary = summarize(rows)
    write_json(output_dir / "summary.json", summary)
    fields = sorted({key for row in rows for key in row})
    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
