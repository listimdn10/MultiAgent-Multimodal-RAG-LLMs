"""Print evidence-bearing baseline findings for manual incident-level RCVR review.

This utility does not assign RCVR credit. A reviewer must compare each finding
with the benchmark mechanism (DeFi) or coarse incident family (VERITE), and
check that the same mechanism is supported by the quoted code-level evidence.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFI_GOLD = {
    case["id"]: case
    for case in json.loads((ROOT / "defilogic_extended" / "manifest.json").read_text(encoding="utf-8"))["cases"]
}


def shorten(value: object, limit: int) -> str:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    return " ".join(str(value or "").split())[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory with resolved_summary.json")
    parser.add_argument("--max-chars", type=int, default=600)
    parser.add_argument("--incident", help="Review only this incident name")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    summary = json.loads((run_dir / "resolved_summary.json").read_text(encoding="utf-8"))
    job = summary["job"]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in summary["rows"]:
        grouped[row["incident_name"]].append(row)
    print(f"# {job}, repeat {summary['repeat']}: detected incidents only")
    for name, rows in sorted(grouped.items()):
        if args.incident and name != args.incident:
            continue
        detected = [row for row in rows if row.get("detected")]
        if not detected:
            continue
        first = rows[0]
        print(f"\n## {name}")
        if job.endswith("_defi"):
            gold = DEFI_GOLD[first["case_id"]]
            print(f"Gold family: {gold['gold_bug_family']}; property: {gold['gold_property_family']}")
            print(f"Gold mechanism: {gold['gold_root_cause']}")
        else:
            print(f"Gold coarse family: {first['gold_bug_family']}")
        for row in detected:
            artifact_dir = Path(summary["artifact_dirs"][row["case_id"]])
            print(f"\n- Case: {row['case_id']} | artifact: {artifact_dir}")
            if job.startswith("gptlens"):
                ranked = json.loads((artifact_dir / "ranked.json").read_text(encoding="utf-8"))
                for index, item in enumerate((item for item in ranked if item.get("final_score", 0) >= 5), 1):
                    print(f"  {index}. score={item['final_score']}; function={item.get('function_name', '')}; title={item.get('vulnerability', '')}")
                    print(f"     Cause: {shorten(item.get('reason'), args.max_chars)}")
                    print(f"     Code: {shorten(item.get('code'), 300)}")
            else:
                payload = json.loads((artifact_dir / "output.json").read_text(encoding="utf-8"))
                findings = payload.get("parsed_response", {}).get("findings", [])
                for index, item in enumerate(findings, 1):
                    if not isinstance(item, dict) or not all(str(item.get(key, "")).strip() for key in ("function", "cause", "evidence")):
                        continue
                    print(f"  {index}. function={item.get('function', '')}; title={item.get('title', '')}")
                    print(f"     Cause: {shorten(item.get('cause'), args.max_chars)}")
                    print(f"     Evidence: {shorten(item.get('evidence'), 300)}")


if __name__ == "__main__":
    main()
