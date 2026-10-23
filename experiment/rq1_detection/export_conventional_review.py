"""Export held-out labels and six-stage predictions for post-run human review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "excel_row", "gold_label", "exact_training_source_overlap",
    "rag_vuln_type", "fusion_label", "reason_vuln_type",
    "consensus_decision", "consensus_summary", "artifact_dir",
    "manual_predicted_label", "manual_correct", "review_notes",
)


def export(run_dir: Path, output: Path, require_complete: bool = False) -> int:
    summary = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    records = summary["results"]
    if require_complete and (len(records) != summary["selected_cases"] or any(row["status"] != "ok" for row in records)):
        raise ValueError("Run is not complete and fully successful; review export would be partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for record in records:
            if record["status"] != "ok":
                continue
            writer.writerow({field: record.get(field, "") for field in FIELDS})
    return sum(record["status"] == "ok" for record in records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    output = args.output or args.run_dir / "review_candidates.csv"
    count = export(args.run_dir, output, args.require_complete)
    print(f"Exported {count} successful cases to {output}; manual labels are intentionally blank.")


if __name__ == "__main__":
    main()
