"""Conservatively score GPTScan's VERITE root-cause recall from saved outputs.

VERITE supplies incident-level bug families, not precise root-cause annotations.
The mapping below credits only a specific GPTScan rule whose causal description
matches a gold family and whose reported code range contains relevant source.
The broad ``logical_bug`` family is intentionally not inferred from a generic
finding. The resulting RCVR is a reproducible conservative lower bound.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


RULE_FAMILY_MAP = {
    "price-manipulation": {"miscalculation", "price_manipulation"},
}

RULE_SOURCE_CUES = {
    "price-manipulation": ("price", "reserve", "rate", "value"),
}

RULE_TITLES = {"price-manipulation": "MWE-200: Insecure LP Token Value Calculation"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def has_grounded_location(finding: dict[str, Any], source_root: Path, code: str) -> bool:
    """Require an existing in-scope code span with a rule-relevant expression."""
    root = source_root.resolve()
    for affected in finding.get("affectedFiles", []):
        try:
            path = Path(affected["filePath"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                continue
            start = int(affected["range"]["start"]["line"])
            end = int(affected["range"]["end"]["line"])
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            if start < 1 or end < start or end > len(lines):
                continue
            excerpt = "\n".join(lines[start - 1 : end]).lower()
            if any(cue in excerpt for cue in RULE_SOURCE_CUES[code]):
                return True
        except (KeyError, TypeError, ValueError, OSError):
            continue
    return False


def score_finding(finding: dict[str, Any], family: str, source_root: Path) -> tuple[bool, str]:
    code = str(finding.get("code", ""))
    if family == "logical_bug":
        return False, "gold_family_too_broad_for_automatic_root_cause_match"
    if family not in RULE_FAMILY_MAP.get(code, set()):
        return False, "rule_does_not_match_gold_family"
    if finding.get("title") != RULE_TITLES[code] or not finding.get("description"):
        return False, "missing_causal_rule_description"
    if not has_grounded_location(finding, source_root, code):
        return False, "missing_grounded_code_location"
    return True, "matching_rule_with_causal_description_and_grounded_code_location"


def score_repeat(repeat_dir: Path, summary: dict[str, Any], denominator: int) -> dict[str, Any]:
    rows_by_incident: dict[str, list[dict[str, Any]]] = {}
    for row in summary["source_address_rows"]:
        rows_by_incident.setdefault(row["incident_id"], []).append(row)

    reviews = []
    for incident in summary["incident_rows"]:
        family = incident["gold_bug_family"]
        findings = []
        for row in rows_by_incident[incident["incident_id"]]:
            if row["status"] != "ok" or not row["detected"]:
                continue
            report = read_json(repeat_dir / row["output_path"])
            if report.get("success") is not True:
                continue
            for finding in report.get("results", []):
                if not isinstance(finding, dict):
                    continue
                credit, reason = score_finding(finding, family, Path(row["source_dir"]))
                findings.append({
                    "source_address_id": row["id"],
                    "rule_code": finding.get("code"),
                    "rule_title": finding.get("title"),
                    "rule_description": finding.get("description"),
                    "credit": credit,
                    "reason": reason,
                    "affected_files": finding.get("affectedFiles", []),
                })
        reviews.append({
            "incident_id": incident["incident_id"],
            "gold_bug_family": family,
            "detected": incident["detected"],
            "rcvr_credit": any(item["credit"] for item in findings),
            "findings": findings,
        })

    complete = summary["eligible_incidents"] == denominator and summary["failed_incidents"] == 0
    hits = sum(item["rcvr_credit"] for item in reviews)
    return {
        "status": "complete" if complete else "incomplete",
        "hits": hits,
        "denominator": denominator,
        "rate": hits / denominator if complete else None,
        "review_rows": reviews,
    }


def score_run(run_dir: Path) -> dict[str, Any]:
    config = read_json(run_dir / "run_config.json")
    aggregate_path = run_dir / "aggregate_summary.json"
    aggregate = read_json(aggregate_path)
    denominator = config["selected_incidents"]
    repeat_scores = []
    for repetition in range(1, config["repetitions"] + 1):
        repeat_dir = run_dir / f"repeat_{repetition:02d}"
        summary_path = repeat_dir / "summary.json"
        summary = read_json(summary_path)
        result = score_repeat(repeat_dir, summary, denominator)
        write_json(repeat_dir / "rcvr_review.json", {
            "repeat": repetition,
            "scoring_policy": "Conservative, rule-family match plus causal description and localized source code; logical_bug is too broad to credit automatically.",
            "rule_family_map": {key: sorted(value) for key, value in RULE_FAMILY_MAP.items()},
            **result,
        })
        summary["rcvr"] = {key: result[key] for key in ("status", "hits", "denominator", "rate")}
        write_json(summary_path, summary)
        repeat_scores.append(summary)

    rates = [entry["rcvr"]["rate"] for entry in repeat_scores]
    complete = all(rate is not None for rate in rates)
    aggregate["runs"] = repeat_scores
    aggregate["metric_definition"]["RCVR"] = (
        "Conservative automatic lower bound: benchmark-matching specific GPTScan rule, "
        "causal rule description, and localized source code. The generic logical_bug "
        "label is not sufficient for automatic root-cause credit."
    )
    aggregate["metric_statistics"]["rcvr"] = (
        {
            "status": "complete",
            "mean": statistics.mean(rates),
            "sample_std": statistics.stdev(rates) if len(rates) > 1 else 0.0,
            "min": min(rates),
            "max": max(rates),
        }
        if complete else {"status": "incomplete", "reason": "At least one incident has no usable GPTScan result."}
    )
    write_json(aggregate_path, aggregate)
    return aggregate["metric_statistics"]["rcvr"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(score_run(args.run_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
