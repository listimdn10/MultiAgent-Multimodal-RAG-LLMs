"""Run official GPTScan rules over the 30-incident VERITE business-logic subset.

The VERITE manifest contains 35 selected source-address cases for 30 incidents.
This runner invokes GPTScan per source-address case and aggregates detection at
the incident level: an incident is detected when any selected address produces
at least one GPTScan finding. Gold bug-family labels are recorded only after
inference for scoring and are never supplied to GPTScan.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from defilogic_extended.run_gptscan_mistral import (
    GPTSCAN_PYTHON,
    HTTP_HEADERS,
    ROOT,
    SOLC_LIST_URL,
    compiler_version,
    ensure_solc,
    gptscan_environment,
    read_json,
    result_rules,
    run_case,
    write_json,
)
from rw_dataset.score_gptscan_verite import score_run


RW_ROOT = Path(__file__).resolve().parent
MANIFEST = RW_ROOT / "manifests" / "rw_manifest_business_logic.json"
RESULTS = RW_ROOT / "results" / "gptscan_verite"


def load_cases() -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Return selected source-address cases and their post-inference labels."""
    manifest = read_json(MANIFEST)
    cases: list[dict[str, Any]] = []
    incident_labels: dict[str, str] = {}
    for incident in manifest["incidents"]:
        incident_id = incident["incident_name"]
        has_source_ready_case = False
        for address in incident["addresses"]:
            if not (address.get("selected") and address.get("exists")):
                continue
            source_root = ROOT / Path(address["address_dir"])
            source_dir = source_root / "sources"
            metadata_path = source_root / "metadata.json"
            if not source_dir.exists() or not metadata_path.exists():
                continue
            metadata = read_json(metadata_path)
            raw_compiler = metadata.get("compiler_version", "")
            if not raw_compiler:
                continue
            cases.append(
                {
                    "id": f"{incident_id}__{address['address'].lower()}",
                    "incident_id": incident_id,
                    "address": address["address"],
                    "source_dir": source_dir,
                    "compiler": compiler_version(raw_compiler),
                    "gold_bug_family": incident["gold_bug_family"],
                }
            )
            has_source_ready_case = True
        if has_source_ready_case:
            incident_labels[incident_id] = incident["gold_bug_family"]
    if not cases:
        raise RuntimeError("No source-ready selected VERITE cases were found.")
    return cases, incident_labels


def incident_summary(rows: list[dict[str, Any]], incident_labels: dict[str, str]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["incident_id"], []).append(row)
    incidents: list[dict[str, Any]] = []
    for incident_id, members in grouped.items():
        usable = [member for member in members if member["status"] == "ok"]
        incidents.append(
            {
                "incident_id": incident_id,
                "gold_bug_family": incident_labels[incident_id],
                "selected_addresses": len(members),
                "usable_addresses": len(usable),
                "detected": any(member["detected"] for member in usable),
                "finding_case_ids": [member["id"] for member in usable if member["detected"]],
            }
        )
    incidents.sort(key=lambda entry: entry["incident_id"])
    eligible = [incident for incident in incidents if incident["usable_addresses"]]
    hits = sum(incident["detected"] for incident in eligible)
    return {
        "incident_rows": incidents,
        "eligible_incidents": len(eligible),
        "failed_incidents": len(incidents) - len(eligible),
        "idr": {"hits": hits, "rate": hits / len(eligible) if eligible else 0.0},
        "rcvr": {
            "status": "not_automatically_scored",
            "reason": "VERITE provides coarse bug-family labels only; post-hoc family and evidence review is required.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--model", default="ministral-8b-latest")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--force", action="store_true", help="Rerun case outputs already present.")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if not GPTSCAN_PYTHON.exists():
        parser.error(f"GPTScan virtual environment not found: {GPTSCAN_PYTHON}")

    load_dotenv(ROOT / ".env")
    cases, incident_labels = load_cases()
    output_dir = Path(args.run_dir).resolve() if args.run_dir else RESULTS / datetime.now().strftime("mistral_%Y%m%d_%H%M%S")
    releases_request = urllib.request.Request(SOLC_LIST_URL, headers=HTTP_HEADERS)
    releases = json.loads(urllib.request.urlopen(releases_request, timeout=60).read())["releases"]
    run_metadata = {
        "dataset": "VERITE",
        "split": "business_logic",
        "selected_source_addresses": len(cases),
        "selected_incidents": len(incident_labels),
        "model": {"provider": "mistral", "name": args.model},
        "repetitions": args.repetitions,
        "metric_definition": {
            "IDR": "An incident is detected when any selected source-address case emits at least one GPTScan finding.",
            "RCVR": "Conservative automatic lower bound from a specific rule-family match, causal description, and localized source code.",
        },
    }
    write_json(output_dir / "run_config.json", run_metadata)

    summaries: list[dict[str, Any]] = []
    for repetition in range(1, args.repetitions + 1):
        repeat_dir = output_dir / f"repeat_{repetition:02d}"
        rows = [run_case(case, repeat_dir, releases, args.model, args.force) for case in cases]
        summary = {"repeat": repetition, "source_address_rows": rows, **incident_summary(rows, incident_labels)}
        write_json(repeat_dir / "summary.json", summary)
        summaries.append(summary)
        print(
            f"repeat {repetition}: IDR {summary['idr']['hits']}/{summary['eligible_incidents']}",
            flush=True,
        )

    idr_rates = [summary["idr"]["rate"] for summary in summaries]
    aggregate = {
        **run_metadata,
        "metric_statistics": {
            "idr": {
                "mean": statistics.mean(idr_rates),
                "sample_std": statistics.stdev(idr_rates) if len(idr_rates) > 1 else 0.0,
                "min": min(idr_rates),
                "max": max(idr_rates),
            },
            "rcvr": {"status": "not_automatically_scored"},
        },
        "runs": summaries,
    }
    write_json(output_dir / "aggregate_summary.json", aggregate)
    rcvr = score_run(output_dir)
    if rcvr["status"] == "complete":
        print(f"RCVR {rcvr['mean']:.4f} +/- {rcvr['sample_std']:.4f}", flush=True)
    else:
        print(f"RCVR incomplete: {rcvr['reason']}", flush=True)
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
