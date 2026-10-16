"""Reproducible ablations for the CONVERA Reason-Agent pipeline.

The runner fixes the source code and upstream detector signal for every
configuration, stores every intermediate output, and computes incident-level
metrics with an explicit definition.  It intentionally does not reuse values
from the paper figure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = STUDY_ROOT / "inputs"
RESULTS_ROOT = STUDY_ROOT / "results"
# This completed VERITE run contains the source and upstream fusion signal used
# for the paper's 30-incident result. Reusing it makes every ablation consume
# the same detector output and avoids recomputing large neural embeddings.
FROZEN_INPUT_RUN = ROOT / "rw_dataset" / "results" / "business_logic" / "run_20260827_233023" / "addresses"

# The four booleans are passed directly to ExplainerTool.run_pipeline().
SETTINGS: Dict[str, Dict[str, bool]] = {
    "full_convera": {
        "property_synthesis": True,
        "scenario_generation": True,
        "validator_spec_synthesis": True,
        "lightweight_validation": True,
    },
    "without_property_synthesis": {
        "property_synthesis": False,
        "scenario_generation": True,
        "validator_spec_synthesis": True,
        "lightweight_validation": True,
    },
    "without_scenario_generation": {
        "property_synthesis": True,
        "scenario_generation": False,
        "validator_spec_synthesis": True,
        "lightweight_validation": True,
    },
    "without_validator_spec_synthesis": {
        "property_synthesis": True,
        "scenario_generation": True,
        "validator_spec_synthesis": False,
        # Keep the downstream stage active: it receives the explicitly empty
        # specification bundle and returns inconclusive evidence. This isolates
        # removal of spec synthesis from removal of validation itself.
        "lightweight_validation": True,
    },
    "without_lightweight_validation": {
        "property_synthesis": True,
        "scenario_generation": True,
        "validator_spec_synthesis": True,
        "lightweight_validation": False,
    },
}

POSITIVE_VERDICTS = {"True Positive", "Needs More Evidence"}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_case_id(case_id: str) -> str:
    """Keep case directory names portable while retaining an address suffix."""
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in case_id)
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:10]
    return f"{safe[:100]}__{digest}"


def select_cases(limit_incidents: int, incident_name: str) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for directory in sorted(FROZEN_INPUT_RUN.iterdir()):
        if not directory.is_dir():
            continue
        metadata_path = directory / "case_metadata.json"
        rag_path = directory / "rag_output.json"
        fusion_path = directory / "fusion_output_agent.json"
        if not (metadata_path.exists() and rag_path.exists() and fusion_path.exists()):
            continue
        metadata = read_json(metadata_path)
        if incident_name and metadata.get("incident_name") != incident_name:
            continue
        cases.append(
            {
                "case_id": directory.name,
                "incident_name": metadata["incident_name"],
                "chain": metadata.get("chain", ""),
                "address": metadata["address"],
                "bug_type": metadata.get("bug_type", ""),
                "gold_bug_family": metadata.get("gold_bug_family", "unknown"),
                "frozen_dir": directory,
            }
        )

    if limit_incidents:
        kept_incidents = []
        selected: List[Dict[str, Any]] = []
        for case in cases:
            if case["incident_name"] not in kept_incidents:
                if len(kept_incidents) >= limit_incidents:
                    continue
                kept_incidents.append(case["incident_name"])
            if case["incident_name"] in kept_incidents:
                selected.append(case)
        cases = selected
    return cases


def input_dir(case: Dict[str, Any]) -> Path:
    return INPUT_ROOT / stable_case_id(case["case_id"])


def prepare_inputs(cases: Iterable[Dict[str, Any]], force: bool = False) -> None:
    """Copy one fixed source/detector snapshot before any ablation setting runs."""
    for index, case in enumerate(cases, start=1):
        destination = input_dir(case)
        required = [destination / "rag_output.json", destination / "fusion_output_agent.json", destination / "case.json"]
        if not force and all(path.exists() for path in required):
            print(f"[{index}] cached input: {case['case_id']}")
            continue

        print(f"[{index}] caching frozen detector input: {case['case_id']}")
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(case["frozen_dir"] / "rag_output.json", destination / "rag_output.json")
        shutil.copyfile(case["frozen_dir"] / "fusion_output_agent.json", destination / "fusion_output_agent.json")
        rag = read_json(destination / "rag_output.json")
        write_json(
            destination / "case.json",
            {
                "case_id": case["case_id"],
                "incident_name": case["incident_name"],
                "chain": case["chain"],
                "address": case["address"],
                "bug_type": case["bug_type"],
                "gold_bug_family": case["gold_bug_family"],
                "source_fingerprint": hashlib.sha256(rag.get("code", "").encode("utf-8")).hexdigest(),
                "detector_fingerprint": hashlib.sha256(
                    (destination / "fusion_output_agent.json").read_bytes()
                ).hexdigest(),
            },
        )


def assert_cached_inputs(cases: Iterable[Dict[str, Any]]) -> None:
    missing = []
    for case in cases:
        directory = input_dir(case)
        if not (directory / "rag_output.json").exists() or not (directory / "fusion_output_agent.json").exists():
            missing.append(case["case_id"])
    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(
            f"Missing cached inputs for {len(missing)} case(s), e.g. {preview}. "
            "Run `prepare-inputs` before `run`."
        )


def parse_audit_report(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def predicted_family(output: Dict[str, Any]) -> str:
    audit = parse_audit_report(output.get("Audit_report"))
    haystack = " ".join(
        [
            str(output.get("vuln_type", "")),
            str(output.get("validated_bug_type", "")),
            str(audit.get("root_cause_analysis", "")),
            str(audit.get("violated_property", "")),
            str(output.get("validation_bundle", {}).get("summary", "")),
        ]
    ).lower()
    if any(token in haystack for token in ("onlyowner", "owner", "admin", "forbidden", "unauthor", "permission", "access")):
        return "access_control"
    if any(token in haystack for token in ("input validation", "parameter", "sanit", "validation", "bounds check", "require")):
        return "input_validation"
    if any(token in haystack for token in ("calculation", "rounding", "price", "fee", "math", "accounting", "conservation", "equivalence")):
        return "miscalculation"
    relation_types = [
        spec.get("assertion", {}).get("relation_type", "")
        for spec in output.get("validator_spec_bundle", {}).get("validator_specs", [])
        if isinstance(spec, dict) and isinstance(spec.get("assertion"), dict) and spec["assertion"].get("relation_type")
    ]
    return "logical_bug" if relation_types else "unknown"


def run_one_setting(
    setting_name: str,
    config: Dict[str, bool],
    cases: Iterable[Dict[str, Any]],
    output_dir: Path,
    resume: bool = False,
) -> List[Dict[str, Any]]:
    import explainer

    tool = explainer.build_explainer_tool()
    rows: List[Dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        cache = input_dir(case)
        rag = read_json(cache / "rag_output.json")
        fusion = read_json(cache / "fusion_output_agent.json")
        case_dir = output_dir / "cases" / stable_case_id(case["case_id"])
        output_path = case_dir / "explainer_output.json"
        error_path = case_dir / "error.json"
        status = "ok"
        error = ""
        output: Dict[str, Any] = {}
        if resume and output_path.exists():
            # A completed LLM result is immutable experiment evidence. Reuse it
            # and retry only cases that previously failed due to API/quota errors.
            output = read_json(output_path)
            print(f"[{setting_name} {index}] cached success: {case['case_id']}")
        else:
            print(f"[{setting_name} {index}] {case['case_id']}")
            try:
                output = tool.run_pipeline(
                    code_snippet=rag.get("code", ""),
                    rag_metadata=rag.get("metadata", {}),
                    vuln_type=fusion.get("Predict", ""),
                    predicted_line=fusion.get("Predicted_Line_of_Vulnerability", "Unknown"),
                    stage_config=config,
                )
                write_json(output_path, output)
                # This error file belongs to a failed prior attempt and is now
                # superseded by a successful response for the same case/config.
                if error_path.exists():
                    error_path.unlink()
            except Exception as exc:  # Preserve failures as benchmark data.
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                write_json(error_path, {"error": error})

        audit = parse_audit_report(output.get("Audit_report"))
        validation = output.get("validation_bundle", {})
        verdict = audit.get("verdict", "")
        row = {
            "case_id": case["case_id"],
            "incident_name": case["incident_name"],
            "address": case["address"],
            "gold_bug_family": case["gold_bug_family"],
            "status": status,
            "error": error,
            "verdict": verdict,
            "confidence": audit.get("confidence", 0.0),
            "validation_status": validation.get("aggregate_status", ""),
            "predicted_family": predicted_family(output) if output else "unknown",
            "output_path": str(output_path.relative_to(output_dir)),
        }
        rows.append(row)
    return rows


def strict_rcvr_hit(row: Dict[str, Any]) -> bool:
    """Exact paper criterion: confirmed root cause plus explicit violation evidence."""
    return (
        row["status"] == "ok"
        and row["verdict"] == "True Positive"
        and row["validation_status"] == "violated"
        and row["predicted_family"] == row["gold_bug_family"]
    )


def summarize_rows(rows: List[Dict[str, Any]], config: Dict[str, bool]) -> Dict[str, Any]:
    by_incident: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_incident[row["incident_name"]].append(row)

    incident_rows: List[Dict[str, Any]] = []
    for incident_name, members in sorted(by_incident.items()):
        incident_rows.append(
            {
                "incident_name": incident_name,
                "gold_bug_family": members[0]["gold_bug_family"],
                # Keep IDR aligned with the existing benchmark convention: an
                # unresolved finding is a detected incident, not a safe verdict.
                "idr_hit": any(row["status"] == "ok" and row["verdict"] in POSITIVE_VERDICTS for row in members),
                "rcvr_hit": any(strict_rcvr_hit(row) for row in members),
                "num_addresses": len(members),
                "case_rows": members,
            }
        )

    total = len(incident_rows)
    idr_hits = sum(item["idr_hit"] for item in incident_rows)
    validation_available = config["validator_spec_synthesis"] and config["lightweight_validation"]
    rcvr_hits = sum(item["rcvr_hit"] for item in incident_rows) if validation_available else None
    failed_cases = sum(row["status"] != "ok" for row in rows)
    return {
        "metric_definition": {
            "IDR": "incident has at least one True Positive or Needs More Evidence verdict",
            "RCVR": "incident has at least one True Positive whose predicted family matches the benchmark family and whose Lightweight Validation status is violated",
            "RCVR_denominator": "all benchmark incidents; null when Validator-Spec Synthesis or Lightweight Validation is disabled",
        },
        "num_incidents": total,
        "num_address_cases": len(rows),
        "failed_address_cases": failed_cases,
        "idr": {"hits": idr_hits, "rate": idr_hits / total if total else 0.0},
        "rcvr": (
            {"hits": rcvr_hits, "rate": rcvr_hits / total if total else 0.0}
            if rcvr_hits is not None
            else None
        ),
        "incidents": incident_rows,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id", "incident_name", "address", "gold_bug_family", "status", "error",
        "verdict", "confidence", "validation_status", "predicted_family", "output_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_repetitions(per_setting: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {}
    for name, summaries in per_setting.items():
        def distribution(key: str) -> Optional[Dict[str, float]]:
            values = [summary[key]["rate"] for summary in summaries if summary[key] is not None]
            if not values:
                return None
            return {"mean": mean(values), "population_std": pstdev(values) if len(values) > 1 else 0.0}

        aggregate[name] = {
            "repetitions": len(summaries),
            "idr": distribution("idr"),
            "rcvr": distribution("rcvr"),
            "runs": summaries,
        }
    return aggregate


def run_experiment(
    cases: List[Dict[str, Any]],
    setting_names: List[str],
    repetitions: int,
    run_dir: Path,
    resume: bool = False,
) -> None:
    assert_cached_inputs(cases)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {name: SETTINGS[name] for name in setting_names},
        "repetitions": repetitions,
        "num_cached_address_cases": len(cases),
        "openai_model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "llm_backend": "openai" if os.environ.get("OPENAI_API_KEY") else "ollama",
        "warning": "LLM sampling settings are controlled by the configured provider; report mean and standard deviation across repetitions.",
    }
    if not resume or not (run_dir / "experiment_config.json").exists():
        write_json(run_dir / "experiment_config.json", metadata)

    per_setting: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for repeat_index in range(1, repetitions + 1):
        for setting_name in setting_names:
            setting_dir = run_dir / f"repeat_{repeat_index:02d}" / setting_name
            rows = run_one_setting(setting_name, SETTINGS[setting_name], cases, setting_dir, resume=resume)
            write_csv(setting_dir / "address_results.csv", rows)
            summary = summarize_rows(rows, SETTINGS[setting_name])
            write_json(setting_dir / "summary.json", summary)
            per_setting[setting_name].append(summary)

    write_json(run_dir / "aggregate_summary.json", aggregate_repetitions(per_setting))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible CONVERA Reason-Agent ablations.")
    parser.add_argument("command", choices=["prepare-inputs", "run"])
    parser.add_argument("--setting", action="append", choices=sorted(SETTINGS), help="Setting to run; repeat option for several settings.")
    parser.add_argument("--repetitions", type=int, default=3, help="Independent repetitions per setting (default: 3).")
    parser.add_argument("--limit-incidents", type=int, default=0, help="Use the first N incidents for a smoke test.")
    parser.add_argument("--incident-name", default="", help="Run a single named incident.")
    parser.add_argument("--force", action="store_true", help="Rebuild cached upstream detector inputs.")
    parser.add_argument("--run-dir", default="", help="Output directory for `run` (default: timestamped results directory).")
    parser.add_argument("--resume-run-dir", default="", help="Resume a previous run directory; completed outputs are reused and only failed cases are retried.")
    args = parser.parse_args()

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    cases = select_cases(args.limit_incidents, args.incident_name)
    if not cases:
        raise RuntimeError("No benchmark cases selected.")

    if args.command == "prepare-inputs":
        prepare_inputs(cases, force=args.force)
        print(f"Prepared {len(cases)} cached address case(s) in {INPUT_ROOT}.")
        return

    if args.repetitions < 1:
        raise ValueError("--repetitions must be at least 1.")
    setting_names = args.setting or list(SETTINGS)
    if args.run_dir and args.resume_run_dir:
        raise ValueError("Use only one of --run-dir and --resume-run-dir.")
    resume_dir = Path(args.resume_run_dir).resolve() if args.resume_run_dir else None
    run_dir = resume_dir or (Path(args.run_dir).resolve() if args.run_dir else RESULTS_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=resume_dir is not None)
    run_experiment(cases, setting_names, args.repetitions, run_dir, resume=resume_dir is not None)
    print(f"Saved ablation results to {run_dir}")


if __name__ == "__main__":
    main()
