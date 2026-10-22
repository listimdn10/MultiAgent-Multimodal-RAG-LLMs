"""Run two additional independent GPTLens/VulnHunt-GPT repetitions.

The first repetition for each method/dataset already exists. This script
creates repeat_02 and repeat_03; it never reuses predictions between runs.
Failed API calls are retried case-by-case in a separate directory.

Usage: python experiment/rq1_detection/repeat_baselines.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "rw_dataset" / "results" / "baseline_repeats_20261022"
JOBS = {
    "gptlens_verite": ["ablation_study/run_gptlens_verite.py"],
    "gptlens_defi": ["defilogic_extended/run_gptlens_mistral.py"],
    "vulnhunt_verite": ["defilogic_extended/run_vulnhuntgpt_mistral.py", "--dataset", "verite"],
    "vulnhunt_defi": ["defilogic_extended/run_vulnhuntgpt_mistral.py", "--dataset", "defi", "--max-source-chars", "250000"],
}


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    return completed.returncode


def resolve_idr(job_name: str, rows: list[dict]) -> dict:
    if job_name == "gptlens_verite":
        by_incident: dict[str, list[dict]] = {}
        for row in rows:
            by_incident.setdefault(row["incident_name"], []).append(row)
        hits = sum(any(r.get("detected", False) for r in group) for group in by_incident.values())
        denominator = len(by_incident)
    else:
        hits = sum(bool(row.get("detected")) for row in rows)
        denominator = len(rows)
    return {"hits": hits, "denominator": denominator, "rate": hits / denominator if denominator else None}


def execute_job(job_name: str, repeat: int, output_root: Path, max_retries: int) -> dict:
    run_dir = output_root / job_name / f"repeat_{repeat:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    command_prefix = [sys.executable, str(ROOT / JOBS[job_name][0]), *JOBS[job_name][1:]]
    initial_summary = run_dir / "summary.json"
    if not initial_summary.exists():
        command = [*command_prefix, "--run-dir", str(run_dir)]
        save_json(run_dir / "command.json", {"argv": command, "started_utc": datetime.now(timezone.utc).isoformat()})
        exit_code = run_command(command, run_dir / "run.log")
        if exit_code != 0 or not initial_summary.exists():
            raise RuntimeError(f"{job_name} repeat_{repeat:02d}: runner failed; inspect {run_dir / 'run.log'}")

    raw = load_json(initial_summary)
    rows = list(raw["rows"])
    attempts: dict[str, str] = {row["case_id"]: str(run_dir / "cases" / row["case_id"]) for row in rows if row.get("status") == "ok"}
    for retry_number in range(1, max_retries + 1):
        failures = [row for row in rows if row.get("status") != "ok"]
        if not failures:
            break
        for failure in failures:
            case_id = failure["case_id"]
            retry_dir = run_dir / "retries" / f"attempt_{retry_number:02d}" / case_id
            command = [*command_prefix, "--case", case_id, "--run-dir", str(retry_dir)]
            if retry_number > 1:
                time.sleep(min(120, 30 * retry_number))
            exit_code = run_command(command, retry_dir / "run.log")
            summary_path = retry_dir / "summary.json"
            if exit_code != 0 or not summary_path.exists():
                continue
            retry_rows = load_json(summary_path)["rows"]
            replacement = next((row for row in retry_rows if row["case_id"] == case_id and row.get("status") == "ok"), None)
            if replacement is None:
                continue
            rows = [replacement if row["case_id"] == case_id else row for row in rows]
            attempts[case_id] = str(retry_dir / "cases" / case_id)

    unresolved = [row["case_id"] for row in rows if row.get("status") != "ok"]
    resolved = {
        "job": job_name,
        "repeat": repeat,
        "model": raw.get("model") or os.environ.get("MISTRAL_MODEL") or dotenv_values(ROOT / ".env").get("MISTRAL_MODEL") or "ministral-8b-latest",
        "original_summary": str(initial_summary),
        "idr": resolve_idr(job_name, rows),
        "unresolved_cases": unresolved,
        "artifact_dirs": attempts,
        "rows": rows,
        "rcvr": None,
        "rcvr_note": "Requires separate case-level root-cause adjudication; not inferred from IDR.",
    }
    save_json(run_dir / "resolved_summary.json", resolved)
    if unresolved:
        raise RuntimeError(f"{job_name} repeat_{repeat:02d}: unresolved API failures: {unresolved}")
    return {"job": job_name, "repeat": repeat, "idr": resolved["idr"], "summary": str(run_dir / "resolved_summary.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--only", choices=JOBS)
    parser.add_argument("--max-retries", type=int, default=4)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    jobs = [args.only] if args.only else list(JOBS)
    progress_path = output_root / "progress.json"
    progress = {"started_utc": datetime.now(timezone.utc).isoformat(), "completed": [], "current": None}
    save_json(progress_path, progress)
    for job_name in jobs:
        for repeat in (2, 3):
            progress["current"] = {"job": job_name, "repeat": repeat}
            save_json(progress_path, progress)
            result = execute_job(job_name, repeat, output_root, args.max_retries)
            progress["completed"].append(result)
            print(f"{job_name} repeat_{repeat:02d}: IDR {result['idr']['hits']}/{result['idr']['denominator']}", flush=True)
            save_json(progress_path, progress)
    progress["current"] = None
    progress["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(progress_path, progress)


if __name__ == "__main__":
    main()
