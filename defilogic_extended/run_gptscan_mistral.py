"""Run the official GPTScan rules on the CONVERA DeFiLogic common subset.

The original GPTScan implementation requires one ``solc`` binary visible on
``PATH`` and an OpenAI-compatible chat endpoint.  This runner isolates a
compiler per case according to the verified-source metadata, keeps the secret
only in the child environment, and records raw GPTScan JSON per case.

It reports incident-detection recall (IDR) directly: an incident is detected
when GPTScan emits at least one finding. GPTScan rule labels are not protocol
root-cause labels, so RCVR is deliberately *not* inferred automatically; a
separate post-hoc rule-to-root-cause review is required before reporting it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
BENCH = Path(__file__).resolve().parent
MANIFEST = BENCH / "manifest.json"
SOURCES = BENCH / "sources"
RESULTS = BENCH / "results"
GPTSCAN = BENCH / "third_party" / "GPTScan"
GPTSCAN_SRC = GPTSCAN / "src"
GPTSCAN_PYTHON = GPTSCAN / ".venv" / "Scripts" / "python.exe"
SOLC_ROOT = GPTSCAN / "solc_cache"
SOLC_LIST_URL = "https://binaries.soliditylang.org/windows-amd64/list.json"
HTTP_HEADERS = {"User-Agent": "CONVERA-replication/1.0"}

# The exact 15 source-ready incidents used by the existing CONVERA DeFiLogic
# evaluation (five pilot cases plus ten expansion cases).
CONVERA15 = (
    "setprotocol_unit_share_rounding",
    "ajna_self_controlled_liquidation",
    "aventa_reward_claim_bound",
    "popsicle_repeated_reward_claim",
    "compound_uni_price_feeder",
    "euler_donation_liquidation",
    "platypus_flashloan_collateral",
    "hundred_finance_exchange_rate",
    "indexed_finance_pool_weight",
    "cream_yusd_collateral_valuation",
    "qubit_bridge_zero_deposit",
    "roe_finance_delegation_collateral",
    "wise_lending_share_withdrawal",
    "harvest_vault_share_accounting",
    "yearn_ydai_share_accounting",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def compiler_version(raw: str) -> str:
    match = re.search(r"(\d+\.\d+\.\d+)", raw)
    if not match:
        raise ValueError(f"Cannot parse Solidity compiler version: {raw!r}")
    return match.group(1)


def load_cases(case_ids: list[str]) -> list[dict[str, Any]]:
    requested = set(case_ids or CONVERA15)
    records = {record["id"]: record for record in read_json(MANIFEST)["cases"]}
    unknown = requested.difference(records)
    if unknown:
        raise ValueError(f"Unknown case id(s): {', '.join(sorted(unknown))}")

    cases: list[dict[str, Any]] = []
    # Preserve the common-subset order, or the manifest order for explicit cases.
    ordered = CONVERA15 if not case_ids else tuple(record["id"] for record in read_json(MANIFEST)["cases"])
    for case_id in ordered:
        if case_id not in requested:
            continue
        status_path = SOURCES / case_id / "source_status.json"
        source_dir = SOURCES / case_id / "source"
        if not status_path.exists() or not source_dir.exists():
            continue
        status = read_json(status_path)
        if not status.get("ready"):
            continue
        cases.append(
            {
                "id": case_id,
                "incident_name": records[case_id]["name"],
                "source_dir": source_dir,
                "compiler": compiler_version(status.get("compiler_version", "")),
                "gold_bug_family": records[case_id]["gold_bug_family"],
                "gold_property_family": records[case_id]["gold_property_family"],
            }
        )
    if not cases:
        raise RuntimeError("No source-ready cases were selected.")
    return cases


def solc_filename(version: str, releases: dict[str, str]) -> str:
    filename = releases.get(version)
    if not filename:
        raise RuntimeError(f"No Windows solc artifact listed for {version}.")
    return filename


def ensure_solc(version: str, releases: dict[str, str]) -> Path:
    target = SOLC_ROOT / version
    executable = target / "solc.exe"
    if executable.exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    artifact = solc_filename(version, releases)
    download = target / artifact
    request = urllib.request.Request(
        f"https://binaries.soliditylang.org/windows-amd64/{artifact}", headers=HTTP_HEADERS
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        download.write_bytes(response.read())
    if artifact.endswith(".zip"):
        shutil.unpack_archive(download, target)
        download.unlink()
        discovered = next(target.rglob("solc.exe"), None)
        if discovered is None:
            raise RuntimeError(f"Downloaded compiler archive for {version} has no solc.exe")
        if discovered != executable:
            shutil.copy2(discovered, executable)
    else:
        download.rename(executable)
    return target


def gptscan_environment(compiler_dir: Path, model: str) -> dict[str, str]:
    key = os.environ.get("MISTRAL_API_KEY") or os.environ.get("mistral")
    if not key:
        raise RuntimeError("Set MISTRAL_API_KEY (or legacy mistral) in .env.")
    env = os.environ.copy()
    env.update(
        {
            "OPENAI_API_KEY": key,
            "OPENAI_API_BASE": "https://api.mistral.ai/v1",
            "GPTSCAN_MODEL": model,
            "GPTSCAN_GPT4_MODEL": model,
            "PYTHONUTF8": "1",
            "PATH": str(compiler_dir) + os.pathsep + env.get("PATH", ""),
        }
    )
    return env


def result_rules(payload: dict[str, Any]) -> list[str]:
    rules: list[str] = []
    for finding in payload.get("results", []):
        if not isinstance(finding, dict):
            continue
        value = finding.get("type") or finding.get("name") or finding.get("rule")
        if value:
            rules.append(str(value))
    return sorted(set(rules))


def run_case(case: dict[str, Any], output_dir: Path, releases: dict[str, str], model: str, force: bool) -> dict[str, Any]:
    case_dir = output_dir / "cases" / case["id"]
    report_path = case_dir / "gptscan_output.json"
    row_case = {key: str(value) if isinstance(value, Path) else value for key, value in case.items()}
    if report_path.exists() and not force:
        payload = read_json(report_path)
        if payload.get("success") is True:
            return {
                **row_case,
                "status": "ok",
                "detected": bool(payload.get("results")),
                "rules": result_rules(payload),
                "output_path": str(report_path.relative_to(output_dir)),
                "reused": True,
            }

    compiler_dir = ensure_solc(case["compiler"], releases)
    case_dir.mkdir(parents=True, exist_ok=True)
    command = [str(GPTSCAN_PYTHON), "main.py", "-s", str(case["source_dir"]), "-o", str(report_path)]
    completed = subprocess.run(
        command,
        cwd=GPTSCAN_SRC,
        env=gptscan_environment(compiler_dir, model),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=1800,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    (case_dir / "stdout.log").write_text(completed.stdout or "", encoding="utf-8", errors="replace")
    (case_dir / "stderr.log").write_text(completed.stderr or "", encoding="utf-8", errors="replace")
    payload = read_json(report_path) if report_path.exists() else {}
    status = "ok" if completed.returncode == 0 and payload.get("success") else "tool_failure"
    return {
        **row_case,
        "status": status,
        "detected": bool(payload.get("results")) if status == "ok" else False,
        "rules": result_rules(payload),
        "returncode": completed.returncode,
        "output_path": str(report_path.relative_to(output_dir)),
        "reused": False,
    }


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["status"] == "ok"]
    hits = sum(bool(row["detected"]) for row in eligible)
    return {
        "eligible_incidents": len(eligible),
        "failed_cases": len(rows) - len(eligible),
        "idr": {"hits": hits, "rate": hits / len(eligible) if eligible else 0.0},
        "rcvr": {
            "status": "not_automatically_scored",
            "reason": "GPTScan rules are not benchmark root-cause labels; post-hoc rule-to-root-cause review is required.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", default=[], help="Case ID; default is the CONVERA 15-case subset.")
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
    cases = load_cases(args.case)
    output_dir = Path(args.run_dir).resolve() if args.run_dir else RESULTS / datetime.now().strftime("gptscan_mistral_%Y%m%d_%H%M%S")
    releases_request = urllib.request.Request(SOLC_LIST_URL, headers=HTTP_HEADERS)
    releases = json.loads(urllib.request.urlopen(releases_request, timeout=60).read())["releases"]
    run_metadata = {
        "dataset": "DeFiLogic-Extended",
        "subset": "CONVERA-15" if not args.case else "explicit",
        "model": {"provider": "mistral", "name": args.model},
        "repetitions": args.repetitions,
        "metric_definition": {
            "IDR": "A positive incident has at least one GPTScan finding.",
            "RCVR": "Not automatically reported: GPTScan's rule taxonomy is not a root-cause taxonomy.",
        },
        "case_ids": [case["id"] for case in cases],
    }
    write_json(output_dir / "run_config.json", run_metadata)
    summaries: list[dict[str, Any]] = []
    for repetition in range(1, args.repetitions + 1):
        repeat_dir = output_dir / f"repeat_{repetition:02d}"
        rows = [run_case(case, repeat_dir, releases, args.model, args.force) for case in cases]
        summary = {"repeat": repetition, "rows": rows, **score(rows)}
        write_json(repeat_dir / "summary.json", summary)
        summaries.append(summary)
        print(f"repeat {repetition}: IDR {summary['idr']['hits']}/{summary['eligible_incidents']}", flush=True)
    rates = [entry["idr"]["rate"] for entry in summaries]
    aggregate = {
        **run_metadata,
        "metric_statistics": {
            "idr": {
                "mean": statistics.mean(rates),
                "sample_std": statistics.stdev(rates) if len(rates) > 1 else 0.0,
                "min": min(rates),
                "max": max(rates),
            },
            "rcvr": {"status": "not_automatically_scored"},
        },
        "runs": summaries,
    }
    write_json(output_dir / "aggregate_summary.json", aggregate)
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
