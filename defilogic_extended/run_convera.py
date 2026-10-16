"""Run the full CONVERA detector-to-evidence pipeline on DeFiLogic-Extended.

Only verified deployed source retrieved by ``collect_sources.py`` is presented
to the pipeline.  The DeFiHackLabs reproduction and all fields beginning with
``gold_`` remain evaluation-only metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = BENCH_ROOT / "manifest.json"
SOURCE_ROOT = BENCH_ROOT / "sources"
PREPARED_ROOT = BENCH_ROOT / "prepared"
RESULTS_ROOT = BENCH_ROOT / "results"
POSITIVE_VERDICTS = {"True Positive", "Needs More Evidence"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def case_dir(case: dict[str, Any]) -> Path:
    return SOURCE_ROOT / case["id"]


def ready_cases(manifest: dict[str, Any], requested: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chosen = [case for case in manifest["cases"] if not requested or case["id"] in requested]
    unknown = requested - {case["id"] for case in chosen}
    if unknown:
        raise ValueError(f"Unknown case id(s): {', '.join(sorted(unknown))}")

    ready, skipped = [], []
    for case in chosen:
        status_path = case_dir(case) / "source_status.json"
        status = read_json(status_path) if status_path.exists() else {"ready": False, "error": "source not collected"}
        if status.get("ready"):
            ready.append({**case, "source_status": status})
        else:
            skipped.append({"case_id": case["id"], "reason": status.get("error", "source not ready")})
    return ready, skipped


def source_bundle(case: dict[str, Any]) -> tuple[str, str, list[str]]:
    root = case_dir(case) / "source"
    all_files = sorted(root.rglob("*.sol"))
    if not all_files:
        raise FileNotFoundError(f"No Solidity source files for {case['id']}")

    analysis_response = case_dir(case) / "analysis_response.json"
    explorer = read_json(analysis_response if analysis_response.exists() else case_dir(case) / "explorer_response.json")
    records = explorer.get("result") or []
    record = records[0] if isinstance(records, list) and records else {}
    contract_name = record.get("ContractName") or case["name"]

    # Explorer responses for a deployed contract may include every vendored
    # dependency.  Feeding all of them to the detector/LLM both dilutes the
    # target signal and can exceed the model context.  Prefer the verified
    # Solidity unit that declares the deployed contract; fall back to the
    # complete source bundle only when the declaration cannot be identified.
    declaration = re.compile(
        rf"\b(?:abstract\s+)?(?:contract|interface|library)\s+{re.escape(contract_name)}\b"
    )
    target_files = [
        path
        for path in all_files
        if declaration.search(path.read_text(encoding="utf-8", errors="ignore"))
    ]
    files = target_files or all_files
    contents = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        contents.extend((f"// FILE: {rel}", path.read_text(encoding="utf-8", errors="ignore")))

    abi = record.get("ABI", "")
    try:
        functions = [item["name"] for item in json.loads(abi) if item.get("type") == "function" and item.get("name")]
    except (json.JSONDecodeError, TypeError):
        functions = []
    semantic = (
        f"Verified deployed contract {contract_name} on {case['chain']}. "
        f"ABI functions include: {', '.join(functions[:24]) or 'unavailable'}."
    )
    return "\n\n".join(contents), semantic, [path.relative_to(root).as_posix() for path in files]


def prepare_case(case: dict[str, Any], force: bool) -> dict[str, Any]:
    """Create frozen source, parser, and detector artifacts once per case."""
    import business_bench_utils

    output = PREPARED_ROOT / case["id"]
    rag_path = output / "rag_output.json"
    parser_path = output / "parser_output.json"
    fusion_path = output / "fusion_output_agent.json"
    if not force and all(path.exists() for path in (rag_path, parser_path, fusion_path)):
        return {"rag_path": rag_path, "fusion_path": fusion_path}

    code, semantic, files = source_bundle(case)
    status = case["source_status"]
    write_json(
        rag_path,
        {
            "contract_name": status.get("contract_name") or case["name"],
            "code": code,
            "functional_semantic": semantic,
            "Audit_report": "",
            "metadata": {
                "incident_name": case["name"],
                "chain": case["chain"],
                "address": case["address"],
                "source_type": status.get("source_type", ""),
                "compiler_version": status.get("compiler_version", ""),
                "main_file": files[0],
                "dataset": "DeFiLogic-Extended",
            },
        },
    )
    business_bench_utils.generate_parser_output(str(rag_path), str(parser_path))
    business_bench_utils.generate_fusion_seed(str(parser_path), str(fusion_path))
    write_json(
        output / "input_provenance.json",
        {
            "case_id": case["id"],
            "source_files": files,
            "source_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
            "detector_sha256": hashlib.sha256(fusion_path.read_bytes()).hexdigest(),
            "gold_labels_withheld_from_input": True,
        },
    )
    return {"rag_path": rag_path, "fusion_path": fusion_path}


def parse_report(output: dict[str, Any]) -> dict[str, Any]:
    raw = output.get("Audit_report", {})
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            # Some OpenAI-compatible endpoints wrap an otherwise usable answer
            # in a Markdown fence or emit literal newlines inside a JSON string.
            # Do not silently turn such an answer into an empty verdict: retain
            # the fields needed by the evaluation, while the original response
            # remains saved verbatim in ``explainer_output.json`` for audit.
            fields: dict[str, Any] = {}
            for name in (
                "verdict",
                "violated_property",
                "root_cause_analysis",
                "exploit_scenario",
                "recommendation",
            ):
                match = re.search(rf'"{name}"\s*:\s*"(.*?)"\s*(?:,|\n\s*}})', raw, re.DOTALL)
                if match:
                    fields[name] = match.group(1).replace('\\"', '"').replace('\\n', '\n')
            confidence = re.search(r'"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)', raw)
            if confidence:
                fields["confidence"] = float(confidence.group(1))
            return fields
    return {}


def predicted_property_family(output: dict[str, Any]) -> str:
    """Map a generated property/root-cause explanation to the paper taxonomy.

    This mapping deliberately ignores generic Solidity terms such as ``require``:
    they occur in both safe and vulnerable code and caused a prior evaluator to
    label nearly every report as input validation.  The mapping is used only
    after a run is complete; gold labels are never part of the model input.
    """
    report = parse_report(output)
    text = " ".join(
        str(value)
        for value in (
            report.get("violated_property", ""),
            report.get("root_cause_analysis", ""),
            report.get("exploit_scenario", ""),
            output.get("validation_bundle", {}).get("summary", ""),
        )
    ).lower()
    if any(word in text for word in ("collateral", "solvency", "liquidation", "borrow", "health factor")):
        return "collateral_solvency"
    if any(word in text for word in ("entitlement", "single-use", "single use", "consume", "consumption", "already claimed")):
        return "entitlement_consumption"
    if any(word in text for word in ("reward", "payout", "eligiblefor", "threshold", "accrued fee", "fee accrual")):
        return "reward_payout_boundedness"
    if any(word in text for word in ("unit_share", "rounding", "accounting", "conservation", "equivalence", "share supply")):
        return "value_accounting_conservation"
    if any(word in text for word in ("asset identity", "asset mismatch", "wrong asset", "token identity")):
        return "asset_identity_consistency"
    if any(word in text for word in ("state transition", "phase", "ordering", "order of operations")):
        return "state_transition_correctness"
    if any(word in text for word in ("onlyowner", "owner", "admin", "unauthor", "permission", "access control")):
        return "authorization_governance"
    return "unknown"


def run_once(cases: list[dict[str, Any]], repetition: int, run_dir: Path, force_prepare: bool) -> list[dict[str, Any]]:
    import explainer

    tool = explainer.build_explainer_tool()
    rows = []
    for index, case in enumerate(cases, start=1):
        print(f"[repeat {repetition} | {index}/{len(cases)}] {case['id']}")
        prepared = prepare_case(case, force_prepare)
        rag = read_json(prepared["rag_path"])
        fusion = read_json(prepared["fusion_path"])
        result_path = run_dir / f"repeat_{repetition:02d}" / "cases" / case["id"] / "explainer_output.json"
        status, error, output = "ok", "", {}
        try:
            output = tool.run_pipeline(
                code_snippet=rag["code"],
                rag_metadata=rag["metadata"],
                vuln_type=fusion.get("Predict", ""),
                predicted_line=fusion.get("Predicted_Line_of_Vulnerability", "Unknown"),
            )
            write_json(result_path, output)
        except Exception as exc:
            status, error = "failed", f"{type(exc).__name__}: {exc}"
            write_json(result_path.with_name("error.json"), {"error": error})

        report = parse_report(output)
        validation = output.get("validation_bundle", {})
        rows.append(
            {
                "case_id": case["id"],
                "incident_name": case["name"],
                "status": status,
                "error": error,
                "gold_bug_family": case["gold_bug_family"],
                "gold_property_family": case["gold_property_family"],
                "verdict": report.get("verdict", ""),
                "confidence": report.get("confidence", 0.0),
                "validation_status": validation.get("aggregate_status", ""),
                "predicted_property_family": predicted_property_family(output) if output else "unknown",
                "output_path": str(result_path.relative_to(run_dir)),
            }
        )
    return rows


def rescore_once(cases: list[dict[str, Any]], repetition_dir: Path, run_dir: Path) -> list[dict[str, Any]]:
    """Recompute metrics from saved model outputs without making API calls."""
    rows = []
    for case in cases:
        result_path = repetition_dir / "cases" / case["id"] / "explainer_output.json"
        output: dict[str, Any] = {}
        status, error = "ok", ""
        if result_path.exists():
            try:
                output = read_json(result_path)
            except (OSError, json.JSONDecodeError) as exc:
                status, error = "failed", f"saved output unreadable: {exc}"
        else:
            status, error = "failed", "saved output not found"
        report = parse_report(output)
        validation = output.get("validation_bundle", {})
        rows.append(
            {
                "case_id": case["id"],
                "incident_name": case["name"],
                "status": status,
                "error": error,
                "gold_bug_family": case["gold_bug_family"],
                "gold_property_family": case["gold_property_family"],
                "verdict": report.get("verdict", ""),
                "confidence": report.get("confidence", 0.0),
                "validation_status": validation.get("aggregate_status", ""),
                "predicted_property_family": predicted_property_family(output) if output else "unknown",
                "output_path": str(result_path.relative_to(run_dir)),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]], eligible: int) -> dict[str, Any]:
    idr_hits = sum(row["status"] == "ok" and row["verdict"] in POSITIVE_VERDICTS for row in rows)
    rcvr_hits = sum(
        row["status"] == "ok"
        and row["verdict"] == "True Positive"
        and row["validation_status"] == "violated"
        and row["predicted_property_family"] == row["gold_property_family"]
        for row in rows
    )
    return {
        "eligible_incidents": eligible,
        "failed_cases": sum(row["status"] != "ok" for row in rows),
        "metric_definition": {
            "IDR": "positive incident has a True Positive or Needs More Evidence verdict",
            "RCVR": "positive incident has a True Positive verdict, a predicted property family matching the hidden gold property family, and Lightweight Validation status violated",
        },
        "idr": {"hits": idr_hits, "rate": idr_hits / eligible if eligible else 0.0},
        "rcvr": {"hits": rcvr_hits, "rate": rcvr_hits / eligible if eligible else 0.0},
        "rows": rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric_statistics(summaries: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Report sample mean and standard deviation across independent repeats."""
    result: dict[str, dict[str, float]] = {}
    for metric in ("idr", "rcvr"):
        rates = [float(summary[metric]["rate"]) for summary in summaries]
        result[metric] = {
            "mean": statistics.mean(rates) if rates else 0.0,
            "sample_std": statistics.stdev(rates) if len(rates) > 1 else 0.0,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CONVERA on DeFiLogic-Extended.")
    parser.add_argument("--case", action="append", default=[], help="Case id to evaluate; repeat for several cases.")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--rescore-dir", default="", help="Re-score saved repeat outputs without model calls.")
    args = parser.parse_args()
    if args.repetitions < 1:
        raise ValueError("--repetitions must be at least one")

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    load_dotenv(ROOT / ".env")

    manifest = read_json(MANIFEST_PATH)
    cases, skipped = ready_cases(manifest, set(args.case))
    if not cases:
        raise RuntimeError("No source-ready cases selected. Run collect_sources.py first.")
    print(f"Source-ready cases: {len(cases)}; excluded for missing/unverified source: {len(skipped)}")

    if args.prepare_only:
        for index, case in enumerate(cases, start=1):
            print(f"[prepare {index}/{len(cases)}] {case['id']}")
            prepare_case(case, args.force_prepare)
        return

    if args.rescore_dir:
        run_dir = Path(args.rescore_dir).resolve()
        repeat_dirs = sorted(path for path in run_dir.glob("repeat_*") if path.is_dir())
        if not repeat_dirs:
            raise RuntimeError("No repeat directories found under --rescore-dir.")
        all_summaries = []
        for repetition_dir in repeat_dirs:
            rows = rescore_once(cases, repetition_dir, run_dir)
            write_csv(repetition_dir / "address_results.csv", rows)
            summary = summarize(rows, len(cases))
            write_json(repetition_dir / "summary.json", summary)
            all_summaries.append(summary)
        aggregate = {
            "dataset": manifest["name"],
            "repetitions": len(all_summaries),
            "llm": {"provider": "mistral", "model": "ministral-8b-latest"},
            "manifest_cases": len(manifest["cases"]),
            "eligible_cases": len(cases),
            "excluded_cases": skipped,
            "metric_statistics": metric_statistics(all_summaries),
            "runs": all_summaries,
        }
        write_json(run_dir / "aggregate_summary.json", aggregate)
        print(f"Re-scored saved results in {run_dir}")
        return

    os.environ["LLM_PROVIDER"] = "mistral"
    if not (os.environ.get("MISTRAL_API_KEY") or os.environ.get("mistral")):
        raise RuntimeError("MISTRAL_API_KEY is not configured; add it to C:\\D\\KLTN\\.env before running.")

    run_dir = Path(args.run_dir).resolve() if args.run_dir else RESULTS_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    all_summaries = []
    for repetition in range(1, args.repetitions + 1):
        rows = run_once(cases, repetition, run_dir, args.force_prepare)
        write_csv(run_dir / f"repeat_{repetition:02d}" / "address_results.csv", rows)
        summary = summarize(rows, len(cases))
        write_json(run_dir / f"repeat_{repetition:02d}" / "summary.json", summary)
        all_summaries.append(summary)

    aggregate = {
        "dataset": manifest["name"],
        "repetitions": args.repetitions,
        "llm": {
            "provider": "mistral",
            "model": os.environ.get("MISTRAL_MODEL", "ministral-8b-latest"),
        },
        "manifest_cases": len(manifest["cases"]),
        "eligible_cases": len(cases),
        "excluded_cases": skipped,
        "metric_statistics": metric_statistics(all_summaries),
        "runs": all_summaries,
    }
    write_json(run_dir / "aggregate_summary.json", aggregate)
    print(f"Saved results to {run_dir}")


if __name__ == "__main__":
    main()
