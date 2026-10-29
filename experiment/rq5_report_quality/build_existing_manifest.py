"""Extract honest RQ5 input manifests from the saved, non-PoPoC outputs.

DeFi: four methods on the 15-case subset; CONVERA is Reason-Agent-only here.
Conventional: only RAG/Explainer branch reports selected by Consensus; Merged
has no saved merged audit report and is excluded, never assigned arbitrarily.
No model/API call is made by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFI = ROOT / "defilogic_extended"
DEFI_RESULTS = DEFI / "results"
CONVENTIONAL = ROOT / "rw_dataset/results/conventional_full_pipeline/run_20261023_084850"

DEFI_RUNS = {
    "CONVERA": [
        DEFI_RESULTS / "run_20261012_defilogic_ministral8b_r1/repeat_01/cases",
        DEFI_RESULTS / "run_20261013_defilogic_expansion10_mistral_r1/repeat_01/cases",
    ],
    "GPTLens": [
        DEFI_RESULTS / "gptlens_mistral_all15_20261013/cases",
        DEFI_RESULTS / "gptlens_mistral_all15_20261013_retry/cases",
    ],
    "VulnHuntGPT": [
        DEFI_RESULTS / "vulnhuntgpt_mistral_defi15_fullsource_retry_20261015/cases",
        DEFI_RESULTS / "vulnhuntgpt_mistral_defi15_20261015/cases",
    ],
    "GPTScan": [
        DEFI_RESULTS / "gptscan_mistral_convera15_8b_all3_20261018/repeat_01/cases",
    ],
}
DEFI_FILES = {"CONVERA": "explainer_output.json", "GPTLens": "ranked.json",
              "VulnHuntGPT": "output.json", "GPTScan": "gptscan_output.json"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_text(case_id: str) -> str:
    root = DEFI / "sources" / case_id / "source"
    paths = sorted(root.rglob("*.sol"))
    if not paths:
        raise FileNotFoundError(f"No Solidity source for {case_id}")
    return "\n".join(f"// FILE: {path.relative_to(root).as_posix()}\n"
                     + path.read_text(encoding="utf-8", errors="replace") for path in paths)


def candidate_path(case_id: str, method: str) -> Path | None:
    for root in DEFI_RUNS[method]:
        path = root / case_id / DEFI_FILES[method]
        if path.is_file():
            return path
    return None


def render_saved_report(method: str, saved: Any) -> str:
    """Render saved output without generating, correcting, or cherry-picking text."""
    if method == "CONVERA":
        report = saved.get("Audit_report", "") if isinstance(saved, dict) else ""
        return report.strip() if isinstance(report, str) else ""
    if method == "GPTLens":
        if not isinstance(saved, list) or not saved or not isinstance(saved[0], dict):
            return ""
        # The original runner sorted this list by final_score descending.
        top = saved[0]
        fields = ("function_name", "vulnerability", "reason", "criticism")
        return json.dumps({key: top.get(key, "") for key in fields}, ensure_ascii=False, indent=2)
    if method == "VulnHuntGPT":
        parsed = saved.get("parsed_response", {}) if isinstance(saved, dict) else {}
        findings = parsed.get("findings", []) if isinstance(parsed, dict) else []
        return json.dumps(findings[0], ensure_ascii=False, indent=2) if findings else ""
    if method == "GPTScan":
        findings = saved.get("results", []) if isinstance(saved, dict) else []
        return json.dumps(findings[0], ensure_ascii=False, indent=2) if findings else ""
    raise ValueError(f"Unsupported method: {method}")


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def build_defi(output_dir: Path) -> dict[str, Any]:
    manifest = {item["id"]: item for item in read_json(DEFI / "manifest.json")["cases"]}
    config = read_json(DEFI_RESULTS / "gptscan_mistral_convera15_8b_all3_20261018/run_config.json")
    case_ids = config["case_ids"]
    rows: list[dict[str, str]] = []
    ledger: list[dict[str, Any]] = []
    for case_id in case_ids:
        context = source_text(case_id)
        vulnerability = manifest[case_id]["gold_root_cause"].strip()
        for method in DEFI_RUNS:
            path = candidate_path(case_id, method)
            report = render_saved_report(method, read_json(path)) if path else ""
            ledger.append({"case_id": case_id, "method": method,
                           "status": "included" if report else "missing_or_empty_report",
                           "source_path": str(path.relative_to(ROOT)) if path else None,
                           "source_sha256": hash_file(path) if path else None,
                           "context_chars": len(context), "report_chars": len(report)})
            if report:
                rows.append({"case_id": case_id, "method": method,
                             "contract_context": context,
                             "identified_vulnerability": vulnerability, "report": report})
    all_path = output_dir / "defi_all_available.jsonl"
    pair_path = output_dir / "defi_convera_gptlens_paired.jsonl"
    write_jsonl(all_path, rows)
    paired = [row for row in rows if row["method"] in ("CONVERA", "GPTLens")]
    write_jsonl(pair_path, paired)
    counts = Counter(row["method"] for row in rows)
    summary = {
        "dataset": "DeFiLogic-Extended", "input_cases": len(case_ids),
        "report_count_by_method": dict(counts),
        "complete_four_method_case_ids": sorted(
            case_id for case_id in case_ids if all(
                any(r["case_id"] == case_id and r["method"] == method for r in rows)
                for method in DEFI_RUNS)),
        "convera_scope": "Property-Guided Reason Agent only; these DeFi runs have no Consensus decision",
        "rendering": {
            "CONVERA": "saved Audit_report verbatim",
            "GPTLens": "top item from saved score-ranked list, fields function_name/vulnerability/reason/criticism",
            "VulnHuntGPT": "first saved finding (no native rank available)",
            "GPTScan": "first saved finding (no native rank available); cases with no findings have no report",
        },
        "vulnerability_context": "gold_root_cause from DeFiLogic manifest, supplied to judge only (not used in generation)",
        "provenance": ledger,
    }
    (output_dir / "defi_provenance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build_conventional(output_dir: Path, run_dir: Path = CONVENTIONAL) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    ledger: list[dict[str, Any]] = []
    for case_dir in sorted(run_dir.glob("row_*")):
        path = case_dir / "consensus_output.json"
        if not path.is_file():
            ledger.append({"case_id": case_dir.name, "decision": None, "status": "no_consensus"})
            continue
        consensus = read_json(path)
        decision = consensus.get("decision")
        entry: dict[str, Any] = {"case_id": case_dir.name, "decision": decision}
        if decision not in ("RAG", "Explainer"):
            entry["status"] = "no_single_selected_branch_report"
            ledger.append(entry)
            continue
        branch_file = case_dir / ("rag_output.json" if decision == "RAG" else "explainer_output.json")
        input_file = case_dir / "input.json"
        if not branch_file.is_file() or not input_file.is_file():
            entry["status"] = "missing_branch_or_input"
            ledger.append(entry)
            continue
        branch = read_json(branch_file)
        context = read_json(input_file).get("code", "")
        report = branch.get("Audit_report", "")
        vulnerability = branch.get("vuln_type", "")
        if not all(isinstance(item, str) and item.strip()
                   for item in (context, report, vulnerability)):
            entry["status"] = "empty_context_or_report_or_type"
            ledger.append(entry)
            continue
        rows.append({"case_id": case_dir.name, "method": "CONVERA",
                     "contract_context": context, "identified_vulnerability": vulnerability,
                     "report": report})
        entry.update({"status": "included", "selected_branch_file": str(branch_file.relative_to(ROOT)),
                      "branch_sha256": hash_file(branch_file), "report_chars": len(report)})
        ledger.append(entry)
    write_jsonl(output_dir / "conventional_consensus_selected.jsonl", rows)
    summary = {
        "dataset": "conventional_full_pipeline", "run_dir": str(run_dir.relative_to(ROOT)),
        "included_reports": len(rows),
        "decision_counts": dict(Counter(item.get("decision") for item in ledger)),
        "note": "RAG/Explainer: saved report from selected branch. Merged: excluded because Consensus saved no merged audit report. No matching baseline reports for this conventional set.",
        "provenance": ledger,
    }
    (output_dir / "conventional_provenance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("defi", "conventional", "both"), default="both")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiment/rq5_report_quality/results/existing_inputs")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset in ("defi", "both"):
        result = build_defi(args.output_dir)
        print(f"DeFi: {result['report_count_by_method']}; four-way intersection="
              f"{len(result['complete_four_method_case_ids'])}")
    if args.dataset in ("conventional", "both"):
        result = build_conventional(args.output_dir)
        print(f"Conventional: {result['included_reports']} branch-selected reports; "
              f"decision counts={result['decision_counts']}")
    print(f"Manifests and provenance: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
