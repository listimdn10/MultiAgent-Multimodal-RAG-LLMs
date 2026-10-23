import argparse
import json
import os
import shutil
import sys
import io
from datetime import datetime
from pathlib import Path

from business_bench_utils import generate_parser_output, generate_fusion_seed
from langgraph_pipeline import build_audit_graph


try:
    if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if sys.stderr.encoding is None or sys.stderr.encoding.lower() != "utf-8":
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass


ROOT = Path(__file__).resolve().parent
RAG_OUTPUT_PATH = ROOT / "rag_output.json"
PARSER_OUTPUT_PATH = ROOT / "parser_output.json"
FUSION_OUTPUT_PATH = ROOT / "fusion_output_agent.json"
EXPLAINER_OUTPUT_PATH = ROOT / "explainer_output.json"
SAMPLE_CONTRACT_PATH = ROOT / "contracts" / "sample.sol"


def save_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_rag_payload(case: dict, code: str) -> dict:
    return {
        "contract_name": case["name"],
        "code": code,
        "functional_semantic": case["functional_semantic"],
        "Audit_report": case.get("seed_audit_report", ""),
        "metadata": {
            "case_id": case["id"],
            "protocol_family": case["protocol_family"],
            "bug_family": case["bug_family"],
            "property_family": case["property_family"],
            "source_reference": case["source_reference"],
        },
    }


def prepare_case_inputs(case: dict):
    contract_path = ROOT / case["contract_path"]
    code = contract_path.read_text(encoding="utf-8")

    shutil.copyfile(contract_path, SAMPLE_CONTRACT_PATH)
    save_json(RAG_OUTPUT_PATH, build_rag_payload(case, code))
    generate_parser_output(str(RAG_OUTPUT_PATH), str(PARSER_OUTPUT_PATH))
    generate_fusion_seed(str(PARSER_OUTPUT_PATH), str(FUSION_OUTPUT_PATH))


def build_case_graph():
    return build_audit_graph(("fusion", "reason"))


def build_explainer_only_graph():
    return build_audit_graph(("reason",))


def run_single():
    result = build_case_graph().invoke({})
    print("\nFinal result:")
    print(result)


def persist_case_outputs(case: dict, result_dir: Path):
    case_dir = result_dir / case["id"]
    case_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(ROOT / case["contract_path"], case_dir / Path(case["contract_path"]).name)
    for src, dst_name in [
        (RAG_OUTPUT_PATH, "rag_output.json"),
        (PARSER_OUTPUT_PATH, "parser_output.json"),
        (FUSION_OUTPUT_PATH, "fusion_output_agent.json"),
        (EXPLAINER_OUTPUT_PATH, "explainer_output.json"),
    ]:
        if src.exists():
            shutil.copyfile(src, case_dir / dst_name)

    summary = {
        "id": case["id"],
        "name": case["name"],
        "expected_label": case.get("expected_label", "vulnerable"),
        "protocol_family": case["protocol_family"],
        "bug_family": case["bug_family"],
        "property_family": case["property_family"],
        "expected_issue": case["expected_issue"],
        "source_reference": case["source_reference"],
    }
    save_json(case_dir / "case_metadata.json", summary)


def run_benchmark(manifest_path: Path, limit: int = 0, case_id: str = ""):
    manifest = load_json(manifest_path)
    results_root = ROOT / manifest.get("results_dir", "business-bench-data/results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cases = manifest["cases"]
    if case_id:
        cases = [case for case in cases if case["id"] == case_id]
    if limit and limit > 0:
        cases = cases[:limit]

    summary_rows = []
    for idx, case in enumerate(cases, start=1):
        print(f"\n=== [{idx}/{len(cases)}] Running {case['id']} ===")
        status = "ok"
        error_message = ""
        try:
            prepare_case_inputs(case)
            result = build_explainer_only_graph().invoke({})
            print(result)
            persist_case_outputs(case, run_dir)
        except Exception as e:
            status = "failed"
            error_message = str(e)
            print(f"Case failed: {case['id']} -> {e}")
            persist_case_outputs(case, run_dir)

        summary_rows.append(
            {
                "id": case["id"],
                "name": case["name"],
                "status": status,
                "error": error_message,
                "result_dir": str((run_dir / case["id"]).relative_to(ROOT)),
            }
        )

    save_json(
        run_dir / "benchmark_summary.json",
        {
            "manifest": str(manifest_path.relative_to(ROOT)),
            "run_dir": str(run_dir.relative_to(ROOT)),
            "cases": summary_rows,
        },
    )
    print(f"\nSaved benchmark results to: {run_dir}")


def main():
    os.chdir(ROOT)  # The stage implementations use project-relative artifact paths.
    parser = argparse.ArgumentParser(description="Run the CONVERA LangGraph flow or business-logic benchmark.")
    parser.add_argument(
        "--benchmark",
        default="",
        help="Path to business benchmark manifest JSON. If omitted, runs Fusion and Reason nodes.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional case limit for benchmark mode.")
    parser.add_argument("--case-id", default="", help="Optional single benchmark case id to run.")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(
            (ROOT / args.benchmark).resolve() if not os.path.isabs(args.benchmark) else Path(args.benchmark),
            limit=args.limit,
            case_id=args.case_id,
        )
    else:
        run_single()


if __name__ == "__main__":
    main()
