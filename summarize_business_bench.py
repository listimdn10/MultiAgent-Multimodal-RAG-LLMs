import argparse
import csv
import json
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_audit_report(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return {"raw": value}
    return {}


def first_property_family(explainer_output: dict) -> str:
    specs = explainer_output.get("validator_spec_bundle", {}).get("validator_specs", [])
    if specs:
        assertion = specs[0].get("assertion", {})
        return assertion.get("relation_type", specs[0].get("validation_mode", ""))
    return ""


def first_property_name(explainer_output: dict) -> str:
    props = explainer_output.get("property_bundle", {}).get("properties", [])
    if props:
        return props[0].get("name", "")
    return ""


def first_focus_condition(explainer_output: dict) -> str:
    scenarios = explainer_output.get("scenario_bundle", {}).get("scenarios", [])
    if scenarios:
        return scenarios[0].get("focus_condition", "")
    return ""


def classify_result(row):
    if row["validation_status"] == "violated":
        return "validated_violation"
    if row["validated_bug_type"] != row["upstream_bug_type"]:
        return "bug_type_refined_only"
    return "reasoning_only"


def build_rows(run_dir: Path):
    benchmark_summary_path = run_dir / "benchmark_summary.json"
    if benchmark_summary_path.exists():
        benchmark_summary = load_json(benchmark_summary_path)
        case_dirs = [ROOT / case_entry["result_dir"] for case_entry in benchmark_summary["cases"]]
    else:
        case_dirs = sorted(
            [
                p for p in run_dir.iterdir()
                if p.is_dir()
                and (p / "case_metadata.json").exists()
                and (p / "explainer_output.json").exists()
            ],
            key=lambda p: p.name,
        )

    rows = []
    for case_dir in case_dirs:
        meta = load_json(case_dir / "case_metadata.json")
        explainer = load_json(case_dir / "explainer_output.json")
        audit = parse_audit_report(explainer.get("Audit_report"))

        row = {
            "case_id": meta["id"],
            "case_name": meta["name"],
            "protocol_family": meta["protocol_family"],
            "expected_bug_family": meta["bug_family"],
            "expected_property_family": meta["property_family"],
            "expected_label": meta.get("expected_label", "vulnerable"),
            "upstream_bug_type": explainer.get("vuln_type", ""),
            "validated_bug_type": explainer.get("validated_bug_type", ""),
            "inferred_protocol_family": explainer.get("property_bundle", {}).get("protocol_family", ""),
            "primary_property": first_property_name(explainer),
            "validator_property_family": first_property_family(explainer),
            "focus_condition": first_focus_condition(explainer),
            "validation_status": explainer.get("validation_bundle", {}).get("aggregate_status", ""),
            "validators_run": ", ".join(explainer.get("validation_bundle", {}).get("executors_used", [])),
            "verdict": audit.get("verdict", ""),
            "violated_property": audit.get("violated_property", ""),
            "confidence": audit.get("confidence", ""),
            "result_class": "",
            "result_dir": str(case_dir.relative_to(ROOT)),
        }
        row["result_class"] = classify_result(row)
        rows.append(row)
    return rows


def compute_binary_metrics(rows, positive_fn):
    tp = fp = tn = fn = 0
    for row in rows:
        gold_positive = row["expected_label"] == "vulnerable"
        pred_positive = positive_fn(row)
        if gold_positive and pred_positive:
            tp += 1
        elif gold_positive and not pred_positive:
            fn += 1
        elif not gold_positive and pred_positive:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def write_csv(rows, path: Path):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, path: Path):
    headers = [
        "Case",
        "Expected Bug",
        "Upstream Type",
        "Validated Type",
        "Property Family",
        "Validation",
        "Verdict",
        "Class",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["case_id"],
                    row["expected_bug_family"],
                    row["upstream_bug_type"],
                    row["validated_bug_type"],
                    row["validator_property_family"],
                    row["validation_status"],
                    row["verdict"],
                    row["result_class"],
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(rows, path: Path):
    reasoning_metrics = compute_binary_metrics(rows, lambda row: row["verdict"] == "True Positive")
    validation_metrics = compute_binary_metrics(rows, lambda row: row["validation_status"] == "violated")
    summary = {
        "cases": rows,
        "counts": {
            "total": len(rows),
            "validated_violation": sum(r["result_class"] == "validated_violation" for r in rows),
            "bug_type_refined_only": sum(r["result_class"] == "bug_type_refined_only" for r in rows),
            "reasoning_only": sum(r["result_class"] == "reasoning_only" for r in rows),
        },
        "metrics": {
            "reasoning_level": reasoning_metrics,
            "validation_grounded": validation_metrics,
        }
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Benchmark run directory, e.g. business-bench-data/results/run_20260601_080557")
    args = parser.parse_args()

    run_dir = (ROOT / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    rows = build_rows(run_dir)

    write_csv(rows, run_dir / "business_bench_table.csv")
    write_markdown(rows, run_dir / "business_bench_table.md")
    write_json(rows, run_dir / "business_bench_summary.json")

    print(f"Saved: {run_dir / 'business_bench_table.csv'}")
    print(f"Saved: {run_dir / 'business_bench_table.md'}")
    print(f"Saved: {run_dir / 'business_bench_summary.json'}")


if __name__ == "__main__":
    main()
