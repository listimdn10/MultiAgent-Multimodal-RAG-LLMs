"""Score the conventional run's explicit final summaries as binary predictions.

This is a descriptive, text-to-binary proxy, NOT an issue-matched or
vulnerability-family accuracy measure. A report about any vulnerability is
positive even if it describes a different flaw from the workbook label.
Operational failures and empty/ambiguous summaries abstain; they are never
silently converted to a negative prediction.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


NEGATIVE = re.compile(
    r"^(?:no\s+(?:(?:confirmed|genuine|exploitable|security)\s+)?"
    r"(?:vulnerabilit\w*|security\s+issues?)|false\s+positive|"
    r"the\s+contract\s+is\s+safe)\b",
    re.IGNORECASE,
)
POSITIVE = re.compile(
    r"\b(?:vulnerab\w*|exploit\w*|attack\w*|flaw\w*|risk\w*|"
    r"unsafe|violat\w*|unchecked|bypass\w*|issue\w*|"
    r"unprotected|missing\s+access\s+control)\b",
    re.IGNORECASE,
)


def classify(record: dict) -> tuple[str, str]:
    if record.get("status") != "ok":
        return "abstain", "pipeline_error"
    if record.get("consensus_decision") not in {"RAG", "Explainer", "Merged"}:
        return "abstain", "invalid_consensus_decision"
    summary = str(record.get("consensus_summary") or "").strip()
    if not summary:
        return "abstain", "empty_final_summary"
    if NEGATIVE.search(summary):
        return "negative", "explicit_negative_summary"
    if POSITIVE.search(summary):
        return "positive", "explicit_positive_summary"
    return "abstain", "ambiguous_final_summary"


def fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def score(progress: dict) -> dict:
    rows = progress["results"]
    selected = progress["selected_cases"]
    if len(rows) != selected:
        raise ValueError(f"Only {len(rows)}/{selected} selected cases attempted")
    confusion = {key: 0 for key in ("tp", "fp", "tn", "fn")}
    abstentions: dict[str, int] = {}
    predictions = []
    for record in rows:
        gold = "negative" if record["gold_label"] == "safe" else "positive"
        predicted, basis = classify(record)
        predictions.append({
            "excel_row": record["excel_row"],
            "gold_label": record["gold_label"],
            "gold_binary": gold,
            "predicted_binary": predicted,
            "basis": basis,
            "artifact_dir": record.get("artifact_dir", ""),
        })
        if predicted == "abstain":
            abstentions[basis] = abstentions.get(basis, 0) + 1
        elif gold == "positive" and predicted == "positive":
            confusion["tp"] += 1
        elif gold == "negative" and predicted == "positive":
            confusion["fp"] += 1
        elif gold == "negative" and predicted == "negative":
            confusion["tn"] += 1
        else:
            confusion["fn"] += 1

    tp, fp, tn, fn = (confusion[key] for key in ("tp", "fp", "tn", "fn"))
    scored = tp + fp + tn + fn
    gold_positive = sum(row["gold_binary"] == "positive" for row in predictions)
    gold_negative = selected - gold_positive
    return {
        "metric_scope": "binary any-vulnerability, text-derived proxy; not issue-matched",
        "denominators": {
            "original_excel_cases": progress["total_excel_cases"],
            "selected_compilable_cases": selected,
            "pipeline_completed": progress["completed"],
            "scored_nonabstaining_cases": scored,
            "gold_positive": gold_positive,
            "gold_safe": gold_negative,
        },
        "confusion_scored_only": confusion,
        "abstentions": abstentions,
        "metrics_scored_only": {
            "accuracy": fraction(tp + tn, scored),
            "precision": fraction(tp, tp + fp),
            "recall": fraction(tp, tp + fn),
            "specificity": fraction(tn, tn + fp),
            "f1": fraction(2 * tp, 2 * tp + fp + fn),
        },
        "coverage": {
            "pipeline_completion": fraction(progress["completed"], selected),
            "scoreable_output": fraction(scored, selected),
        },
        "end_to_end_accuracy_abstentions_count_wrong": fraction(tp + tn, selected),
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    progress_path = args.run_dir / "progress.json"
    result = score(json.loads(progress_path.read_text(encoding="utf-8")))
    output = args.output or args.run_dir / "binary_score.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "predictions"}, indent=2))
    print(f"Case-level audit trail: {output}")


if __name__ == "__main__":
    main()
