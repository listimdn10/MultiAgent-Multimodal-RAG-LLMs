"""Aggregate independently produced DeFiLogic-Extended repeat summaries.

This script never calls a model.  It preserves the per-repeat rows and computes
the mean and sample standard deviation only from already saved evaluation files.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", nargs="+", help="Per-repeat summary.json files")
    parser.add_argument("--output", required=True, help="Combined JSON output path")
    args = parser.parse_args()

    summaries = [load(Path(value)) for value in args.summary]
    if not summaries:
        raise ValueError("At least one summary is required.")
    denominators = {entry["eligible_incidents"] for entry in summaries}
    if len(denominators) != 1:
        raise ValueError("Cannot aggregate summaries with different eligible-incident counts.")

    statistics_by_metric: dict[str, dict[str, float]] = {}
    for metric in ("idr", "rcvr"):
        rates = [float(entry[metric]["rate"]) for entry in summaries]
        statistics_by_metric[metric] = {
            "mean": statistics.mean(rates),
            "sample_std": statistics.stdev(rates) if len(rates) > 1 else 0.0,
            "min": min(rates),
            "max": max(rates),
        }

    output = {
        "dataset": "DeFiLogic-Extended",
        "model": {"provider": "mistral", "name": "ministral-8b-latest"},
        "repetitions": len(summaries),
        "eligible_incidents": denominators.pop(),
        "metric_definition": summaries[0]["metric_definition"],
        "metric_statistics": statistics_by_metric,
        "runs": summaries,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(destination.resolve())


if __name__ == "__main__":
    main()
