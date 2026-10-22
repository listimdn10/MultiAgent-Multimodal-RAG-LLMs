"""Dispatch reviewer-facing experiment targets to their canonical scripts.

The wrapper prevents duplicated copies of evaluation code under ``experiment``.
All arguments after the target are forwarded unchanged to the selected script.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    # RQ1: detection/baseline evaluation.
    "verite-convera": ROOT / "rw_benchmark.py",
    "defi-convera": ROOT / "defilogic_extended" / "run_convera.py",
    "gptlens-verite": ROOT / "ablation_study" / "run_gptlens_verite.py",
    "gptlens-defi": ROOT / "defilogic_extended" / "run_gptlens_mistral.py",
    "vulnhunt": ROOT / "defilogic_extended" / "run_vulnhuntgpt_mistral.py",
    "gptscan-verite": ROOT / "rw_dataset" / "run_gptscan_verite.py",
    "score-gptscan-verite": ROOT / "rw_dataset" / "score_gptscan_verite.py",
    "gptscan-defi": ROOT / "defilogic_extended" / "run_gptscan_mistral.py",
    "repeat-baselines": ROOT / "experiment" / "rq1_detection" / "repeat_baselines.py",
    # RQ2: property-guided component study.
    "ablation": ROOT / "ablation_study" / "run_ablation.py",
    "render-ablation": ROOT / "ablation_study" / "render_ablation_comparison.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a canonical CONVERA experiment without duplicating its implementation."
    )
    parser.add_argument("target", choices=sorted(TARGETS), help="Experiment target to execute.")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the target.")
    args = parser.parse_args()

    script = TARGETS[args.target]
    if not script.is_file():
        raise FileNotFoundError(f"Missing canonical script: {script}")

    # Root imports (e.g., explainer.py) must resolve exactly as when the
    # canonical script is invoked directly from the repository root.
    sys.path.insert(0, str(ROOT))
    sys.argv = [str(script), *args.args]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
