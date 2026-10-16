"""Render a reproducibility comparison for the business-logic ablation study.

The ``repeated`` values are read from the completed three-repeat experiment.
The ``reported`` values are the single-run values currently plotted in
``Images/pgev_benchmark.svg``.  The output is a self-contained SVG so it can
be inspected in a browser or imported into a manuscript as vector artwork.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "ablation_study" / "results" / "run_20261011_125043" / "aggregate_summary.json"
DEFAULT_OUTPUT_DIR = ROOT / "698ea86bae688ce64549ce91" / "Images"
SUMMARY = DEFAULT_SUMMARY
OUTPUT = DEFAULT_OUTPUT_DIR / "pgev_ablation_comparison.svg"
PNG_OUTPUT = OUTPUT.with_suffix(".png")
REPEATED_PNG_OUTPUT = ROOT / "698ea86bae688ce64549ce91" / "Images" / "pgev_ablation_repeated.png"
REPEATED_SVG_OUTPUT = REPEATED_PNG_OUTPUT.with_suffix(".svg")

# Values in the original, single-run CONVERA-ablation panel of Fig. PGEV.
REPORTED = {
    "without_property_synthesis": {"idr": 70.0, "rcvr": 10.0},
    "without_scenario_generation": {"idr": 86.7, "rcvr": 30.0},
    "without_validator_spec_synthesis": {"idr": 80.0, "rcvr": None},
    "without_lightweight_validation": {"idr": 83.3, "rcvr": None},
    "full_convera": {"idr": 93.3, "rcvr": 6.7},
}

LABELS = {
    "without_property_synthesis": ("Without Property", "Synthesis"),
    "without_scenario_generation": ("Without Scenario", "Generation"),
    "without_validator_spec_synthesis": ("Without Validator-Spec", "Synthesis"),
    "without_lightweight_validation": ("Without Lightweight", "Validation"),
    "full_convera": ("Full", "CONVERA"),
}


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def pct(value: float) -> str:
    return f"{value:.1f}%"


def y(value: float) -> float:
    return 735 - value * 5.2


def render_png(repeated: dict[str, dict[str, float | None]]) -> None:
    """Render a manuscript-ready PNG counterpart of the vector figure."""
    names = list(REPORTED)
    x = list(range(len(names)))
    offsets = (-0.30, -0.10, 0.10, 0.30)
    colors = ("#2f6fbb", "#81b4e3", "#e07a2f", "#f6b980")
    series = (
        ("Reported IDR (single run)", [REPORTED[name]["idr"] for name in names], None),
        ("Repeated IDR", [repeated[name]["idr"] for name in names], [repeated[name]["idr_sd"] for name in names]),
        ("Reported RCVR (single run)", [REPORTED[name]["rcvr"] for name in names], None),
        ("Repeated RCVR", [repeated[name]["rcvr"] for name in names], [repeated[name]["rcvr_sd"] for name in names]),
    )
    fig, ax = plt.subplots(figsize=(18, 9.8), dpi=100)
    ax.set_facecolor("#f7f8f5")
    bars = []
    for offset, color, (legend, values, errors) in zip(offsets, colors, series):
        valid = [0 if value is None else value for value in values]
        yerr = None if errors is None else [0 if value is None else value for value in errors]
        container = ax.bar(
            [value + offset for value in x], valid, width=0.18, color=color,
            label=legend, yerr=yerr, capsize=4, error_kw={"ecolor": "#25313c", "elinewidth": 1.2},
        )
        bars.append((container, values, errors))
    for offset, (_, values, errors) in zip(offsets, series):
        for pos, value, error in zip(x, values, errors or [None] * len(values)):
            if value is None:
                ax.text(pos + offset, 2.0, "N/A", ha="center", va="bottom", fontsize=12, fontweight="bold", color="#8a5a32")
            else:
                label = pct(value) if error is None else f"{pct(value)}±{error:.1f}"
                ax.text(pos + offset, min(106, value + (error or 0) + 2.0), label, ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Recall (%)", fontsize=18)
    ax.set_xticks(x, ["\n".join(LABELS[name]) for name in names], fontsize=14)
    ax.grid(axis="y", color="#d8dee4", linewidth=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("CONVERA Ablation: Reported versus Repeated Results", fontsize=24, fontweight="bold", pad=24)
    ax.text(0.5, 1.01, "30 business-logic incidents; repeated results report mean ± population standard deviation over three runs", transform=ax.transAxes, ha="center", fontsize=14, color="#5d6872")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False, fontsize=14)
    fig.text(0.5, 0.015, "RCVR is N/A when Validator-Spec Synthesis or Lightweight Validation is disabled.", ha="center", fontsize=12, color="#52606b")
    fig.subplots_adjust(bottom=0.27, top=0.86, left=0.08, right=0.98)
    fig.savefig(PNG_OUTPUT, dpi=100)
    plt.close(fig)


def render_repeated_figure(repeated: dict[str, dict[str, float | None]]) -> None:
    """Render the manuscript figure using only the repeated experiment."""
    names = list(REPORTED)
    x = list(range(len(names)))
    idr = [repeated[name]["idr"] for name in names]
    idr_sd = [repeated[name]["idr_sd"] for name in names]
    rcvr = [repeated[name]["rcvr"] for name in names]
    rcvr_sd = [repeated[name]["rcvr_sd"] for name in names]
    fig, ax = plt.subplots(figsize=(18, 9.8), dpi=100)
    ax.set_facecolor("#f7f8f5")
    width = 0.28
    ax.bar(
        [value - width / 2 - 0.02 for value in x], idr, width=width,
        color="#2f6fbb", label="IDR", yerr=idr_sd, capsize=5,
        error_kw={"ecolor": "#25313c", "elinewidth": 1.2},
    )
    rcvr_values = [0 if value is None else value for value in rcvr]
    rcvr_errors = [0 if value is None else value for value in rcvr_sd]
    ax.bar(
        [value + width / 2 + 0.02 for value in x], rcvr_values, width=width,
        color="#e07a2f", label="RCVR", yerr=rcvr_errors, capsize=5,
        error_kw={"ecolor": "#25313c", "elinewidth": 1.2},
    )
    for pos, value, error in zip(x, idr, idr_sd):
        ax.text(pos - width / 2 - 0.02, min(106, value + error + 2.0), f"{pct(value)}±{error:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    for pos, value, error in zip(x, rcvr, rcvr_sd):
        if value is None:
            ax.text(pos + width / 2 + 0.02, 2.0, "N/A", ha="center", va="bottom", fontsize=12, fontweight="bold", color="#8a5a32")
        else:
            ax.text(pos + width / 2 + 0.02, min(106, value + error + 2.0), f"{pct(value)}±{error:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Recall (%)", fontsize=18)
    ax.set_xticks(x, ["\n".join(LABELS[name]) for name in names], fontsize=14)
    ax.grid(axis="y", color="#d8dee4", linewidth=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Property-Grounded Evidence Validation Benchmark", fontsize=24, fontweight="bold", pad=24)
    ax.text(0.5, 1.01, "30 business-logic incidents; values report mean ± population standard deviation over three runs", transform=ax.transAxes, ha="center", fontsize=14, color="#5d6872")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=15)
    fig.text(0.5, 0.02, "RCVR is N/A when Validator-Spec Synthesis or Lightweight Validation is disabled.", ha="center", fontsize=12, color="#52606b")
    fig.subplots_adjust(bottom=0.23, top=0.86, left=0.08, right=0.98)
    fig.savefig(REPEATED_PNG_OUTPUT, dpi=100)
    fig.savefig(REPEATED_SVG_OUTPUT)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the repeated CONVERA ablation results from an aggregate summary."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Path to aggregate_summary.json (default: completed three-repeat run).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for SVG/PNG figures (default: manuscript Images directory).",
    )
    return parser.parse_args()


def main() -> None:
    global SUMMARY, OUTPUT, PNG_OUTPUT, REPEATED_PNG_OUTPUT, REPEATED_SVG_OUTPUT
    args = parse_args()
    SUMMARY = args.summary.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT = output_dir / "pgev_ablation_comparison.svg"
    PNG_OUTPUT = OUTPUT.with_suffix(".png")
    REPEATED_PNG_OUTPUT = output_dir / "pgev_ablation_repeated.png"
    REPEATED_SVG_OUTPUT = REPEATED_PNG_OUTPUT.with_suffix(".svg")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    repeated = {}
    for name, data in summary.items():
        rcvr = data["rcvr"]
        repeated[name] = {
            "idr": data["idr"]["mean"] * 100,
            "idr_sd": data["idr"]["population_std"] * 100,
            "rcvr": None if rcvr is None else rcvr["mean"] * 100,
            "rcvr_sd": None if rcvr is None else rcvr["population_std"] * 100,
        }

    s: list[str] = []
    add = s.append
    add('<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="980" viewBox="0 0 1800 980">')
    add('<rect width="1800" height="980" fill="#ffffff"/>')
    add("""<style>
text { font-family: Arial, Helvetica, sans-serif; fill: #25313c; }
.title { font-size: 34px; font-weight: 700; } .subtitle { font-size: 20px; fill: #5d6872; }
.axis { font-size: 18px; fill: #52606b; } .label { font-size: 16px; }
.value { font-size: 13px; font-weight: 700; } .legend { font-size: 18px; font-weight: 700; }
</style>""")
    add('<text x="900" y="54" text-anchor="middle" class="title">CONVERA Ablation: Reported versus Repeated Results</text>')
    add('<text x="900" y="88" text-anchor="middle" class="subtitle">30 business-logic incidents; repeated results report mean ± population standard deviation over three runs</text>')
    add('<rect x="105" y="126" width="1590" height="610" rx="6" fill="#f7f8f5"/>')

    for tick in range(0, 101, 20):
        yy = y(tick)
        add(f'<line x1="105" y1="{yy:.1f}" x2="1695" y2="{yy:.1f}" stroke="#d8dee4" stroke-width="1"/>')
        add(f'<text x="91" y="{yy + 6:.1f}" text-anchor="end" class="axis">{tick}</text>')
    add('<text x="35" y="450" text-anchor="middle" class="axis" transform="rotate(-90 35 450)">Recall (%)</text>')
    add('<line x1="105" y1="215" x2="105" y2="735" stroke="#89949e" stroke-width="1.5"/>')
    add('<line x1="105" y1="735" x2="1695" y2="735" stroke="#89949e" stroke-width="1.5"/>')

    # Each method has four bars: old/new IDR, followed by old/new RCVR.
    starts = [170, 480, 790, 1100, 1410]
    colors = ("#2f6fbb", "#81b4e3", "#e07a2f", "#f6b980")
    for start, name in zip(starts, REPORTED):
        old = REPORTED[name]
        new = repeated[name]
        values = (old["idr"], new["idr"], old["rcvr"], new["rcvr"])
        sds = (None, new["idr_sd"], None, new["rcvr_sd"])
        for i, (value, sd, color) in enumerate(zip(values, sds, colors)):
            xx = start + i * 48
            if value is None:
                add(f'<text x="{xx + 18}" y="720" text-anchor="middle" class="value" fill="#8a5a32">N/A</text>')
                continue
            yy = y(value)
            height = 735 - yy
            add(f'<rect x="{xx}" y="{yy:.1f}" width="36" height="{height:.1f}" fill="{color}"/>')
            if sd is not None and sd > 0:
                top, bottom = y(min(100, value + sd)), y(max(0, value - sd))
                mid = xx + 18
                add(f'<line x1="{mid}" y1="{top:.1f}" x2="{mid}" y2="{bottom:.1f}" stroke="#25313c" stroke-width="1.5"/>')
                add(f'<line x1="{mid - 6}" y1="{top:.1f}" x2="{mid + 6}" y2="{top:.1f}" stroke="#25313c" stroke-width="1.5"/>')
                add(f'<line x1="{mid - 6}" y1="{bottom:.1f}" x2="{mid + 6}" y2="{bottom:.1f}" stroke="#25313c" stroke-width="1.5"/>')
            label = pct(value) if sd is None else f"{pct(value)}±{sd:.1f}"
            label_y = max(205, yy - 9)
            add(f'<text x="{xx + 18}" y="{label_y:.1f}" text-anchor="middle" class="value">{esc(label)}</text>')

        label_x = start + 90
        line1, line2 = LABELS[name]
        add(f'<text x="{label_x}" y="770" text-anchor="middle" class="label"><tspan x="{label_x}">{esc(line1)}</tspan><tspan x="{label_x}" dy="21">{esc(line2)}</tspan></text>')

    add('<g transform="translate(365 900)">')
    legend = [
        ("#2f6fbb", "Reported IDR (single run)"),
        ("#81b4e3", "Repeated IDR"),
        ("#e07a2f", "Reported RCVR (single run)"),
        ("#f6b980", "Repeated RCVR"),
    ]
    for i, (color, text) in enumerate(legend):
        xx = (i % 2) * 380
        yy = (i // 2) * 34
        add(f'<rect x="{xx}" y="{yy - 17}" width="28" height="20" fill="{color}"/>')
        add(f'<text x="{xx + 40}" y="{yy}" class="legend">{text}</text>')
    add('</g>')
    add('<text x="900" y="970" text-anchor="middle" class="axis">RCVR is N/A when Validator-Spec Synthesis or Lightweight Validation is disabled.</text>')
    add('</svg>')
    OUTPUT.write_text("\n".join(s), encoding="utf-8")
    render_png(repeated)
    render_repeated_figure(repeated)
    print(f"Wrote {OUTPUT}")
    print(f"Wrote {PNG_OUTPUT}")
    print(f"Wrote {REPEATED_PNG_OUTPUT}")
    print(f"Wrote {REPEATED_SVG_OUTPUT}")


if __name__ == "__main__":
    main()
