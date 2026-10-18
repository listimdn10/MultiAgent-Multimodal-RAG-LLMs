# RQ1 — Detection and baseline comparison

RQ1 evaluates conventional detection, business-logic incident detection, and
baseline comparisons.  The paper reports conventional Accuracy/Precision/
Recall/F1 and business-logic IDR/RCVR separately.

## Canonical implementations

| Evaluation | Implementation | Example command |
| --- | --- | --- |
| VERITE/CONVERA business-logic benchmark | `rw_benchmark.py` | `python experiment\run.py verite-convera run --limit-incidents 1` |
| DeFiLogic-Extended/CONVERA | `defilogic_extended/run_convera.py` | `python experiment\run.py defi-convera --case <case-id> --repetitions 1` |
| GPTLens on frozen VERITE inputs | `ablation_study/run_gptlens_verite.py` | `python experiment\run.py gptlens-verite --case sut` |
| GPTLens on DeFiLogic-Extended | `defilogic_extended/run_gptlens_mistral.py` | `python experiment\run.py gptlens-defi --case <case-id>` |
| VulnHuntGPT on VERITE or DeFiLogic-Extended | `defilogic_extended/run_vulnhuntgpt_mistral.py` | `python experiment\run.py vulnhunt --dataset defi --limit 1` |

The frozen VERITE GPTLens runner sends only Solidity source to the model.
Gold labels are read after inference to aggregate IDR.  Textual RCVR requires a
separate manual annotation of a score-qualified, benchmark-matching root
cause; it is not inferred merely from a shared coarse family label.

## Outputs

Each runner writes a self-contained output directory in its canonical results
location.  Consult its `summary.json` for the executed settings, denominator,
and metric definition before using a result in the manuscript.
