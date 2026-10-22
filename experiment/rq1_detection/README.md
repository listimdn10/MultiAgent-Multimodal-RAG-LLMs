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
| GPTScan on VERITE | `rw_dataset/run_gptscan_verite.py` | `python experiment\run.py gptscan-verite --help` |
| GPTScan VERITE root-cause scoring | `rw_dataset/score_gptscan_verite.py` | `python experiment\run.py score-gptscan-verite --help` |
| GPTScan on DeFiLogic-Extended | `defilogic_extended/run_gptscan_mistral.py` | `python experiment\run.py gptscan-defi --help` |
| Additional GPTLens/VulnHuntGPT repeats | `experiment/rq1_detection/repeat_baselines.py` | `python experiment\run.py repeat-baselines --help` |

The frozen VERITE GPTLens runner sends only Solidity source to the model.
Gold labels are read after inference to aggregate IDR.  Textual RCVR requires a
separate manual annotation of a score-qualified, benchmark-matching root
cause; it is not inferred merely from a shared coarse family label.

## Outputs

Each runner writes a self-contained output directory in its canonical results
location.  Consult its `summary.json` for the executed settings, denominator,
and metric definition before using a result in the manuscript.

## Repeated GPTLens and VulnHuntGPT runs

`python experiment/rq1_detection/repeat_baselines.py` runs two new independent
repetitions of each baseline on VERITE and DeFiLogic-Extended. The existing
single runs serve as repeat 01; this command creates repeats 02 and 03 under
`rw_dataset/results/baseline_repeats_20261022/`. The four jobs run serially to
avoid competing for the Mistral API rate limit. `progress.json` and each
`repeat_XX/run.log` show progress. Each `resolved_summary.json` records the
prediction artifact directory used for every case; failed API calls are
retried separately rather than counted as non-detections. An unresolved case
stops the orchestrator and must be inspected before reporting a three-run
average.

The runner computes only IDR. RCVR must be adjudicated separately at incident
level against the benchmark root cause and code-level evidence. It must not be
equated with the IDR rate or inferred from a score-qualified finding alone.
For VERITE, the available reference is an incident-level coarse bug-family
label: credit requires a family-consistent causal explanation plus a relevant
code-level rationale, not merely the label. For DeFiLogic-Extended, credit
requires agreement with the annotated mechanism and evidence for that same
mechanism. Each repeat needs its own incident-level review before an RCVR
mean and standard deviation can be reported.
On DeFiLogic-Extended, the VulnHuntGPT source limit is 250,000 characters so
that the two cases that required a full-source retry in repeat 01 receive the
same complete source in repeats 02 and 03.
