# Reproducible CONVERA ablation study

This directory replaces the hand-entered property-grounded ablation values in the paper with a reproducible experiment. It evaluates five pipeline settings over the same cached contract source and upstream detector signal:

- `full_convera`
- `without_property_synthesis`
- `without_scenario_generation`
- `without_validator_spec_synthesis`
- `without_lightweight_validation`

The runner writes every per-case Reason-Agent output, an address-level CSV, an incident-level summary, and a cross-repetition aggregate summary. No value is imported from the manuscript figure.

## Metric definitions

- **IDR**: an incident has at least one `True Positive` or `Needs More Evidence` verdict. This retains the benchmark convention that an unresolved candidate is not treated as confirmed-safe.
- **RCVR**: an incident has at least one `True Positive` whose inferred root-cause family matches the benchmark family and whose Lightweight Validation status is `violated`.
- RCVR is `null` when either Validator-Spec Synthesis or Lightweight Validation is disabled, because those settings cannot produce the evidence required by the metric.

## Run

Run from `C:\D\KLTN` using the environment already used by CONVERA:

```powershell
# Copy the fixed source and upstream detector signals from the completed VERITE run.
python ablation_study\run_ablation.py prepare-inputs

# Recommended journal experiment: all five settings, three independent runs.
python ablation_study\run_ablation.py run --repetitions 3
```

If a run stops because of an API quota or network error, preserve its completed
outputs and retry only failed cases:

```powershell
python ablation_study\run_ablation.py run --repetitions 3 --resume-run-dir ablation_study\results\run_YYYYMMDD_HHMMSS
```

For a cheap smoke test before the complete run:

```powershell
python ablation_study\run_ablation.py prepare-inputs --limit-incidents 1
python ablation_study\run_ablation.py run --limit-incidents 1 --setting full_convera --setting without_scenario_generation --repetitions 1
```

The script uses the stored source/fusion signals from `run_20260827_233023`, the completed 30-incident VERITE run used by the paper, so all settings receive exactly the same upstream candidates. It uses `OPENAI_MODEL` and `OPENAI_API_KEY` when present, otherwise the configured Ollama model. The full three-repetition experiment makes many LLM calls; review the selected model and API cost before executing it.

## Frozen-input protocol and output audit trail

The ablation does **not** rerun the upstream parser, fusion model, or RAG
retrieval. `prepare-inputs` snapshots the same contract source and detector
candidate into `inputs/`, then every configuration consumes that snapshot. This
isolates the causal question: how does removing one Reason-Agent stage affect
the decision from a fixed upstream candidate?

The following files are created under `results/run_<timestamp>/` and are
deliberately Git-ignored because they contain provider output and can be large:

```text
run_config.json               Selected model, settings, limits, and timestamp.
repeat_01/.../case.json       Raw per-case output for one setting/repetition.
repeat_01/.../incident.csv    Incident-level IDR/RCVR scoring decisions.
aggregate_summary.json        Mean and sample standard deviation across repeats.
aggregate_summary.csv         Table-friendly form of the aggregate statistics.
```

Before using a number in the paper, verify that the result directory contains
all five configurations and three completed repeats, inspect the denominator in
each `incident.csv`, and quote the exact model/provider. A resumed run reuses
only an already persisted successful case; it does not silently replace a
failed response with a negative result.

## Rendering the paper figure

The figure generator reads only `aggregate_summary.json`, not hard-coded paper
values. For example:

```powershell
python ablation_study\render_ablation_comparison.py `
  --summary ablation_study\results\run_YYYYMMDD_HHMMSS\aggregate_summary.json `
  --output-dir ablation_study\results\run_YYYYMMDD_HHMMSS\figures
```

Use the generated image and its JSON summary together when updating the LaTeX
table or caption. Do not compare the disabled-validation configurations by
RCVR: the value is intentionally `null`, not zero, because those configurations
cannot emit the validator evidence required by the metric.

## Interpretation constraint

In the current implementation, Scenario Generation conditions Validator-Spec Synthesis and Final Reasoning, but Lightweight Validation does not execute generated transaction sequences. Therefore the scenario ablation measures the effect of scenario context on specification and reasoning, not the benefit of dynamic exploit execution. Report this limitation explicitly.
