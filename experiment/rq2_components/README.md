# RQ2 — Component and property-guided reasoning study

RQ2 evaluates the contribution of the architecture and the Property-Guided
Reason Agent.  Its reproducible core is the five-setting ablation over the same
frozen 35 address-level VERITE inputs:

- `full_convera`
- `without_property_synthesis`
- `without_scenario_generation`
- `without_validator_spec_synthesis`
- `without_lightweight_validation`

## Run

```powershell
# Snapshot fixed source and detector signals first.
python experiment\run.py ablation prepare-inputs

# Full paper configuration: five settings, three repetitions.
python experiment\run.py ablation run --repetitions 3

# Low-cost one-incident smoke test.
python experiment\run.py ablation prepare-inputs --limit-incidents 1
python experiment\run.py ablation run --limit-incidents 1 --setting full_convera --repetitions 1
```

The implementation is `ablation_study/run_ablation.py`; see
`ablation_study/README.md` for the frozen-input protocol, resumption, metric
definitions, and limitations.  Use `render-ablation` to regenerate the paper
figure from an aggregate summary:

```powershell
python experiment\run.py render-ablation --summary ablation_study\results\run_YYYYMMDD_HHMMSS\aggregate_summary.json
```

RCVR is intentionally `null` when Validator-Spec Synthesis or Lightweight
Validation is disabled because those configurations cannot produce the
evidence required by the strict metric.
