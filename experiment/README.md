# CONVERA experiment index

This directory is the reviewer-facing map from the paper's research questions
to the canonical implementation and evaluation artifacts.  It intentionally
does not duplicate implementation files: every command dispatches to the
single source of truth at the repository root.

| Research question | Directory | Automated entry point |
| --- | --- | --- |
| RQ1: detection and baseline comparison | `rq1_detection/` | `python experiment/run.py <target>` |
| RQ2: component and property-guided ablations | `rq2_components/` | `python experiment/run.py ablation ...` |
| RQ3: SLM/RAG effectiveness | `rq3_slm_rag/` | `download_models.py` and `eval_rq3.py` (paired exploratory evaluation; original held-out split unavailable) |
| RQ4: knowledge-node and knowledge-graph effects | `rq4_knowledge/` | documented implementation artifacts |
| RQ5: audit-report quality | `rq5_report_quality/` | `python experiment/run.py build-rq5-inputs` then `judge-reports` (new evaluation on saved outputs; original PoPoC unavailable) |

## Setup

Run from the repository root after creating the environment described in the
top-level `README.md`.  The cloud-backed experiments require the intended
provider credential in `.env`; do not commit this file.

Use the launcher only for scripts that already expose a command-line interface:

```powershell
# List the available reproducible targets.
python experiment\run.py --help

# Show a target's own arguments without calling a model.
python experiment\run.py gptlens-verite --help
python experiment\run.py ablation --help
```

Generated outputs stay in their canonical ignored locations
(`ablation_study/results/`, `defilogic_extended/results/`, and
`rw_dataset/results/`).  They are not versioned because model outputs may be
large and can contain provider-specific nondeterminism.
