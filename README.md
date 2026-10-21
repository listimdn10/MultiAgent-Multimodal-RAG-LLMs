# CONVERA: Reproducibility Package

This repository contains the implementation and evaluation artifacts for
CONVERA, a multi-agent framework for auditing financial business-logic
vulnerabilities in Solidity smart contracts.  The reviewer-facing scope is the
property-guided Reason Agent, the VERITE benchmark driver, the ablation study,
and the DeFiLogic-Extended real-world evaluation.  Older prototype scripts at
the repository root are retained for provenance but are not needed to reproduce
the reported experiments.

## What to read first

| Goal | Entry point | Documentation |
| --- | --- | --- |
| Run the VERITE benchmark | `rw_benchmark.py` | `rw_dataset/` and this README |
| Browse experiments by research question | `experiment/` | `experiment/README.md` |
| Reproduce Reason-Agent ablations | `ablation_study/run_ablation.py` | `ablation_study/README.md` |
| Run GPTLens on frozen VERITE inputs | `ablation_study/run_gptlens_verite.py` | this README |
| Inspect/run DeFiLogic-Extended | `defilogic_extended/run_convera.py` | `defilogic_extended/README.md` |
| Run GPTScan on DeFiLogic-Extended | `defilogic_extended/run_gptscan_mistral.py` | `defilogic_extended/README.md` |
| Inspect FinanceFuzz compatibility | `defilogic_extended/run_financefuzz.py` | `defilogic_extended/BASELINE_STATUS.md` |
| Build the manuscript | `698ea86bae688ce64549ce91/` | its own repository README/settings |

## Scope and repository layout

```text
explainer.py                 Property synthesis, scenario/spec generation,
                             lightweight validation, and final reasoning.
rw_benchmark.py              VERITE evaluation driver and metric aggregation.
rw_dataset/                  VERITE inputs, manifests, cached source, and outputs.
ablation_study/              Five-setting Reason-Agent ablation runner.
defilogic_extended/          Source-provenanced external incident evaluation.
business-bench-data/         Small curated business-logic regression benchmark.
```

`FinanceFuzz/` is a local Git checkout retained for baseline investigation; it
is not modified by the CONVERA runners. The reproducibility scripts configure
third-party baselines locally rather than publishing their source or
credentials in this repository.

## Security and configuration

Copy `.env.example` to `.env` and fill only the provider you intend to use.
`.env` is ignored and must never be committed.  Existing credentials that may
have appeared in historical revisions should be revoked and replaced before a
public release; removing a file from a new commit does not erase Git history.

The active pipelines select providers as follows:

- `LLM_PROVIDER=mistral` with `MISTRAL_API_KEY` uses the Mistral-compatible
  endpoint.
- `OPENAI_API_KEY` enables the OpenAI path used by the ablation runner.
- Without a cloud key, `explainer.py` can use the configured Ollama endpoint.

Live reruns incur model-provider cost and may differ due to model updates.  The
saved aggregate summaries and per-case artifacts should be used to audit the
numbers reported in the manuscript; a rerun tests reproducibility rather than
bitwise determinism.

## Environment setup

The experiments were developed with Python 3.10 on Windows.  Create an
isolated environment and install the pinned research environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

The complete `requirements.txt` is intentionally broad because it also covers
the historical multimodal/RAG prototype.  A GPU is useful for embedding/model
work, but the reviewer-facing reasoning scripts can use a cloud backend.
Solidity compilation requires versions matching the benchmark contract;
install `solc-select`/`py-solc-x` compilers as prompted by the runner.

## Verification order

Start with non-billable checks:

```powershell
python -m py_compile explainer.py rw_benchmark.py `
  ablation_study\run_ablation.py `
  ablation_study\run_gptlens_verite.py `
  defilogic_extended\run_convera.py `
  defilogic_extended\run_gptlens_mistral.py `
  defilogic_extended\run_vulnhuntgpt_mistral.py `
  defilogic_extended\run_gptscan_mistral.py `
  defilogic_extended\run_financefuzz.py

python rw_benchmark.py --help
python ablation_study\run_ablation.py --help
python ablation_study\run_gptlens_verite.py --help
python defilogic_extended\run_convera.py --help
python defilogic_extended\run_gptscan_mistral.py --help
```

Then use the one-incident smoke tests in the subdirectory READMEs before any
full paid run.  Each runner writes a self-contained run directory with its
configuration, per-case outputs, and summary.  Do not overwrite a completed
run; use the runner's resume option where available.

## Metrics and comparability

- **IDR** is incident detection recall: an incident is detected when at least
  one candidate receives the benchmark-defined positive or evidence-needed
  verdict.
- **RCVR** measures root-cause validation recall.  It is mechanism/property
  level only when the dataset provides such labels and the method produces the
  required evidence.  VERITE's available labels support only a coarse
  bug-family variant.
- Do not aggregate RCVR across datasets with different label granularity.
  FinanceFuzz is not assigned an RCVR score because its property-testing
  signals do not produce the candidate-level root-cause/evidence object used by
  CONVERA.

## GPTLens VERITE protocol reproduction

`ablation_study/run_gptlens_verite.py` executes the official GPTLens
auditor--critic--ranker prompts over the frozen VERITE source inputs. The
model receives only Solidity source; the incident name and gold bug family are
used only after inference for incident-level aggregation. Run a no-cost CLI
check first, then a one-incident smoke test before the full paid experiment:

```powershell
python ablation_study\run_gptlens_verite.py --help
python ablation_study\run_gptlens_verite.py --case sut
python ablation_study\run_gptlens_verite.py
```

Each run writes `summary.json`, `results.csv`, and parsed auditor, critic, and
ranker outputs below `ablation_study/results/`. IDR uses the score threshold
specified in `summary.json`. Textual RCVR is deliberately a separate manual
annotation step: a score-qualified finding must match the benchmark root cause,
not merely share its coarse bug-family label. Store that audit trail beside the
run output before reporting RCVR.

## Third-party baselines

The reproduction scripts label non-original backend executions explicitly.
For example, GPTLens and the legacy VulnHunt-GPT reproduction use Mistral when
configured and therefore do not claim to reproduce the original GPT backend.
The GPTScan reproduction uses the official rules at a pinned upstream commit
with the public OpenAI-compatible endpoint patch in
`defilogic_extended/patches/`; its setup and manual RCVR-review protocol are
documented in `defilogic_extended/README.md`.
FinanceFuzz is tested separately because its released runner accepts a single
source file, legacy EVM revisions, and local deployment; details and observed
compatibility limits are documented in `defilogic_extended/BASELINE_STATUS.md`.

## Licensing and data provenance

The DeFiLogic-Extended source cache is retrieved from public verified-contract
explorers.  `source_status.json` and explorer response files preserve the
address, chain, compiler version, and retrieval provenance.  Do not use
DeFiHackLabs exploit-reproduction comments as model input: they contain
post-incident information and would leak labels.
