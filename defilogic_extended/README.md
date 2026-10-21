# DeFiLogic-Extended: reviewer reproduction guide

DeFiLogic-Extended is a source-provenanced extension of the incident-level
VERITE evaluation. It asks whether an auditor can identify and explain
business-logic failures in public, deployed DeFi incidents.

The collection was curated from public DeFiHackLabs exploit reproductions, but
**those exploit files are never supplied to a model**. They can state the
exploit and root cause in comments, which would contaminate an evaluation.
Instead, the runners obtain verified deployed Solidity source and retain its
retrieval provenance.

## Contents

```text
manifest.json                 Incident metadata and evaluation labels.
sources/<case>/               Raw explorer response, normalized source, and provenance.
prepared/<case>/              Frozen model input derived only from verified source.
collect_sources.py            Fetch or refresh the verified-source cache.
run_convera.py                CONVERA evaluation and re-scoring utility.
run_gptlens_mistral.py        Prompt/ranking reproduction of GPTLens with Mistral.
run_vulnhuntgpt_mistral.py    Mistral-backed VulnHunt-GPT-style reproduction.
run_gptscan_mistral.py        Official GPTScan rules on the 15-incident subset.
run_financefuzz.py            Adapter for an external local FinanceFuzz checkout.
aggregate_repeats.py          Aggregate completed CONVERA repetitions.
BASELINE_STATUS.md            Executed-status and fairness limitations for baselines.
patches/                      Compatibility patches for third-party baselines.
results/                      Local generated outputs; intentionally ignored by Git.
third_party/                  Local third-party checkouts; intentionally ignored by Git.
```

`manifest.json` is required for scoring, but it is not passed to the model as
an exploit narrative. The model-facing input is created from `sources/` by
`run_convera.py --prepare-only`. Before reporting new results, inspect the
generated `prepared/<case>/input.json` and confirm that no incident write-up or
gold explanation has been appended to the prompt.

## Prerequisites

From the repository root, create the environment described in the top-level
README and configure one supported model provider. The executed pilot used
Mistral:

```powershell
Copy-Item .env.example .env
# Edit .env locally; never commit it.
# Set LLM_PROVIDER=mistral and MISTRAL_API_KEY=<your key>.
```

All commands below assume the current directory is the repository root
(`C:\D\KLTN`). Start with the help commands; they make no API request:

```powershell
python defilogic_extended\collect_sources.py --help
python defilogic_extended\run_convera.py --help
python defilogic_extended\run_gptlens_mistral.py --help
python defilogic_extended\run_vulnhuntgpt_mistral.py --help
```

## 1. Inspect or refresh source provenance

The committed `sources/` cache lets reviewers inspect the exact retrieved
contracts without relying on a live explorer. To refresh all cases, or only
one case, run:

```powershell
python defilogic_extended\collect_sources.py --refresh
python defilogic_extended\collect_sources.py --case <case-id> --refresh
```

For each usable case, verify `sources/<case-id>/source_status.json` records
`ready: true`, contract address, chain, compiler metadata, and retrieval
details. A case without verified, compilable source must be excluded from the
denominator; it is not a negative detection.

## 2. Run CONVERA

First build the frozen model inputs without a model call:

```powershell
python defilogic_extended\run_convera.py --prepare-only
```

Run one cheap smoke test before spending provider credits:

```powershell
python defilogic_extended\run_convera.py --case <case-id> --repetitions 1 `
  --run-dir defilogic_extended\results\smoke_<case-id>
```

For the full protocol, use three independent repetitions and a fresh output
directory:

```powershell
python defilogic_extended\run_convera.py --repetitions 3 `
  --run-dir defilogic_extended\results\convera_mistral_r3
python defilogic_extended\aggregate_repeats.py `
  --run-dir defilogic_extended\results\convera_mistral_r3
```

If a run is interrupted after model calls, do not overwrite it. Re-score the
saved outputs without additional calls:

```powershell
python defilogic_extended\run_convera.py --rescore-dir `
  defilogic_extended\results\convera_mistral_r3
```

The runner saves command configuration, prepared input, per-repeat JSON, and
an aggregate summary. Result folders are Git-ignored because they can contain
large provider outputs and may be regenerated.

## 3. Run prompt-based baselines

These commands are protocol reproductions, not claims that the original
authors' proprietary GPT models were rerun. The backend must be named in any
table or manuscript text.

```powershell
# GPTLens-style prompt/ranking pipeline on a case.
python defilogic_extended\run_gptlens_mistral.py --case <case-id> `
  --run-dir defilogic_extended\results\gptlens_smoke

# VulnHunt-GPT-style pipeline. Use --dataset defi for this collection,
# or --dataset verite to run the same harness over VERITE.
python defilogic_extended\run_vulnhuntgpt_mistral.py --dataset defi --limit 1 `
  --run-dir defilogic_extended\results\vulnhunt_smoke
```

`BASELINE_STATUS.md` distinguishes executed measurements from unexecuted or
non-comparable baselines. Do not turn provider failures, missing imports, or
unavailable contract dependencies into false negatives.

## 4. GPTScan reproduction

The GPTScan experiment runs the official rules from the upstream revision
`29a174773bd526c32ab7d6a8c78a63870330ccc7`. The public patch below does not
add rules or benchmark-specific prompts: it only permits an OpenAI-compatible
endpoint/model to be configured through environment variables and strips a
single Markdown code fence from JSON responses. The runner selects the Solidity
compiler for each incident from the committed verified-source metadata.

The runner was developed on Windows and expects the baseline virtual
environment at `third_party/GPTScan/.venv`.

```powershell
git clone https://github.com/GPTScan/GPTScan.git `
  defilogic_extended\third_party\GPTScan
git -C defilogic_extended\third_party\GPTScan checkout 29a174773bd526c32ab7d6a8c78a63870330ccc7
py -3.10 -m venv defilogic_extended\third_party\GPTScan\.venv
defilogic_extended\third_party\GPTScan\.venv\Scripts\python -m pip install --upgrade pip
defilogic_extended\third_party\GPTScan\.venv\Scripts\pip install -r `
  defilogic_extended\third_party\GPTScan\requirements.txt
git -C defilogic_extended\third_party\GPTScan apply `
  ..\..\patches\gptscan-openai-compatible.patch

# Set MISTRAL_API_KEY in the ignored .env file, then use a no-cost CLI check.
python defilogic_extended\run_gptscan_mistral.py --help

# One-incident smoke test before the full paid run.
python defilogic_extended\run_gptscan_mistral.py `
  --case cream_yusd_collateral_valuation --repetitions 1 `
  --run-dir defilogic_extended\results\gptscan_smoke

# Full 15-incident protocol.
python defilogic_extended\run_gptscan_mistral.py --repetitions 3 `
  --model ministral-8b-latest `
  --run-dir defilogic_extended\results\gptscan_mistral_r3
```

The runner reports IDR automatically. RCVR must be reviewed after inference:
for each positive output, compare the reported causal mechanism and localized
code evidence with the held-out manifest label. A matching rule name alone is
not sufficient. Preserve this review alongside the generated run directory.

## 5. FinanceFuzz adapter and compatibility boundary

FinanceFuzz is evaluated separately because it is a property-testing fuzzer,
not a candidate-level explanation validator. Its released code needs a local
checkout and an older supported EVM/compiler environment. The included patch
only makes upstream commit `103fff2` parse and start under Python 3.10; it does
not add any oracle, vulnerability rule, or benchmark-specific behavior.

```powershell
git clone https://github.com/duytai/FinanceFuzz.git `
  defilogic_extended\third_party\FinanceFuzz
git -C defilogic_extended\third_party\FinanceFuzz checkout 103fff2
git -C defilogic_extended\third_party\FinanceFuzz apply `
  defilogic_extended\patches\financefuzz-python310.patch

# Run from WSL/Linux after installing FinanceFuzz's own dependencies and solc.
python defilogic_extended/run_financefuzz.py `
  --dataset-root defilogic_extended `
  --financefuzz-root defilogic_extended/third_party/FinanceFuzz `
  --output defilogic_extended/results/financefuzz.json --seconds 30 --seed 1
```

The adapter currently accepts a single Solidity source file. Many real
incidents require imported dependencies, constructor arguments, or a newer
EVM revision; therefore its output is **not an IDR or RCVR baseline** until a
per-incident deployment package is supplied. See `BASELINE_STATUS.md` for the
observed partial-run limitations and requirements for a fair comparison.

## Metrics and reporting rules

- **IDR (incident detection recall):** an incident is detected when at least
  one candidate receives the benchmark-defined positive or evidence-needed
  verdict.
- **Strict RCVR (root-cause validation recall):** an incident counts only when
  the reported root cause matches the annotation at the available granularity
  and CONVERA's lightweight validator emits explicit supporting evidence.
- Report the source-ready denominator, provider/model/version, random seed or
  repetition count, timeout, and all exclusions. Do not average RCVR across
  datasets that use different root-cause label granularities.

The current collection is an external pilot. Its results must be reported with
the case count and repetition spread, not as a replacement for controlled
VERITE evaluation.
