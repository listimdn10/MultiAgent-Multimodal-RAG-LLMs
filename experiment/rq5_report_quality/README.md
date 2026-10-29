# RQ5 — Audit-report quality (LLM-as-judge)

RQ5 compares CONVERA and GPTLens reports on the **same PoPoC contracts**. The
judge receives only the contract context, identified vulnerability, and report;
it does **not** receive the method name. It scores Clarity, Completeness, and
Readability from 1 to 10 and explains each score. The prompt in
`run_llm_judge.py` reproduces the judge prompt printed in the manuscript
appendix. A report's overall score is the arithmetic mean of its three scores.

## Reproducibility boundary

The full original PoPoC report pair set and raw judge outputs behind the
manuscript table are **not in this repository**. The saved VERITE, DeFiLogic,
and conventional runs are different datasets. This script can evaluate a
new set, but it does not recreate the manuscript numbers on its own.
The manuscript used Gemini 2.0 Flash-Lite; [Google shut that model down on
June 1, 2026](https://ai.google.dev/gemini-api/docs/deprecations). The runner
therefore defaults to `gemini-2.5-flash-lite`. A new run is a **new judge-model
evaluation**, even if the original reports become available. Record the model
and results separately from the paper table. **Do not replace the paper RQ5
table with these scores without revising its dataset, model, and protocol.**

## Build new inputs from existing outputs (no API charge)

From the repository root:

```powershell
python experiment/run.py build-rq5-inputs
```

This writes ignored manifests and provenance under
`experiment/rq5_report_quality/results/existing_inputs/`:

| Manifest | Contents | Valid comparison |
| --- | --- | --- |
| `defi_convera_gptlens_paired.jsonl` | 15 DeFiLogic cases, CONVERA Reason Agent and GPTLens | Paired, same 15 cases; **not full CONVERA** |
| `defi_all_available.jsonl` | Same 15 cases, up to four methods | Only 2 cases have a report from all four methods; report coverage must accompany any descriptive scores |
| `conventional_consensus_selected.jsonl` | 47 conventional cases with `RAG`/`Explainer` Consensus decision | CONVERA-only descriptive quality; no matched baseline output here |

For the DeFi inputs, CONVERA is the saved **Reason Agent** `Audit_report`, not
the full Bi-modal + Multi-modal + Consensus pipeline. The GPTLens report is
the top entry from its saved ranking; VulnHuntGPT and GPTScan use the first
emitted finding (neither output has a native rank). Cases with no emitted
finding are **missing reports**, not zero-quality reports. This selection can
bias conditional quality upward; always report coverage alongside quality.
The common vulnerability context comes from the DeFiLogic gold root-cause
description, supplied **only to the judge**, not to the original generators.

For the conventional input, Consensus decisions `RAG` and `Explainer` select
that branch's saved `Audit_report` verbatim. `Merged` is **excluded** because
the saved Consensus output contains a summary, not a complete merged audit
report; assigning it either branch would misrepresent the actual decision.
`FAILED` and missing decisions are also excluded. Every inclusion/exclusion,
artifact path, and SHA-256 is listed in `*_provenance.json`. No score from this
CONVERA-only set should be compared directly to the DeFi baseline scores.

Preflight the sets separately:

```powershell
python experiment/run.py judge-reports `
  --input experiment/rq5_report_quality/results/existing_inputs/defi_convera_gptlens_paired.jsonl --dry-run

python experiment/run.py judge-reports `
  --input experiment/rq5_report_quality/results/existing_inputs/defi_all_available.jsonl `
  --methods CONVERA GPTLens VulnHuntGPT GPTScan --allow-missing --dry-run

python experiment/run.py judge-reports `
  --input experiment/rq5_report_quality/results/existing_inputs/conventional_consensus_selected.jsonl `
  --methods CONVERA --dry-run
```

For the new evaluation using the OpenAI key in root `.env`, run the four-method
DeFi set first. The runner defaults to `gpt-6-luna` for OpenAI and records the
actual provider, model, rubric hash, and per-request token usage:

```powershell
python experiment/run.py judge-reports `
  --input experiment/rq5_report_quality/results/existing_inputs/defi_all_available.jsonl `
  --methods CONVERA GPTLens VulnHuntGPT GPTScan --allow-missing `
  --provider openai --model gpt-6-luna `
  --output-dir experiment/rq5_report_quality/results/defi_openai_luna
```

The summary provides matched **pairwise** comparisons (15 CONVERA–GPTLens
cases, and smaller intersections for the other baselines), as well as the
2-case four-way intersection. To judge conventional branch-selected CONVERA
reports descriptively, use a **separate** run:

```powershell
python experiment/run.py judge-reports `
  --input experiment/rq5_report_quality/results/existing_inputs/conventional_consensus_selected.jsonl `
  --methods CONVERA --provider openai --model gpt-6-luna `
  --output-dir experiment/rq5_report_quality/results/conventional_selected_openai_luna
```

Use a separate `--output-dir` for each paid evaluation.
The four-method summary reports both the complete four-way intersection and
matched pairwise subsets; per-method available-report means have different
sample sets and are **descriptive only**. Each input row is scored three times by
default; the DeFi paired, DeFi four-method, and conventional manifests need
90, 120, and 141 API calls respectively. Start with the paired DeFi manifest
if cost matters; do not run all three and treat the overlapping judgments as
independent evidence.

## Input format

For a custom evaluation, create a UTF-8 JSONL file with **two rows per
`case_id`**, one for `CONVERA` and one for `GPTLens`. Both rows must have
identical contract context and identified vulnerability. Each `report` must be
the complete generated audit report, not a summary. For example, this is only a
schema illustration, **not benchmark data**:

```jsonl
{"case_id":"example-01","method":"CONVERA","contract_context":"pragma solidity ...","identified_vulnerability":"Unauthorised withdrawal","report":"Complete CONVERA report ..."}
{"case_id":"example-01","method":"GPTLens","contract_context":"pragma solidity ...","identified_vulnerability":"Unauthorised withdrawal","report":"Complete GPTLens report ..."}
```

Use the same case IDs and context in both methods. Do not include system
identity or post-hoc edits in the material sent to the judge. If gold labels
are supplied as the identified vulnerability, disclose this choice.
Make sure the report text itself does not reveal the method name if strict
blinding is required. The script validates the pair structure before making a
paid API call, but it cannot prove that two different reports came from the
same generation backbone or that the PoPoC selection matches the original.

## Run

From the repository root, use Python 3.10+; the runner uses only the standard
library. Put `GEMINI_API_KEY` or `OPENAI_API_KEY` in the environment or root
`.env` (never commit it). First preflight the manifest without billing:

```powershell
python experiment/rq5_report_quality/run_llm_judge.py `
  --input path/to/paired_popoc_reports.jsonl --dry-run
```

Then run three independent judge requests per report (Gemini example):

```powershell
python experiment/rq5_report_quality/run_llm_judge.py `
  --input path/to/paired_popoc_reports.jsonl `
  --output-dir experiment/rq5_report_quality/results/popoc_gemini25 `
  --repetitions 3 --model gemini-2.5-flash-lite
```

The same command resumes an interrupted run: already successful judgments
are skipped, failed ones are retried. Changing the input bytes, model, rubric,
temperature/reasoning setting, or repetition count requires a **new** output directory. The
runner writes `config.json` (manifest/prompt hashes and judge settings),
`judgments.jsonl` (raw response, parsed scores, justifications, usage, errors),
and `summary.json` (means and population standard deviations across the three
repetition means). Direct comparisons use only cases with all requested
methods, and pairwise comparisons use the corresponding matched subset;
incomplete cases are explicitly counted. Outputs are
ignored by Git because reports can include sensitive contract text.

The script is the automated LLM-judge portion only. The manuscript's separate
human review of recommendation accuracy/actionability is **not automated** by
this runner.

Offline checks (no API calls):

```powershell
python -m unittest discover -s tests -p test_rq5_llm_judge.py
```
