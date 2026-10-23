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
| Conventional full CONVERA graph | `experiment/rq1_detection/run_conventional_full_pipeline.py` | `python experiment\rq1_detection\run_conventional_full_pipeline.py --preflight-only` |

## Conventional Excel rerun (all six nodes)

The workbook `convera_benchmark_with_results_new.xlsx` contains 173 rows:
156 labeled vulnerabilities and 17 labeled `safe`. The runner uses only its
`code` column as graph input; `label` and the historical GPTLens output columns
are held back for post-run evaluation. It runs `semantic → rag → embedding →
fusion → reason → consensus` for every selected row. By default it removes
explicit benchmark-label comments such as `@vulnerable_at_lines` and
`// <yes> <report> ...` before inference; `--raw-code` disables this, but can
leak gold labels into prompts.

Run preflight before starting the costly full evaluation:

```powershell
.\.venv\rq1\Scripts\python.exe experiment\rq1_detection\preflight_compilation.py --check-cfg
.\.venv\rq1\Scripts\python.exe experiment\rq1_detection\run_conventional_full_pipeline.py --rag-backend mistral --compiled-only --preflight-only
.\.venv\rq1\Scripts\python.exe experiment\rq1_detection\run_conventional_full_pipeline.py --rag-backend mistral --compiled-only --limit 1
.\.venv\rq1\Scripts\python.exe experiment\rq1_detection\run_conventional_full_pipeline.py --rag-backend mistral --compiled-only --continue-on-error
```

The original flow uses Gemini for functional semantics and Unsloth/Phi-3.5
for RAG generation. This RQ1 runner explicitly substitutes Ministral for
Gemini and still requires Neo4j retrieval, embedding/classifier dependencies,
Java/EtherSolve, Solidity compiler tools, and a configured chat model. Each
case executes in an isolated temporary directory;
source code and pre-existing root-level JSON artifacts are not overwritten.
Outputs and failure traces are written under
`rw_dataset/results/conventional_full_pipeline/` and the runner stops on its
first failed case. `progress.json` records the actually completed cases; a
failed or unrun case must not be counted as a non-detection. The output records
raw stage predictions rather than automatically claiming Accuracy/F1, because
the fusion label space has no `safe` class and the consensus output selects a
branch rather than directly encoding the workbook's five-class gold label.
For the hardware-limited Ministral variant, pass `--rag-backend mistral`;
`--continue-on-error` keeps isolated failures in the denominator audit trail
and stops after three consecutive failures. Resume the *same* set of rows and
settings with `--resume-run <existing-run-dir>`; successful rows are skipped,
while a failed row gets a separate `retry_XX` directory. Each run keeps one
ChromaDB copy under its output directory so Windows can release temporary
per-case workspaces. The RQ1 runner requires a real Solidity CFG: it uses
native Windows `solc` binaries managed by `py-solc-x`, EtherSolve, and a
64-dimensional GATv2 output matching the fusion training workbook. It fails
the case if CFG extraction fails; it does not count a synthetic fallback as a
full-pipeline result. A successful six-stage execution is not necessarily a
correct prediction: inspect `consensus_output.json` against the held-out label.

The offline compilation preflight currently accepts 150/173 workbook rows.
Of the 23 rejected rows, 21 have code cells exactly 32,767 characters long
and are truncated at the Excel cell limit; two others contain Solidity source
errors. `--compiled-only` excludes these rows before any API call, saves the
preflight report alongside the run, and reports them separately rather than
counting them as missed detections. The usable subset therefore has n=150,
not n=173. Do not compare its aggregate score directly to a 173-case result.
The preflight also marks exact source overlap with the Fusion training workbook:
30 of the 150 compilable contracts overlap and 120 do not. Report these strata
separately when judging generalization; exact overlap is a lower bound on
possible near-duplicate leakage.

After the run finishes, export a case-level review sheet with:

```powershell
.\.venv\rq1\Scripts\python.exe experiment\rq1_detection\export_conventional_review.py rw_dataset\results\conventional_full_pipeline\run_20261023_084850 --require-complete
```

The script does not auto-score free-text Consensus summaries. Fill its blank
manual prediction and correctness fields after checking the source and the
held-out family label; do not infer a correct verdict from a matching keyword
in one intermediate agent output.

For the October 2026 RQ1 rerun, the runner uses Ministral for the semantic
stage (`SEMANTIC_PROVIDER=mistral`) and loads the current Neo4j credentials
from the local `Neo4j-*.txt` file. These credentials and the original notebook
are gitignored because they contain plaintext secrets. The graph can be
reconstructed through exactly notebook cells 5–8 with:

```powershell
python experiment\rq1_detection\build_neo4j_to_vector_indexes.py
python experiment\rq1_detection\build_neo4j_to_vector_indexes.py --execute
```

The first command validates the 69-row CSV without writing to Neo4j; the
second MERGEs the seven node classes and their relationships, then builds
`file_index`, `vcode_index`, `fcode_index`, `purpose_index`, `behavior_index`,
`vulnerability_index`, and `solution_index`. The benchmark runner uses
`behavior_index` for retrieval. Its original RAG generator is the local
Unsloth/Phi-3.5 model. On machines where that model is unavailable,
`--rag-backend mistral` explicitly selects a *different backbone* while
retaining all six graph stages; results from that variant must not be labeled
as the original CONVERA configuration or silently compared with the paper's
reported scores.

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
