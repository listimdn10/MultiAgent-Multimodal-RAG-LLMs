# RQ3 — Fine-tuned SLMs, with and without Neo4j RAG

RQ3 studies whether retrieval improves a fine-tuned Solidity auditor. This
folder provides a downloader for six **public LoRA adapters** and a script that
evaluates the *same examples and adapter* in no-RAG and with-RAG modes. The
adapters are not complete models: inference also downloads their base-model
weights. The larger settings require a suitable CUDA GPU.

## Models

| Paper model | Hugging Face adapter |
| --- | --- |
| Qwen-2.5-0.5B-FT | [Nhudang/Qwen2.5-0.5B-Solidity](https://huggingface.co/Nhudang/Qwen2.5-0.5B-Solidity) |
| Qwen-3-0.6B-FT | [Nhudang/Qwen3-0.6B-Solidity](https://huggingface.co/Nhudang/Qwen3-0.6B-Solidity) |
| Phi-3.5-mini-4B-FT | [Nhudang/Phi-3.5-mini-instruct-Solidity](https://huggingface.co/Nhudang/Phi-3.5-mini-instruct-Solidity) |
| Qwen-2.5-coder-3B-FT | [Nhudang/Qwen-Coder-2.5-3B-Solidity](https://huggingface.co/Nhudang/Qwen-Coder-2.5-3B-Solidity) |
| DeepSeek-FT-8B-FT | [Nhudang/DeepSeek-R1-Distill-Llama-8B-SOLIDITY-EVAL](https://huggingface.co/Nhudang/DeepSeek-R1-Distill-Llama-8B-SOLIDITY-EVAL) |
| Qwen2.5-Coder-14B-FT | [Nhudang/Qwen-Coder-2.5-14B-Solidity](https://huggingface.co/Nhudang/Qwen-Coder-2.5-14B-Solidity) |

`download_models.py` pins exact Hub revisions, validates each adapter, and
writes an ignored `models/manifest.json` listing the required base model.
The repositories are public; no Hugging Face token is required. Never commit
or paste credentials into a command. Revoke any token shared in an exposed
location before using it again.

## Requirements and download

Run from the repository root with Python 3.10. Install the root
`requirements.txt`, which pins `transformers`, `peft`, `accelerate`,
`bitsandbytes`, `sentence-transformers`, `neo4j`, `python-dotenv`, and
`scikit-learn`. Inference needs a compatible PyTorch/CUDA installation and
enough memory for the chosen base model. A CPU-only environment can download
and inspect the adapters but is not the validated environment for all six.

```powershell
python -m pip install -r requirements.txt
python experiment\rq3_slm_rag\download_models.py --dry-run
python experiment\rq3_slm_rag\download_models.py
python experiment\rq3_slm_rag\download_models.py --verify-only

# Or download one adapter:
python experiment\rq3_slm_rag\download_models.py --model Qwen2.5-0.5B-Solidity
```

Snapshots go to ignored `experiment/rq3_slm_rag/models/`. The downloader
can be rerun after a network interruption; it never stores a token.

## Dataset and RAG setup

The **original 2,472-item Audited Report Dataset and its exact held-out
split are not in this repository**. The script accepts that split as JSONL:
one record per line with non-empty string fields `id`, `code`, and `label`.
For an immediately downloadable exploratory test, it instead fetches pinned
[SmartBugs Curated](https://github.com/smartbugs/smartbugs-curated) Solidity
contracts. Only samples with one unique vulnerability category are evaluated;
multi-label contracts are excluded. This substitute is **not** the RQ3 test
split, and its metrics must not be presented as a reproduction of Table
`slm_llm_combined`. It may overlap with SWC-derived graph knowledge, so it is
a functionality check, not a leakage-controlled benchmark.

For with-RAG, set `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`,
`NEO4J_DATABASE`, and `NEO4J_VECTOR_INDEX=behavior_index` in the ignored
root `.env`. The index must contain `Behavior` nodes embedded with
`sentence-transformers/all-MiniLM-L6-v2`, and the graph must have
`Behavior-[:EXPOSES]->Vulnerability-[:FIXED_BY]->FCode-[:MITIGATED_BY]->Solution`
edges. Graph preparation is documented in the
[RQ1 README](../rq1_detection/README.md). The RQ3 script only reads the
existing graph. It embeds source code directly for retrieval, whereas the
original Bi-modal flow generated a functional-semantic summary first. Thus
this is a controlled paired test of current Neo4j context, not an exact
reconstruction of the historical RAG prompt.

If your Neo4j settings are in a separate ignored credential file, pass its
path with `--neo4j-env-file` rather than copying its values into a command.
That explicit file overrides any stale Neo4j entries in the root `.env`.

## Paired evaluation

```powershell
# Download/check the public dataset and a downloaded adapter; no inference.
python experiment\rq3_slm_rag\eval_rq3.py --model Qwen2.5-0.5B-Solidity --prepare-only

# Optional: check the existing Neo4j Behavior index without running an LLM.
python experiment\rq3_slm_rag\eval_rq3.py --model Qwen2.5-0.5B-Solidity --prepare-only --check-rag --neo4j-env-file path\to\neo4j_credentials.txt

# GPU machine: both modes on the same 10 contracts, sampled across categories.
python experiment\rq3_slm_rag\eval_rq3.py --model Qwen2.5-0.5B-Solidity --mode both --max-samples 10

# Use the paper's original held-out split when it becomes available.
python experiment\rq3_slm_rag\eval_rq3.py --model Qwen2.5-0.5B-Solidity --mode both --dataset-jsonl path\to\held_out.jsonl
```

Repeat `--model` for the other adapter names. `--mode no-rag` needs no
Neo4j connection; `--mode with-rag` runs only retrieval-enhanced inference.
Gold labels are kept out of prompts and retrieval queries. The script uses
deterministic decoding, requires an exact JSON category response, and records
invalid outputs without guessing. Results go to ignored
`experiment/rq3_slm_rag/results/<UTC timestamp>_<model>/` as per-mode JSONL
and `summary.json` with accuracy and macro precision, recall, and F1. New
scores are not automatically inserted into the manuscript. Retraining remains
outside this runner.
