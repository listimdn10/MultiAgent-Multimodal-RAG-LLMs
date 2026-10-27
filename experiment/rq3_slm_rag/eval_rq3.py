"""Evaluate one fine-tuned adapter with and without Neo4j RAG.

By default this downloads a pinned SmartBugs Curated *exploratory* dataset.
Pass --dataset-jsonl with the original held-out RQ3 split to reproduce the
paper protocol; the repository does not currently include that split.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

try:
    from .download_models import HERE, MODELS
except ImportError:  # direct execution: python experiment/rq3_slm_rag/eval_rq3.py
    from download_models import HERE, MODELS


SMARTBUGS_REPO = "https://github.com/smartbugs/smartbugs-curated.git"
SMARTBUGS_REVISION = "230e649123477eff332742a59a1c7cc6dc286cab"
DEFAULT_DATA = HERE / "data" / "smartbugs-curated"
DEFAULT_RESULTS = HERE / "results"


def ensure_smartbugs(path: Path) -> str:
    manifest = path / "vulnerabilities.json"
    if not manifest.is_file():
        if path.exists():
            raise RuntimeError(f"Dataset directory exists but is incomplete: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--branch", "main",
                        SMARTBUGS_REPO, str(path)], check=True)
    current = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True).stdout.strip()
    if current != SMARTBUGS_REVISION:
        subprocess.run(["git", "-C", str(path), "fetch", "--depth", "1",
                        "origin", SMARTBUGS_REVISION], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "--detach",
                        SMARTBUGS_REVISION], check=True)
    if not manifest.is_file():
        raise RuntimeError(f"Pinned dataset has no vulnerabilities.json: {path}")
    return SMARTBUGS_REVISION


def load_smartbugs(path: Path) -> list[dict[str, str]]:
    entries = json.loads((path / "vulnerabilities.json").read_text(encoding="utf-8"))
    samples = []
    for entry in entries:
        labels = {v.get("category", "") for v in entry.get("vulnerabilities", [])}
        labels.discard("")
        if len(labels) != 1:  # one-label classification; do not invent labels for multi-label cases
            continue
        source = path / entry["path"]
        if not source.is_file():
            raise FileNotFoundError(f"Dataset manifest references missing source: {source}")
        samples.append({
            "id": entry["path"],
            "code": source.read_text(encoding="utf-8", errors="replace"),
            "label": next(iter(labels)),
        })
    return sorted(samples, key=lambda sample: sample["id"])


def load_jsonl(path: Path) -> list[dict[str, str]]:
    samples = []
    with path.open("r", encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            if not all(isinstance(sample.get(key), str) and sample[key].strip()
                       for key in ("id", "code", "label")):
                raise ValueError(f"{path}:{line_no} needs non-empty string id/code/label")
            samples.append({key: sample[key] for key in ("id", "code", "label")})
    if len({item["id"] for item in samples}) != len(samples):
        raise ValueError("Dataset has duplicate ids")
    return samples


def select_balanced(samples: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    """Deterministic round-robin so a small smoke test spans categories."""
    if limit == 0 or limit >= len(samples):
        return samples
    buckets: dict[str, list[dict[str, str]]] = {}
    for sample in samples:
        buckets.setdefault(sample["label"], []).append(sample)
    chosen = []
    while len(chosen) < limit:
        for label in sorted(buckets):
            if buckets[label]:
                chosen.append(buckets[label].pop(0))
                if len(chosen) == limit:
                    break
    return chosen


def parse_label(output: str, labels: list[str]) -> str:
    """Accept only an exact allowed class, never a loose keyword match."""
    cleaned = output.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.S | re.I)
    candidate = fenced.group(1) if fenced else cleaned
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return "__invalid__"
    value = parsed.get("category") if isinstance(parsed, dict) else None
    return value if value in labels else "__invalid__"


def make_prompt(code: str, labels: list[str], context: str) -> str:
    label_list = json.dumps(labels, ensure_ascii=False)
    return (
        "Classify the Solidity contract into exactly one vulnerability category. "
        "Use the source code as the primary evidence. Retrieved patterns are "
        "background examples, not ground-truth labels for this contract. "
        f"Allowed categories: {label_list}. "
        'Return only JSON of the form {"category":"one allowed category"}.\n\n'
        f"Retrieved vulnerability patterns:\n{context or '(none)'}\n\n"
        f"Solidity source:\n{code}\n"
    )


class Neo4jRetriever:
    """Read the existing Behavior vector index and follow graph security edges."""

    QUERY = """
    CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
    YIELD node, score
    WHERE node:Behavior
    OPTIONAL MATCH (node)-[:EXPOSES]->(v:Vulnerability)
    OPTIONAL MATCH (v)-[:FIXED_BY]->(:FCode)-[:MITIGATED_BY]->(s:Solution)
    RETURN node.text AS behavior, v.text AS vulnerability,
           collect(DISTINCT s.text) AS solutions, score
    ORDER BY score DESC
    """

    def __init__(self, index_name: str):
        from neo4j import GraphDatabase

        required = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")
        missing = [key for key in required if not os.environ.get(key)]
        if missing:
            raise RuntimeError("Missing Neo4j environment variables: " + ", ".join(missing))
        self.database = os.environ.get("NEO4J_DATABASE", "neo4j")
        self.index_name = index_name
        self.driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        )
        try:
            self.driver.verify_connectivity()
        except Exception as exc:
            self.driver.close()
            raise RuntimeError("Neo4j is unreachable; check URI/DNS and the credential file") from exc
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def retrieve(self, code: str, top_k: int) -> tuple[str, list[dict]]:
        embedding = self.encoder.encode(code[:4000]).tolist()
        with self.driver.session(database=self.database) as session:
            rows = [dict(record) for record in session.run(
                self.QUERY, index_name=self.index_name, top_k=top_k,
                embedding=embedding,
            )]
        if not rows:
            raise RuntimeError(f"Neo4j vector index {self.index_name!r} returned no Behavior nodes")
        parts = []
        for row in rows:
            parts.append(
                "Behavior: " + str(row.get("behavior") or "") + "\n" +
                "Vulnerability: " + str(row.get("vulnerability") or "") + "\n" +
                "Mitigation: " + "; ".join(x for x in row.get("solutions", []) if x)
            )
        return "\n---\n".join(parts), rows

    def close(self) -> None:
        self.driver.close()


class AdapterGenerator:
    def __init__(self, adapter_dir: Path, max_input_tokens: int):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "These adapters reference 4-bit base models; run inference on a CUDA GPU. "
                "Adapter download and dataset preparation work without CUDA."
            )
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
        if not (adapter_dir / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Download the adapter first: {adapter_dir}")
        self.torch = torch
        self.max_input_tokens = max_input_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=False)
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            adapter_dir, device_map="auto", torch_dtype="auto", trust_remote_code=False,
        )
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": prompt}]
        if self.tokenizer.chat_template:
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt
        encoded = self.tokenizer(
            formatted, return_tensors="pt", truncation=True,
            max_length=self.max_input_tokens,
        )
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        continuation = generated[0, encoded["input_ids"].shape[1]:]
        return self.tokenizer.decode(continuation, skip_special_tokens=True).strip()


def score(rows: list[dict], labels: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    gold = [row["gold"] for row in rows]
    predicted = [row["predicted"] for row in rows]
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, predicted, labels=labels, average="macro", zero_division=0
    )
    return {
        "n": len(rows), "accuracy": float(accuracy_score(gold, predicted)),
        "macro_precision": float(precision), "macro_recall": float(recall),
        "macro_f1": float(f1),
        "invalid_outputs": sum(value == "__invalid__" for value in predicted),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--mode", choices=("both", "no-rag", "with-rag"), default="both")
    parser.add_argument("--dataset-jsonl", type=Path,
                        help="Original held-out split: JSONL with id, code, label")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-dir", type=Path, default=HERE / "models")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples")
    parser.add_argument("--max-source-chars", type=int, default=8000)
    parser.add_argument("--max-input-tokens", type=int, default=3072)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--neo4j-index", default=None,
                        help="Behavior vector index; defaults to NEO4J_VECTOR_INDEX or behavior_index")
    parser.add_argument("--neo4j-env-file", type=Path,
                        help="Optional local Neo4j credential file (never commit it)")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Download/validate dataset and adapter without inference")
    parser.add_argument("--check-rag", action="store_true",
                        help="With --prepare-only, run one read-only Neo4j retrieval check")
    args = parser.parse_args()
    if args.check_rag and not args.prepare_only:
        parser.error("--check-rag requires --prepare-only")
    if args.max_samples < 0 or args.top_k < 1 or args.max_source_chars < 1:
        parser.error("--max-samples must be nonnegative; --top-k and --max-source-chars positive")

    from dotenv import load_dotenv
    load_dotenv(HERE.parents[1] / ".env", override=False)
    if args.neo4j_env_file:
        if not args.neo4j_env_file.is_file():
            raise FileNotFoundError(f"Neo4j credential file not found: {args.neo4j_env_file}")
        # An explicitly selected credential file takes precedence over a stale
        # root .env; otherwise a rerun can silently query the wrong instance.
        load_dotenv(args.neo4j_env_file, override=True)

    if args.dataset_jsonl:
        samples = load_jsonl(args.dataset_jsonl)
        dataset = {"type": "provided_jsonl", "path": str(args.dataset_jsonl.resolve())}
    else:
        revision = ensure_smartbugs(args.dataset_dir)
        samples = load_smartbugs(args.dataset_dir)
        dataset = {"type": "smartbugs_curated_exploratory", "revision": revision}
    if not samples:
        raise RuntimeError("No evaluable samples found")
    labels = sorted({sample["label"] for sample in samples})
    if len(labels) < 2:
        raise RuntimeError("At least two gold categories are required for classification")
    if args.max_samples:
        samples = select_balanced(samples, args.max_samples)
    adapter_dir = args.model_dir / args.model
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"First run download_models.py --model {args.model}")
    print(f"Dataset: {dataset['type']}; cases={len(samples)}; labels={labels}")
    print(f"Adapter: {adapter_dir} @ {MODELS[args.model]}")
    if args.prepare_only:
        if args.check_rag:
            index = args.neo4j_index or os.environ.get("NEO4J_VECTOR_INDEX") or "behavior_index"
            checker = Neo4jRetriever(index)
            try:
                _, hits = checker.retrieve(samples[0]["code"][:args.max_source_chars], args.top_k)
                print(f"Read-only Neo4j retrieval OK: {len(hits)} hit(s) from {index}")
            finally:
                checker.close()
        print("Prepared only; no model inference was performed.")
        return

    modes = ("no-rag", "with-rag") if args.mode == "both" else (args.mode,)
    retriever = None
    if "with-rag" in modes:
        index = args.neo4j_index or os.environ.get("NEO4J_VECTOR_INDEX") or "behavior_index"
        retriever = Neo4jRetriever(index)
    generator = AdapterGenerator(adapter_dir, args.max_input_tokens)
    run_dir = args.output_dir / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + args.model)
    run_dir.mkdir(parents=True, exist_ok=False)
    results = {mode: [] for mode in modes}
    try:
        for i, sample in enumerate(samples, 1):
            code = sample["code"][:args.max_source_chars]
            context, hits = ("", [])
            if retriever:
                context, hits = retriever.retrieve(code, args.top_k)
            for mode in modes:
                prompt = make_prompt(code, labels, context if mode == "with-rag" else "")
                raw = generator.generate(prompt, args.max_new_tokens)
                row = {
                    "id": sample["id"], "gold": sample["label"],
                    "predicted": parse_label(raw, labels), "raw_output": raw,
                    "retrieved": hits if mode == "with-rag" else [],
                }
                results[mode].append(row)
                with (run_dir / f"{mode}.jsonl").open("a", encoding="utf-8") as sink:
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{i}/{len(samples)} {sample['id']}", flush=True)
    finally:
        if retriever:
            retriever.close()
    summary = {
        "model": f"Nhudang/{args.model}", "adapter_revision": MODELS[args.model],
        "dataset": dataset, "labels": labels, "mode": args.mode,
        "top_k": args.top_k, "max_source_chars": args.max_source_chars,
        "max_input_tokens": args.max_input_tokens,
        "scores": {mode: score(rows, labels) for mode, rows in results.items()},
        "note": "Exploratory SmartBugs results are not the manuscript RQ3 held-out split."
                if not args.dataset_jsonl else "User-provided held-out split; verify provenance before comparing with manuscript.",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
