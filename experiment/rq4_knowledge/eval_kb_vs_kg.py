"""So sánh Knowledge Base (flat) và Knowledge Graph (Neo4j) bằng Accuracy / Precision / Recall / F1.

Hai cấu hình dùng cùng nguồn tri thức (knowledge-graph-relationship.csv của RQ1,
đã nạp vào Neo4j bởi experiment/rq1_detection/build_neo4j_to_vector_indexes.py):

* KB (flat): mỗi dòng CSV là một tài liệu độc lập; truy xuất cosine top-k trên
  embedding của trường văn bản (vulnerable code / Detail Behaviors), không dùng quan hệ.
* KG (Neo4j): truy xuất vector index (vcode_index, behavior_index, ...) rồi đi theo
  quan hệ (File)-[:HAS_VCODE]->(VCode)-[:HAS_PURPOSE]->(Purpose)-[:HAS_BEHAVIORS]->(Behavior)
  -[:EXPOSES]->(Vulnerability) ngược về nút File để lấy nhãn, gộp điểm của mọi nút seed.

Dự đoán nhãn = nhãn (tên file không đuôi, ví dụ SWC-100, hoặc ánh xạ qua --label-map)
của File có tổng điểm tương đồng cao nhất. Không gọi LLM nên kết quả tất định.

Dữ liệu đánh giá (--dataset): .jsonl / .csv / .xlsx với cột id, code, label.
Cảnh báo rò rỉ: nếu mẫu test cũng nằm trong knowledge base, dùng --exclude-self
(bỏ File có tên trùng id của mẫu) để tránh trả lại đúng đáp án.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_KB_CSV = REPO / "experiment" / "rq1_detection" / "knowledge-graph-relationship.csv"
DEFAULT_RESULTS = HERE / "results"
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # trùng encoder của rq3 / indexes

# index Neo4j -> (nhãn nút, cột CSV tương ứng cho cấu hình KB flat)
FIELDS = {
    "vcode_index": ("VCode", "vulnerable code"),
    "behavior_index": ("Behavior", "Detail Behaviors"),
    "vulnerability_index": ("Vulnerability", "Vulnerability Description"),
    "purpose_index": ("Purpose", "Abstract purpose"),
}

# Từ nút seed đi ngược về File (độ sâu tối đa 4 cạnh).
KG_QUERY = """
CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
YIELD node, score
MATCH (f:File)-[:HAS_VCODE|HAS_PURPOSE|HAS_BEHAVIORS|EXPOSES*1..4]-(node)
RETURN f.name AS file, max(score) AS score
"""


# ---------------------------------------------------------------- dữ liệu
def load_samples(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".csv":
        csv.field_size_limit(10**9)
        with path.open(encoding="utf-8-sig", newline="") as src:
            rows = list(csv.DictReader(src))
    elif suffix in (".xlsx", ".xls"):
        import pandas as pd
        rows = pd.read_excel(path).to_dict("records")
    else:
        raise ValueError(f"Định dạng dataset không hỗ trợ: {suffix}")
    samples = []
    for i, row in enumerate(rows):
        code, label = row.get("code"), row.get("label")
        if not isinstance(code, str) or not code.strip() or label is None or str(label).strip() == "":
            continue
        samples.append({"id": str(row.get("id", i)), "code": code, "label": str(label).strip()})
    if not samples:
        raise ValueError(f"Không có mẫu hợp lệ (cần cột code, label) trong {path}")
    return samples


def load_kb(path: Path) -> list[dict]:
    csv.field_size_limit(10**9)
    with path.open(encoding="utf-8-sig", newline="") as src:
        return list(csv.DictReader(src))


def file_stem(name: str) -> str:
    return Path(name).stem


# ---------------------------------------------------------------- bộ dự đoán
def vote(hits: list[tuple[str, float]], label_map: dict[str, str]) -> str:
    """Cộng điểm tương đồng theo từng File; trả nhãn của File điểm cao nhất."""
    totals: dict[str, float] = defaultdict(float)
    for file, score in hits:
        totals[file] += score
    if not totals:
        return "__none__"
    best = max(totals, key=totals.get)
    stem = file_stem(best)
    return label_map.get(best, label_map.get(stem, stem))


class FlatKB:
    """Knowledge base phẳng: cosine top-k trên từng trường, không dùng quan hệ."""

    def __init__(self, rows, fields, encoder, top_k):
        import numpy as np
        self.np, self.top_k = np, top_k
        self.files = [r["filename"] for r in rows]
        self.matrices = {}
        for index in fields:
            column = FIELDS[index][1]
            texts = [(r.get(column) or "")[:4000] for r in rows]
            self.matrices[index] = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def hits(self, query_vec, skip=None):
        out = []
        for matrix in self.matrices.values():
            sims = matrix @ query_vec
            kept = 0
            for i in self.np.argsort(-sims):
                if skip and file_stem(self.files[i]) == skip:
                    continue
                out.append((self.files[i], float(sims[i])))
                kept += 1
                if kept == self.top_k:
                    break
        return out


class GraphKB:
    """Knowledge graph trên Neo4j: vector index + duyệt quan hệ về nút File."""

    def __init__(self, indexes, top_k):
        from neo4j import GraphDatabase
        missing = [k for k in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD") if not os.environ.get(k)]
        if missing:
            raise RuntimeError("Thiếu biến môi trường Neo4j: " + ", ".join(missing))
        self.database = os.environ.get("NEO4J_DATABASE", "neo4j")
        self.indexes, self.top_k = indexes, top_k
        self.driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))
        self.driver.verify_connectivity()

    def hits(self, query_vec, skip=None):
        out = []
        embedding = [float(x) for x in query_vec]
        with self.driver.session(database=self.database) as session:
            for index in self.indexes:
                # lấy dư 1 seed phòng khi phải bỏ File của chính mẫu
                rows = session.run(KG_QUERY, index_name=index,
                                   top_k=self.top_k + (1 if skip else 0), embedding=embedding)
                for rec in rows:
                    if skip and file_stem(rec["file"]) == skip:
                        continue
                    out.append((rec["file"], float(rec["score"])))
        return out

    def close(self):
        self.driver.close()


# ---------------------------------------------------------------- chỉ số
def metrics(gold: list[str], pred: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    labels = sorted(set(gold))
    out = {"n": len(gold), "accuracy": float(accuracy_score(gold, pred))}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(gold, pred, labels=labels, average=avg, zero_division=0)
        out[f"{avg}_precision"], out[f"{avg}_recall"], out[f"{avg}_f1"] = float(p), float(r), float(f)
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=labels, average=None, zero_division=0)
    out["per_class"] = {
        lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i, lab in enumerate(labels)
    }
    return out


def print_table(results: dict) -> None:
    cols = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")
    print(f"\n{'Config':<8}" + "".join(f"{c:>18}" for c in cols))
    for name, m in results.items():
        print(f"{name:<8}" + "".join(f"{m[c] * 100:>17.2f}%" for c in cols))


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True, help="jsonl/csv/xlsx với cột id, code, label")
    ap.add_argument("--kb-csv", type=Path, default=DEFAULT_KB_CSV, help="CSV nguồn của knowledge base (flat)")
    ap.add_argument("--label-map", type=Path, help="JSON ánh xạ tên file/stem trong KB -> nhãn của dataset")
    ap.add_argument("--indexes", default="vcode_index,behavior_index",
                    help="Index dùng cho cả KB và KG, phân tách bởi dấu phẩy: " + ", ".join(FIELDS))
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--exclude-self", action="store_true", help="Bỏ File có stem trùng id mẫu (tránh rò rỉ)")
    ap.add_argument("--only", choices=("kb", "kg", "both"), default="both")
    ap.add_argument("--neo4j-env-file", type=Path, help="File .env chứa NEO4J_* (không commit)")
    ap.add_argument("--limit", type=int, help="Chỉ đánh giá N mẫu đầu (chạy thử)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS)
    args = ap.parse_args()

    indexes = [i.strip() for i in args.indexes.split(",") if i.strip()]
    unknown = [i for i in indexes if i not in FIELDS]
    if unknown:
        ap.error(f"Index không hỗ trợ: {unknown}")

    if args.neo4j_env_file:
        from dotenv import load_dotenv
        if not args.neo4j_env_file.is_file():
            raise FileNotFoundError(args.neo4j_env_file)
        load_dotenv(args.neo4j_env_file, override=True)

    samples = load_samples(args.dataset)[: args.limit]
    label_map = json.loads(args.label_map.read_text(encoding="utf-8")) if args.label_map else {}

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(ENCODER_NAME)
    queries = encoder.encode([s["code"][:4000] for s in samples], normalize_embeddings=True, show_progress_bar=True)

    systems = {}
    if args.only in ("kb", "both"):
        systems["KB"] = FlatKB(load_kb(args.kb_csv), indexes, encoder, args.top_k)
    if args.only in ("kg", "both"):
        systems["KG"] = GraphKB(indexes, args.top_k)

    gold = [s["label"] for s in samples]
    preds = {name: [] for name in systems}
    for sample, vec in zip(samples, queries):
        skip = file_stem(sample["id"]) if args.exclude_self else None
        for name, system in systems.items():
            preds[name].append(vote(system.hits(vec, skip), label_map))

    if "KG" in systems:
        systems["KG"].close()

    results = {name: metrics(gold, p) for name, p in preds.items()}
    print_table(results)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "created_utc": stamp, "dataset": str(args.dataset), "indexes": indexes, "top_k": args.top_k,
        "exclude_self": args.exclude_self, "encoder": ENCODER_NAME, "results": results,
    }
    (args.out_dir / f"kb_vs_kg_{stamp}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.out_dir / f"kb_vs_kg_{stamp}_predictions.csv").open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["id", "gold", *[f"pred_{n}" for n in systems]])
        for i, sample in enumerate(samples):
            writer.writerow([sample["id"], sample["label"], *[preds[n][i] for n in systems]])
    print(f"\nĐã ghi kết quả vào {args.out_dir}")


if __name__ == "__main__":
    main()
