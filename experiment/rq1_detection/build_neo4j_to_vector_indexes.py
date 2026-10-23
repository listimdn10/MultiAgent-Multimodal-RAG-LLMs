"""Reproduce notebook cells 5–8 on the RQ1 CSV, using the current Aura credential file.

This performs only MERGE-based graph insertion and creation/population of the
seven vector indexes in notebook cell 8. It never executes later notebook cells.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from dotenv import dotenv_values


HERE = Path(__file__).resolve().parent
CSV_FILE = HERE / "knowledge-graph-relationship.csv"
CREDENTIALS = HERE / "Neo4j-8d51a59e-Created-2026-10-22.txt"
STATUS_FILE = HERE.parents[1] / "rw_dataset" / "results" / "neo4j_build_rq1" / "status.json"
NODE_CONFIGS = (
    ("File", "name"),
    ("VCode", "content"),
    ("FCode", "content"),
    ("Purpose", "text"),
    ("Behavior", "text"),
    ("Vulnerability", "text"),
    ("Solution", "text"),
)
INSERT_QUERY = """
MERGE (f:File {name: $filename})
MERGE (vc:VCode {content: $vulnerable_code, type: 'vulnerable'})
MERGE (p:Purpose {text: $abstract_purpose})
MERGE (b:Behavior {text: $detail_behaviors})
MERGE (v:Vulnerability {text: $vuln_description})
MERGE (fc:FCode {content: $fixed_code, type: 'fixed'})
MERGE (s:Solution {text: $solution_description})
MERGE (f)-[:HAS_VCODE]->(vc)
MERGE (vc)-[:HAS_PURPOSE]->(p)
MERGE (p)-[:HAS_BEHAVIORS]->(b)
MERGE (b)-[:EXPOSES]->(v)
MERGE (v)-[:FIXED_BY]->(fc)
MERGE (fc)-[:MITIGATED_BY]->(s)
MERGE (v)-[:HAS_SOLUTION]->(s)
"""


def config() -> dict[str, str]:
    if not CREDENTIALS.is_file():
        raise FileNotFoundError(f"Missing credential file: {CREDENTIALS.name}")
    data = dotenv_values(CREDENTIALS)
    needed = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE")
    missing = [name for name in needed if not data.get(name)]
    if missing:
        raise ValueError("Missing Neo4j fields: " + ", ".join(missing))
    return {name: str(data[name]) for name in needed}


def load_rows() -> list[dict[str, str]]:
    with CSV_FILE.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    expected = {
        "filename", "vulnerable code", "Abstract purpose", "Detail Behaviors",
        "fixed code", "Vulnerability Description", "Solution Description",
    }
    if not rows or not expected.issubset(rows[0]):
        raise ValueError("CSV is empty or does not have the notebook's seven columns")
    return [{key: value or "" for key, value in row.items()} for row in rows]


def save_status(status: dict) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")


def insert_row(tx, row: dict[str, str]) -> None:
    tx.run(
        INSERT_QUERY,
        filename=row["filename"],
        vulnerable_code=row["vulnerable code"],
        abstract_purpose=row["Abstract purpose"],
        detail_behaviors=row["Detail Behaviors"],
        vuln_description=row["Vulnerability Description"],
        fixed_code=row["fixed code"],
        solution_description=row["Solution Description"],
    ).consume()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Perform the MERGE writes and build all seven indexes")
    args = parser.parse_args()
    rows, credentials = load_rows(), config()
    print(f"CSV rows: {len(rows)}; vector indexes: {len(NODE_CONFIGS)}")
    if not args.execute:
        print("Dry run complete; no Neo4j writes or API calls made. Pass --execute to build the graph.")
        return

    from neo4j import GraphDatabase
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_neo4j import Neo4jVector

    status = {"csv_rows": len(rows), "inserted_rows": 0, "vector_indexes_completed": []}
    driver = GraphDatabase.driver(
        credentials["NEO4J_URI"],
        auth=(credentials["NEO4J_USERNAME"], credentials["NEO4J_PASSWORD"]),
    )
    try:
        driver.verify_connectivity()
        print("Neo4j connection verified.")
        with driver.session(database=credentials["NEO4J_DATABASE"]) as session:
            for position, row in enumerate(rows, start=1):
                session.execute_write(insert_row, row)
                status["inserted_rows"] = position
                if position % 10 == 0 or position == len(rows):
                    save_status(status)
                    print(f"MERGE rows: {position}/{len(rows)}", flush=True)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        for label, property_name in NODE_CONFIGS:
            index_name = f"{label.lower()}_index"
            print(f"Building {index_name} ({label}.{property_name})...", flush=True)
            Neo4jVector.from_existing_graph(
                embedding=embeddings,
                url=credentials["NEO4J_URI"],
                username=credentials["NEO4J_USERNAME"],
                password=credentials["NEO4J_PASSWORD"],
                database=credentials["NEO4J_DATABASE"],
                index_name=index_name,
                node_label=label,
                text_node_properties=[property_name],
                embedding_node_property="embedding",
            )
            status["vector_indexes_completed"].append(index_name)
            save_status(status)
        print("Notebook cells 5–8 completed.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
