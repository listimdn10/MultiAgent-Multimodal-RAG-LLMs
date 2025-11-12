# -*- coding: utf-8 -*-
"""
embedding_agent.py
------------------
Purpose:
    - Load rag_output.json (kết quả của RAGRetrieveTool)
    - Thêm node tri thức vào CFG
    - Sinh embeddings: CFG, Code, Functional Semantic
    - Ghi parser_output.json
"""

import os
import re
import json
import torch
import numpy as np
from typing import Type, Optional
from pydantic import BaseModel, Field
from crewai import Agent, Task
from crewai.tools import BaseTool
from torch.nn import Embedding, Linear
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from transformers import RobertaTokenizer, RobertaModel
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings

# ============================================================
# === Model Initialization
# ============================================================

print("🧠 Loading models (GATv2, MiniLM, CodeBERT)...")

TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CODE_MODEL_NAME = "microsoft/codebert-base"

text_encoder = SentenceTransformer(TEXT_MODEL_NAME)
fs_embedder = HuggingFaceEmbeddings(model_name=TEXT_MODEL_NAME)
code_tokenizer = RobertaTokenizer.from_pretrained(CODE_MODEL_NAME)
code_model = RobertaModel.from_pretrained(CODE_MODEL_NAME)

# ============================================================
# === Import RAG Context
# ============================================================

from rag_agent_defined import rag_agent, rag_task, rag_tool
from rag_agent_setup import (
    gemini_model,
    unsloth_model,
    unsloth_tok,
    llm_local,
    driver_rag_agent,
    vcode_vector_store,
    embeddings,
    log
)

# ============================================================
# === Utility Functions
# ============================================================

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_version(code: str) -> str:
    match = re.search(r"pragma\s+solidity\s+\^?([\d.]+)", code)
    return match.group(1) if match else "0.8.0"

def extract_contract_name(code: str) -> str:
    match = re.search(r'\bcontract\s+([A-Za-z_]\w*)\b', code)
    if not match:
        raise ValueError("❌ Không tìm thấy contract name trong code Solidity.")
    return match.group(1)

# ============================================================
# === CFG Embedding
# ============================================================

def extract_cfg_embedding(path: str) -> list:
    rag_output = load_json(path)
    code = rag_output.get("code", "")
    if not code:
        raise ValueError("❌ Không có code trong rag_agent_output.json.")

    base_dir = "/content/drive/MyDrive/code_trong_day/kltn/output_parser"
    os.makedirs(f"{base_dir}/output", exist_ok=True)

    sol_path = f"{base_dir}/detected.sol"
    with open(sol_path, "w") as f:
        f.write(code)

    version = extract_version(code)
    contract_name = extract_contract_name(code)

    # Biên dịch và phân tích CFG
    os.system(f"solc --bin-runtime {sol_path} -o {base_dir}/output --overwrite")
    os.system(f"cp {base_dir}/output/{contract_name}.bin-runtime {base_dir}/detected.evm")
    report_path = f"{base_dir}/report.json"
    os.system(f"java -jar {base_dir}/EtherSolve.jar -r -j -o {report_path} {base_dir}/detected.evm")

    cfg = load_json(report_path)
    nodes = cfg["runtimeCfg"]["nodes"]
    edges = cfg["runtimeCfg"]["successors"]

    # === Add RAG knowledge node ===
    print(f"Before adding new node: {len(nodes)} nodes")

    max_offset = max((n["offset"] for n in nodes), default=0)
    rag_offset = max_offset + 1

    node_rag = {
        "offset": rag_offset,
        "type": "rag_knowledge",
        "length": 0,
        "stackBalance": 0,
        "bytecodeHex": "",
        "parsedOpcodes": "",
        "rag_output": {k: v for k, v in rag_output.items() if k not in ['code', 'functional_semantic']}
    }

    nodes.append(node_rag)
    if len(nodes) > 1:
        edges.append({"from": nodes[-2]["offset"], "to": [rag_offset]})

    report_with_rag = f"{base_dir}/report_with_rag.json"
    with open(report_with_rag, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"✅ Added RAG knowledge node at offset {rag_offset}")

    # === Build node feature vectors ===
    opcode_vocab, node_features, rag_indices = {}, [], []
    max_op_len, embed_dim, proj_dim, knowledge_dim = 20, 16, 64, 384

    text_embedder = HuggingFaceEmbeddings(model_name=TEXT_MODEL_NAME)

    def opcode_to_index(op):
        if op not in opcode_vocab:
            opcode_vocab[op] = len(opcode_vocab)
        return opcode_vocab[op]

    for idx, node in enumerate(nodes):
        if node.get("rag_output"):  # Knowledge node
            summary = node["rag_output"].get("summary", "")
            emb = text_embedder.embed_query(summary)
            node_features.append(emb)
            rag_indices.append(idx)
        elif node.get("parsedOpcodes"):
            ops = [line.split(":")[1].strip().split()[0]
                   for line in node["parsedOpcodes"].split("\n") if ":" in line]
            ids = [opcode_to_index(op) for op in ops][:max_op_len]
            ids += [0] * (max_op_len - len(ids))
            node_features.append(ids)
        else:
            node_features.append([0] * max_op_len)

    # === Project embeddings ===
    opcode_embed = Embedding(len(opcode_vocab), embed_dim)
    opcode_proj = Linear(embed_dim, proj_dim)
    knowledge_proj = Linear(knowledge_dim, proj_dim)

    final_features = []
    for idx, feat in enumerate(node_features):
        if idx in rag_indices:  # Knowledge node
            vec = torch.tensor(feat, dtype=torch.float32)
            final_features.append(knowledge_proj(vec))
        else:  # Opcode sequence
            idxs = torch.tensor(feat, dtype=torch.long)
            embed = opcode_embed(idxs).mean(dim=0)
            final_features.append(opcode_proj(embed))

    x_all = torch.stack(final_features, dim=0)

    # === Build edge index ===
    offset_to_idx = {n["offset"]: i for i, n in enumerate(nodes)}
    edge_pairs = [[offset_to_idx[e["from"]], offset_to_idx[dst]]
                  for e in edges for dst in e["to"]
                  if e["from"] in offset_to_idx and dst in offset_to_idx]

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()

    # === GAT Embedding ===
    gat = GATv2Conv(in_channels=proj_dim, out_channels=32, heads=2)
    out = gat(x_all, edge_index)
    cfg_emb = out.detach().cpu().numpy().mean(axis=0).tolist()

    return cfg_emb

# ============================================================
# === CodeBERT Embedding
# ============================================================

def extract_code_embedding(path: str) -> list:
    rag_output = load_json(path)
    code = rag_output.get("code", "")
    if not code:
        raise ValueError("❌ Không có Solidity code trong rag_agent_output.json.")

    inputs = code_tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = code_model(**inputs)

    return outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()

# ============================================================
# === Functional Semantic Embedding
# ============================================================

def extract_functional_semantic_embedding(path: str) -> list:
    rag_output = load_json(path)
    fs_text = rag_output.get("functional_semantic", "")
    if not fs_text:
        raise ValueError("❌ Không có functional_semantic trong rag_output.json.")
    return fs_embedder.embed_query(fs_text)

# ============================================================
# === CrewAI Tool / Agent / Task
# ============================================================

class EmbeddingToolInput(BaseModel):
    path: str = Field("rag_agent_output.json", description="Path to the JSON file contain rag agent output return above.")

class EmbeddingTool(BaseTool):
    name: str = "Parser Agent Generate Embeddings Tool"
    description: str = "Generating embeddings CFG, Code, Functional Semantic from value in rag_agent_output.json"
    args_schema: Type[BaseModel] = EmbeddingToolInput

    def _run(self, path: str, output_path: Optional[str] = None) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Không tìm thấy file: {path}")

        print(f"📂 Đang xử lý file: {path}")

        result = {
            "cfg_embeddings": extract_cfg_embedding(path),
            "source_code_embeddings": extract_code_embedding(path),
            "functional_semantic_embeddings": extract_functional_semantic_embedding(path)
        }

        output_path = output_path or "parser_output.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"✅ parser_output.json saved at: {output_path}")

        # ✅ Kết hợp Thought + JSON kết quả thành 1 return
        return (
            f"Thought: I now know the final answer.\n"
            f"Final Answer: 🎯\n\n{json.dumps(result, ensure_ascii=False, indent=2)}"
        )

# === CrewAI Agent & Task ===

embedding_tool = EmbeddingTool()

embedding_agent = Agent(
    role="Embeddings Agent",
    goal=(
        "Your primary objective is to generate and store embeddings for multiple data modalities "
        "from a given RAG analysis output file. These embeddings include:\n"
        "1. Control Flow Graph (CFG) Embeddings — structural representation of code logic.\n"
        "2. Code Embeddings — semantic representation of Solidity source code.\n"
        "3. Functional Semantic Embeddings — high-level functional meaning extracted from Gemini or LLM analysis.\n\n"
        "The embeddings will be saved into a single JSON file for later use in vulnerability prediction, "
        "graph reasoning, and retrieval-augmented generation."
    ),
    backstory=(
        "You are a specialized AI agent trained in embedding extraction for smart contract security analysis. "
        "Your role is to transform RAG Agent outputs into vectorized embeddings that represent code structure, "
        "semantics, and intent. These embeddings are used to build graph databases and support automated reasoning "
        "pipelines for vulnerability detection."
    ),
    tools=[embedding_tool],
    allow_delegation=False,
    verbose=True,
    llm=llm_local
)

embedding_task = Task(
    name="embedding_task",
    description=(
        "Read the file `rag_agent_output.json`, which contains the output of the RAG reasoning pipeline. "
        "This file includes information about the Solidity code, its contextual analysis, and functional semantics. "
        "You must call the `EmbeddingTool` to:\n"
        " - Extract and encode CFG embeddings (code structure graph)\n"
        " - Extract and encode code embeddings (source-level semantics)\n"
        " - Extract and encode functional semantic embeddings (behavioral understanding)\n\n"
        "After generating all three embeddings, write them to a new JSON file named `parser_output.json`."
    ),
    expected_output=(
        "`parser_output.json` — A JSON file containing the generated embeddings for CFG, Code, "
        "and Functional Semantic data. Each embedding should be a numerical vector array ready for "
        "storage in the knowledge graph or for input into ML-based vulnerability predictors."
    ),
    agent=embedding_agent,
    input={"path": "rag_agent_output.json"}
)

print("✅ Embedding Agent & Task initialized.")

print("✅ Embedding Agent & Task initialized.")
