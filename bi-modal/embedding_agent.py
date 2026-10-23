# -*- coding: utf-8 -*-
"""
embedding_agent.py
Purpose:
    - Load rag_output.json (kết quả của RAGRetrieveTool)
    - Thêm node tri thức vào CFG
    - Sinh embeddings: CFG, Code, Functional Semantic
    - Ghi parser_output.json
"""

import os
import json
import torch
import numpy as np
import re
from torch.nn import Embedding, Linear
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import RobertaTokenizer, RobertaModel
from typing import Type, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


# ----------------------------
# Helper functions
# ----------------------------

def load_rag_output(path):
    """Load RAG output JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_version(code):
    """Extract Solidity compiler version from pragma statement."""
    match = re.search(r"pragma\s+solidity\s+\^?([0-9]+\.[0-9]+(?:\.[0-9]+)?)", code)
    return match.group(1) if match else "0.8.0"


def extract_contract_name(code: str) -> str:
    """Extract contract name from Solidity code."""
    match = re.search(r'\bcontract\s+([A-Za-z_][A-Za-z0-9_]*)\b', code)
    if match:
        return match.group(1)
    else:
        raise ValueError("No contract name found in code.")


# ----------------------------
# CFG embedding extraction
# ----------------------------

def extract_cfg_embedding(path):
    rag_output = load_rag_output(path)
    code = rag_output.get("code", "")

    if not code:
        raise ValueError("No code found in rag_output.json.")

    # Compile and extract report
    with open("detected.sol", "w") as f:
        f.write(code)

    version = extract_version(code)
    contract_name = extract_contract_name(code)
    os.system(f"solc-select install {version}")
    os.system(f"solc-select use {version}")
    os.system(f"solc --bin-runtime detected.sol -o output --overwrite")
    # 👇 [SỬA 1] Thêm 'output/' vào đường dẫn nguồn
    os.system(f"cp output/{contract_name}.bin-runtime detected.evm")

    report_path = "report.json"
    os.system(f"java -jar EtherSolve.jar -r -j -o {report_path} detected.evm")

    with open(report_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Add RAG knowledge node
    nodes = cfg["runtimeCfg"]["nodes"]
    edges = cfg["runtimeCfg"]["successors"]

    print("Before adding new node, number of nodes:", len(nodes))

    max_offset = max((node["offset"] for node in nodes), default=0)
    rag_node_offset = max_offset + 1

    node_rag = {
        "offset": rag_node_offset,
        "type": "rag_knowledge",
        "length": 0,
        "stackBalance": 0,
        "bytecodeHex": "",
        "parsedOpcodes": "",
        "rag_output": {k: v for k, v in rag_output.items() if k not in ['code', 'functional_semantic']}
    }
    nodes.append(node_rag)

    print("After adding new node, number of nodes:", len(nodes))

    if len(nodes) > 1:
        prev_offset = nodes[-2]["offset"]
        edges.append({"from": prev_offset, "to": [rag_node_offset]})

    with open("report_with_rag.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"Đã thêm node tri thức RAG vào file CFG. Offset knowledge node: {rag_node_offset}")
    print("Node knowledge vừa thêm:")
    print(json.dumps(cfg["runtimeCfg"]["nodes"][-1], indent=2, ensure_ascii=False))

    # Embedding node features
    opcode_vocab = {}
    def opcode_to_index(op):
        if op not in opcode_vocab:
            opcode_vocab[op] = len(opcode_vocab)
        return opcode_vocab[op]

    max_op_len = 20
    embed_dim = 16
    proj_dim = 64
    knowledge_emb_dim = 384
    text_encoder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    node_features = []
    rag_node_indices = []

    for idx, node in enumerate(nodes):
        if node.get("rag_output"):
            rag_node_indices.append(idx)
            summary = node["rag_output"].get("summary", "")
            rag_emb = text_encoder.embed_query(summary)
            node_features.append(rag_emb)
        elif node.get("parsedOpcodes"):
            opcodes = [line.split(":")[1].strip().split()[0] for line in node["parsedOpcodes"].split("\n")]
            opcode_ids = [opcode_to_index(op) for op in opcodes]
            padded = opcode_ids[:max_op_len] + [0] * (max_op_len - len(opcode_ids))
            node_features.append(padded)
        else:
            node_features.append([0] * max_op_len)

    node_features = np.array(node_features, dtype=object)
    knowledge_proj = Linear(knowledge_emb_dim, proj_dim)
    opcode_embed = Embedding(len(opcode_vocab), embed_dim)
    opcode_proj = Linear(embed_dim, proj_dim)

    final_features = []
    for idx, feat in enumerate(node_features):
        if idx in rag_node_indices:
            vec = torch.tensor(feat, dtype=torch.float32)
            vec_proj = knowledge_proj(vec)
            final_features.append(vec_proj)
        else:
            idxs = torch.tensor(feat, dtype=torch.long)
            embed = opcode_embed(idxs).mean(dim=0)
            vec_proj = opcode_proj(embed)
            final_features.append(vec_proj)

    x_all = torch.stack(final_features, dim=0)

    offset_to_idx = {node["offset"]: idx for idx, node in enumerate(nodes)}
    edge_index = []
    for entry in edges:
        src = entry["from"]
        for dst in entry["to"]:
            if src in offset_to_idx and dst in offset_to_idx:
                edge_index.append([offset_to_idx[src], offset_to_idx[dst]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    data = Data(x=x_all, edge_index=edge_index)
    gat = GATv2Conv(in_channels=proj_dim, out_channels=32, heads=2)
    output = gat(data.x, data.edge_index)
    cfg_embeddings = output.detach().cpu().numpy().mean(axis=0).tolist()

    return cfg_embeddings


# ============================================================
# CodeBERT & Functional Semantic Embeddings
# ============================================================

def extract_and_embed_code(path):
    rag_output = load_rag_output(path)
    code = rag_output.get("code", "")

    if not code:
        raise ValueError("No Solidity code found in rag_output.json.")

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaModel.from_pretrained("microsoft/codebert-base")

    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state[:, 0, :]
    src_code_embeddings = embedding.detach().cpu().numpy().tolist()

    return src_code_embeddings


def embed_functional_semantics(path):
    rag_output = load_rag_output(path)
    functional_semantic = rag_output.get("functional_semantic", "")

    if not functional_semantic:
        raise ValueError("No functional_semantic found in rag_output.json.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embedded_query = embeddings.embed_query(functional_semantic)

    return embedded_query


# ============================================================
# LangGraph embedding node tool
# ============================================================

# ✅ THAY ĐỔI 1: Sửa Input Schema (giống DummyInput)
class EmbeddingToolInput(BaseModel):
    """Không nhận tham số, tool sẽ tự đọc file rag_output.json."""
    pass


class EmbeddingTool(BaseTool):
    name: str = "Embedding Tool"
    description: str = "Generate embeddings for CFG, Code, and Functional Semantic from rag_output.json"
    args_schema: Type[BaseModel] = EmbeddingToolInput  # Sửa schema

    # ✅ THAY ĐỔI 2: Sửa chữ ký hàm _run (bỏ 'path')
    def _run(self) -> str:
        
        # ✅ THAY ĐỔI 3: Hardcode đường dẫn file input
        input_file_path = "rag_output.json"
        output_path = "parser_output.json"

        # Kiểm tra file tồn tại
        if not os.path.exists(input_file_path):
            # Đây là nơi lỗi của bạn xảy ra, nên raise lỗi rõ ràng
            raise FileNotFoundError(f"❌ Không tìm thấy file input: {input_file_path}. Có thể RAG Agent đã chạy lỗi.")
        
        print(f"📂 Đang đọc file input: {input_file_path}")

        try:
            cfg_embeddings = extract_cfg_embedding(input_file_path)
            code_embeddings = extract_and_embed_code(input_file_path)
            semantic_embeddings = embed_functional_semantics(input_file_path)

            # Gộp kết quả
            parser_output = {
                "cfg_embeddings": cfg_embeddings,
                "code_embeddings": code_embeddings,
                "functional_semantic_embeddings": semantic_embeddings
            }

            # Ghi output file (Logic giống mẫu bạn đưa)
            dir_path = os.path.dirname(output_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(parser_output, f, indent=2, ensure_ascii=False)

            print(f"✅ Đã lưu parser_output.json tại: {output_path}")

            # ✅ QUAN TRỌNG: Trả về string để Agent biết nhiệm vụ đã xong (tránh loop)
            return f"TASK COMPLETED. Embeddings generated and saved to {output_path}. You can stop now."

        except Exception as e:
            print(f"❌ Lỗi trong quá trình sinh embedding: {e}")
            return f"Error generating embeddings: {e}"

embedding_tool = EmbeddingTool()
