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
from transformers import AutoTokenizer, AutoModel
from typing import Type, Optional
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import sys, io

# Ensure console stdout/stderr are UTF-8 encoded on Windows to avoid
# UnicodeEncodeError when printing emoji characters.
try:
    if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if sys.stderr.encoding is None or sys.stderr.encoding.lower() != "utf-8":
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

# Thiết lập thiết bị (GPU nếu có)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Import từ module RAG (đã có llm_local, rag_agent, rag_task)
from rag_agent import llm_local, rag_agent, rag_task


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

    print(f"Code start: {repr(code[:100])}")
    # Compile and extract report
    with open("detected.sol", "w") as f:
        f.write(code)

    version = extract_version(code)
    print(f"Detected Solidity version: {version}")
    contract_name = extract_contract_name(code)
    os.system(f"solc-select install {version}")
    os.system(f"solc-select use {version}")
    os.system(f"solc --bin-runtime detected.sol -o output --overwrite")
    # 👇 [SỬA 1] Thêm 'output/' vào đường dẫn nguồn
    # 👇 [SỬA 2] Dùng lệnh cross-platform thay vì 'cp'
    import shutil
    shutil.copy(f"output/{contract_name}.bin-runtime", "detected.evm")

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
        "rag_output": {"Audit_report": rag_output.get("Audit_report", "")}
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
    proj_dim = 768
    knowledge_emb_dim = 768
    text_encoder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    node_features = []
    rag_node_indices = []

    for idx, node in enumerate(nodes):
        if node.get("rag_output"):
            rag_node_indices.append(idx)
            summary = node["rag_output"].get("Audit_report", "")
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
    gat = GATv2Conv(in_channels=proj_dim, out_channels=224, heads=2)
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

    # Tải model và tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    model = AutoModel.from_pretrained("microsoft/graphcodebert-base").to(device)

    def get_code_embedding_sliding_window(code_text, window_size=510, stride=256):
        """
        Hàm tạo embedding cho code dài sử dụng sliding window.
        window_size=510 chừa chỗ cho 2 special tokens [CLS] và [SEP].
        """
        if not isinstance(code_text, str) or not code_text.strip():
            # Trả về vector 0 nếu code rỗng (kích thước hidden size của GraphCodeBERT là 768)
            return np.zeros(768)

        # Tokenize toàn bộ đoạn code
        tokens = tokenizer(code_text, add_special_tokens=True, return_tensors='pt', truncation=False, padding=False)
        input_ids = tokens['input_ids'][0] # Lấy tensor 1 chiều

        # Nếu code ngắn hơn window, chạy thẳng model
        if len(input_ids) <= 512:
            input_ids = input_ids.unsqueeze(0).to(device) # Batch size = 1
            with torch.no_grad():
                outputs = model(input_ids)
                # Lấy vector [CLS] (token đầu tiên) hoặc mean pooling
                # Ở đây dùng mean pooling của last_hidden_state để đại diện tốt hơn
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            return embedding

        # Nếu code dài, dùng sliding window
        all_chunk_embeddings = []

        # Duyệt qua các cửa sổ
        # Lưu ý: GraphCodeBERT max len là 512.
        for i in range(0, len(input_ids), stride):
            chunk_ids = input_ids[i : i + 512]

            # Bỏ qua nếu chunk quá ngắn (ví dụ phần dư cuối cùng < 10 token)
            if len(chunk_ids) < 10:
                continue

            chunk_ids = chunk_ids.unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(chunk_ids)
                # Lấy mean pooling của chunk hiện tại
                chunk_emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                all_chunk_embeddings.append(chunk_emb)

        if len(all_chunk_embeddings) > 0:
            # Trung bình cộng các vector của các chunk
            final_embedding = np.mean(all_chunk_embeddings, axis=0)
            return final_embedding
        else:
            return np.zeros(768)

    embedding = get_code_embedding_sliding_window(code)
    src_code_embeddings = embedding.tolist()

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
# CrewAI Tool, Agent & Task
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

            print(f"Dimensions: CFG={len(cfg_embeddings)}, Code={len(code_embeddings)}, Semantic={len(semantic_embeddings)}, Total={len(cfg_embeddings) + len(code_embeddings) + len(semantic_embeddings)}")

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

# ---- Define Agent and Task ----
embedding_tool = EmbeddingTool()

embedding_agent = Agent(
    role="Embeddings Agent",
    goal="Generate embeddings for CFG, Code, and Functional Semantic from RAG results.",
    backstory=(
        "You are an expert in CFG embedding and Solidity code. "
        "Your task is to create multi-dimensional embedding representations "
        "for code, CFG graphs, and semantic descriptions to support security analysis."
    ),
    tools=[embedding_tool],
    allow_delegation=False,
    verbose=True,
    llm=llm_local,
    max_iter=1 
)

embedding_task = Task(
    name="embedding_task",
    description="Use EmbeddingTool to generate embeddings for CFG, Code, and Functional Semantic from rag_output.json.",
    expected_output="parser_output.json containing embeddings for 3 data types.",
    agent=embedding_agent,
)

print("✅ Embedding Agent & Task initialized.")

# Test run
if __name__ == "__main__":
    print("Testing Embedding Agent with Crew...")
    crew = Crew(agents=[embedding_agent], tasks=[embedding_task])
    result = crew.kickoff()
    print("Result:", result)
