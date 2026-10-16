import json
import os
import re
import shutil
import subprocess
from functools import lru_cache
from typing import Dict, Any

import numpy as np
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from torch.nn import Embedding, Linear
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from transformers import AutoModel, AutoTokenizer
import joblib

from tools.fusion_model import EarlyFusionModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def extract_version(code: str) -> str:
    match = re.search(r"pragma\s+solidity\s+\^?([0-9]+\.[0-9]+(?:\.[0-9]+)?)", code)
    return match.group(1) if match else "0.8.20"


def extract_contract_name(code: str) -> str:
    match = re.search(r"\bcontract\s+([A-Za-z_][A-Za-z0-9_]*)\b", code)
    if not match:
        raise ValueError("No contract name found in code.")
    return match.group(1)


@lru_cache(maxsize=1)
def get_text_encoder():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


@lru_cache(maxsize=1)
def get_code_encoder():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    model = AutoModel.from_pretrained("microsoft/graphcodebert-base").to(DEVICE)
    model.eval()
    return tokenizer, model


def build_cfg_fallback_embedding(code: str, rag_report: str = "", reason: str = ""):
    text_encoder = get_text_encoder()
    combined_text = "\n".join(
        [
            "CFG fallback embedding",
            reason[:300],
            rag_report[:1200],
            code[:4000],
        ]
    ).strip()

    semantic_vec = np.array(text_encoder.embed_query(combined_text), dtype=np.float32)
    if semantic_vec.shape[0] < 384:
        semantic_vec = np.pad(semantic_vec, (0, 384 - semantic_vec.shape[0]))

    stats = np.array(
        [
            code.count("function"),
            code.count("mapping"),
            code.count("require"),
            code.count("if"),
            code.count("for"),
            code.count("while"),
            code.count("return"),
            code.count("emit"),
        ],
        dtype=np.float32,
    )
    stats = np.tile(stats, 8)
    fallback = np.concatenate([semantic_vec[:384], stats], axis=0)
    return fallback[:448].tolist()


def extract_cfg_embedding_from_rag(rag_output_path: str):
    rag_output = load_json(rag_output_path)
    code = rag_output.get("code", "")
    rag_report = rag_output.get("Audit_report", "")

    if not code:
        raise ValueError("No code found in rag output.")

    try:
        with open("detected.sol", "w", encoding="utf-8") as f:
            f.write(code)

        contract_name = extract_contract_name(code)
        compile_cmd = ["solc", "--bin-runtime", "detected.sol", "-o", "output", "--overwrite"]
        compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if compile_result.returncode != 0:
            raise RuntimeError(f"solc compilation failed: {compile_result.stderr.strip()}")

        runtime_path = os.path.join("output", f"{contract_name}.bin-runtime")
        if not os.path.exists(runtime_path):
            raise FileNotFoundError(f"Runtime bytecode not found: {runtime_path}")
        shutil.copy(runtime_path, "detected.evm")

        report_path = "report.json"
        ethersolve_cmd = ["java", "-jar", "EtherSolve.jar", "-r", "-j", "-o", report_path, "detected.evm"]
        ethersolve_result = subprocess.run(ethersolve_cmd, capture_output=True, text=True)
        if ethersolve_result.returncode != 0:
            raise RuntimeError(f"EtherSolve failed: {ethersolve_result.stderr.strip()}")

        with open(report_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        nodes = cfg["runtimeCfg"]["nodes"]
        edges = cfg["runtimeCfg"]["successors"]
        max_offset = max((node["offset"] for node in nodes), default=0)
        rag_node_offset = max_offset + 1
        nodes.append(
            {
                "offset": rag_node_offset,
                "type": "rag_knowledge",
                "length": 0,
                "stackBalance": 0,
                "bytecodeHex": "",
                "parsedOpcodes": "",
                "rag_output": {"Audit_report": rag_report},
            }
        )
        if len(nodes) > 1:
            edges.append({"from": nodes[-2]["offset"], "to": [rag_node_offset]})

        opcode_vocab = {}

        def opcode_to_index(op):
            if op not in opcode_vocab:
                opcode_vocab[op] = len(opcode_vocab)
            return opcode_vocab[op]

        text_encoder = get_text_encoder()
        node_features = []
        rag_node_indices = []
        max_op_len = 20
        embed_dim = 16
        proj_dim = 768
        knowledge_emb_dim = 768

        for idx, node in enumerate(nodes):
            if node.get("rag_output"):
                rag_node_indices.append(idx)
                summary = node["rag_output"].get("Audit_report", "")
                node_features.append(text_encoder.embed_query(summary))
            elif node.get("parsedOpcodes"):
                opcodes = [line.split(":")[1].strip().split()[0] for line in node["parsedOpcodes"].split("\n")]
                opcode_ids = [opcode_to_index(op) for op in opcodes]
                padded = opcode_ids[:max_op_len] + [0] * (max_op_len - len(opcode_ids))
                node_features.append(padded)
            else:
                node_features.append([0] * max_op_len)

        node_features = np.array(node_features, dtype=object)
        knowledge_proj = Linear(knowledge_emb_dim, proj_dim)
        opcode_embed = Embedding(max(1, len(opcode_vocab)), embed_dim)
        opcode_proj = Linear(embed_dim, proj_dim)
        final_features = []
        for idx, feat in enumerate(node_features):
            if idx in rag_node_indices:
                vec = torch.tensor(feat, dtype=torch.float32)
                final_features.append(knowledge_proj(vec))
            else:
                idxs = torch.tensor(feat, dtype=torch.long)
                final_features.append(opcode_proj(opcode_embed(idxs).mean(dim=0)))

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
        return output.detach().cpu().numpy().mean(axis=0).tolist()
    except Exception as e:
        return build_cfg_fallback_embedding(code, rag_report=rag_report, reason=str(e))


def extract_code_embedding_from_rag(rag_output_path: str):
    rag_output = load_json(rag_output_path)
    code = rag_output.get("code", "")
    if not code:
        raise ValueError("No Solidity code found in rag output.")

    tokenizer, model = get_code_encoder()
    tokens = tokenizer(code, add_special_tokens=True, return_tensors="pt", truncation=False, padding=False)
    input_ids = tokens["input_ids"][0]

    def run_chunk(chunk_ids):
        chunk_ids = chunk_ids.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(chunk_ids)
            return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    if len(input_ids) <= 512:
        return run_chunk(input_ids)[:384].tolist()

    all_chunk_embeddings = []
    stride = 256
    for i in range(0, len(input_ids), stride):
        chunk_ids = input_ids[i : i + 512]
        if len(chunk_ids) < 10:
            continue
        all_chunk_embeddings.append(run_chunk(chunk_ids))

    if not all_chunk_embeddings:
        return np.zeros(384).tolist()
    return np.mean(all_chunk_embeddings, axis=0)[:384].tolist()


def extract_semantic_embedding_from_rag(rag_output_path: str):
    rag_output = load_json(rag_output_path)
    functional_semantic = rag_output.get("functional_semantic", "")
    if not functional_semantic:
        raise ValueError("No functional_semantic found in rag output.")
    return get_text_encoder().embed_query(functional_semantic)


def generate_parser_output(rag_output_path: str = "rag_output.json", output_path: str = "parser_output.json") -> Dict[str, Any]:
    parser_output = {
        "cfg_embeddings": extract_cfg_embedding_from_rag(rag_output_path),
        "code_embeddings": extract_code_embedding_from_rag(rag_output_path),
        "functional_semantic_embeddings": extract_semantic_embedding_from_rag(rag_output_path),
    }
    save_json(output_path, parser_output)
    return parser_output


@lru_cache(maxsize=1)
def get_fusion_label_encoder():
    return joblib.load("tools/early_fusion_label_encoder.pkl")


@lru_cache(maxsize=1)
def get_fusion_model(d_scode: int, d_fsem: int, d_cfg: int):
    label_encoder = get_fusion_label_encoder()
    model = EarlyFusionModel(
        d_sc=d_scode,
        d_fs=d_fsem,
        d_cfg=d_cfg,
        n_classes=len(label_encoder.classes_),
    )
    state = torch.load("tools/early_fusion.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def generate_fusion_seed(parser_output_path: str = "parser_output.json", output_path: str = "fusion_output_agent.json") -> Dict[str, Any]:
    data = load_json(parser_output_path)

    def extract(key):
        value = data.get(key, [])
        if isinstance(value, list) and value and isinstance(value[0], list):
            value = value[0]
        return np.array(value, dtype=np.float32)

    fsem_vec = extract("functional_semantic_embeddings")
    scode_vec = extract("code_embeddings")
    cfg_vec = extract("cfg_embeddings")

    model = get_fusion_model(scode_vec.shape[0], fsem_vec.shape[0], cfg_vec.shape[0])
    label_encoder = get_fusion_label_encoder()

    fs = torch.tensor(fsem_vec, dtype=torch.float32).unsqueeze(0)
    sc = torch.tensor(scode_vec, dtype=torch.float32).unsqueeze(0)
    cf = torch.tensor(cfg_vec, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(sc, fs, cf)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

    output = {
        "Predict": label_encoder.inverse_transform([pred_idx])[0],
        "Confidence": float(probs[0, pred_idx]),
        "Predicted_Line_of_Vulnerability": "Unknown",
        "Generated_By": "business_bench_utils.generate_fusion_seed",
    }
    save_json(output_path, output)
    return output
