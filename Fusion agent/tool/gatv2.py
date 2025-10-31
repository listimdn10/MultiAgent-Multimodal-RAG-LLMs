import os
import csv
import re
import subprocess
import json
import torch
import pandas as pd
import numpy as np
from torch.nn import Embedding
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

# ========== CONFIG ==========
input_excel = "solidity_code_with_labels.xlsx"
output_dir = "gatv2_embeddings"
os.makedirs(output_dir, exist_ok=True)

# ========== UTILS ==========
def extract_version(code):
    match = re.search(r"pragma\s+solidity\s+\^?([0-9]+\.[0-9]+(?:\.[0-9]+)?)", code)
    if match:
        return match.group(1)
    else:
        raise ValueError("Solidity version not found.")

def extract_contract_name(code):
    match = re.search(r'\bcontract\s+([A-Za-z_][A-Za-z0-9_]*)\b', code)
    if match:
        return match.group(1)
    else:
        raise ValueError("Contract name not found.")

def write_temp_sol_file(code, filename="temp.sol"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(code)

def run_shell(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ CMD error:", cmd)
        print(result.stderr)
    return result

# ========== MAIN LOOP ==========
df = pd.read_excel(input_excel)
all_embeddings = []

for idx, row in df.iterrows():
    code = row["code"]
    label = row["label"]
    try:
        print(f"🔍 [{idx+1}/{len(df)}] Processing...")

        # Step 1: Trích version và contract name
        version = extract_version(code)
        contract_name = extract_contract_name(code)

        # Step 2: Chọn đúng phiên bản solc
        run_shell(f"solc-select install {version}")
        run_shell(f"solc-select use {version}")

        # Step 3: Ghi file .sol và compile
        write_temp_sol_file(code, "temp.sol")
        run_shell(f"solc --bin-runtime temp.sol -o output --overwrite")

        # Step 4: Lấy .bin-runtime
        bin_path = f"output/{contract_name}.bin-runtime"
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"{bin_path} not found")

        run_shell(f"cp {bin_path} temp.evm")

        # Step 5: Sinh CFG bằng EtherSolve
        run_shell(f"java -jar EtherSolve.jar -c -H temp.evm")
        run_shell(f"java -jar EtherSolve.jar -r -j -o report.json temp.evm")

        # Step 6: Load report.json và xử lý GATv2
        with open("report.json") as f:
            cfg_json = json.load(f)

        nodes = cfg_json["runtimeCfg"]["nodes"]
        edges_raw = cfg_json["runtimeCfg"]["successors"]

        opcode_vocab = {}
        def opcode_to_index(op):
            if op not in opcode_vocab:
                opcode_vocab[op] = len(opcode_vocab)
            return opcode_vocab[op]

        max_op_len = 20
        node_features = []

        for node in nodes:
            if node["parsedOpcodes"]:
                opcodes = [line.split(":")[1].strip().split()[0]
                           for line in node["parsedOpcodes"].split("\n") if line.strip()]
                opcode_ids = [opcode_to_index(op) for op in opcodes]
                padded = opcode_ids[:max_op_len] + [0] * (max_op_len - len(opcode_ids))
            else:
                padded = [0] * max_op_len
            node_features.append(padded)

        x = torch.tensor(node_features, dtype=torch.long)
        embed = Embedding(len(opcode_vocab), 16)
        x_embed = embed(x).mean(dim=1)

        offset_to_idx = {node["offset"]: i for i, node in enumerate(nodes)}
        edge_index = []
        for entry in edges_raw:
            src = entry["from"]
            for dst in entry["to"]:
                if src in offset_to_idx and dst in offset_to_idx:
                    edge_index.append([offset_to_idx[src], offset_to_idx[dst]])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        data = Data(x=x_embed, edge_index=edge_index)
        gat = GATv2Conv(in_channels=16, out_channels=32, heads=2)
        output = gat(data.x, data.edge_index)
        node_embedding = output.mean(dim=0).detach().cpu().numpy()  # embedding tổng cho contract

        # Lưu embedding + label
        all_embeddings.append(np.concatenate([[label], node_embedding]))

    except Exception as e:
        print(f"❌ [{idx+1}] Lỗi: {e}")
        continue

# Save all to CSV
header = ["label"] + [f"dim_{i}" for i in range(64)]
df_out = pd.DataFrame(all_embeddings, columns=header)
df_out.to_csv(os.path.join(output_dir, "contract_embeddings.csv"), index=False)
print("✅ DONE: Embeddings saved.")
