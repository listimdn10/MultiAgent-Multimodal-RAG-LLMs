# agent_fusion.py (Version: Đã sửa lỗi NoneType và Lặp vô hạn)
import json, pickle, torch, numpy as np
import torch.nn as nn
from pydantic import BaseModel, Field
from typing import Type
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from llm_backend import build_chat_model
import joblib # Import joblib ở đầu
from tools.fusion_model import EarlyFusionModel # Đảm bảo bạn có file này

# ===========================
# --- HELPER VÀ GLOBAL READ (MỚI) ---
# ===========================

def safe_read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Could not read JSON file {path}: {e}")
        return {}

def safe_read_text(path):
    """Hàm helper để đọc file text một cách an toàn."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"⚠️ Could not read text file {path}: {e}")
        return ""

# ✅ Đọc file source code MỘT LẦN ở global, giống hệt consensus_agent.py
SOURCE_CODE_PATH = "contracts/sample.sol"
SOURCE_CODE_CONTENT = safe_read_text(SOURCE_CODE_PATH)

# ✅ Đọc file embedding MỘT LẦN ở global
SOURCE_EMBEDDING_PATH = "parser_output.json"
SOURCE_EMBEDDING_CONTENT = safe_read_json(SOURCE_EMBEDDING_PATH)

if SOURCE_CODE_CONTENT:
    print(f"✅ Đã đọc thành công file source code global: {SOURCE_CODE_PATH}")
else:
    print(f"❌ LỖI NGHIÊM TRỌNG: Không thể đọc file {SOURCE_CODE_PATH} ở global.")

if SOURCE_EMBEDDING_CONTENT:
    print(f"✅ Đã đọc thành công file embedding global: {SOURCE_EMBEDDING_PATH}")
else:
    print(f"❌ LỖI NGHIÊM TRỌNG: Không thể đọc file {SOURCE_EMBEDDING_PATH} ở global.")


# # ==== MLP MODEL (Giữ nguyên) ====
# class MLP(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(MLP, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_size, 256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(128, num_classes)
#         )

#     def forward(self, x):
#         return self.model(x)


# ==== TOOL (CẬP NHẬT) ====

# ✅ THAY ĐỔI 1: Tạo một class Pydantic Rỗng
class NoArgs(BaseModel):
    """Không nhận bất kỳ đối số nào."""
    pass

class EmbeddingPredictorTool(BaseTool):
    name: str = "Fusion Transformer Vulnerability Predictor and Line Finder"
    description: str = "Predicts security vulnerability from 3 embedding types using Fusion Transformer AND uses an LLM to find the vulnerable line."
    
    # ✅ THAY ĐỔI 1: Sử dụng class Pydantic rỗng thay vì None
    args_schema: Type[BaseModel] = NoArgs 
    _llm: ChatOpenAI = None

    def __init__(self, llm: ChatOpenAI, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm

    # -----------------------------------------
    # LOAD FUSION TRANSFORMER MODEL (NEW)
    # -----------------------------------------
    def _load_model(self, d_scode, d_fsem, d_cfg):
        # Load label encoder
        self._label_encoder = joblib.load("tools/early_fusion_label_encoder.pkl")
        num_classes = len(self._label_encoder.classes_)

        # Build model
        self._model = EarlyFusionModel(
            d_sc=d_scode,
            d_fs=d_fsem,
            d_cfg=d_cfg,
            n_classes=num_classes
        )
        # Load weights (Sử dụng đường dẫn của bạn)
        state = torch.load("tools/early_fusion.pth", map_location="cpu")
        expected = state["fc.0.weight"].shape[1]
        actual = d_scode + d_fsem + d_cfg
        trained_dims = (768, 768, 64)
        if (d_scode, d_fsem, d_cfg) != trained_dims:
            raise ValueError(
                "Fusion inputs must match the training workbook dimensions "
                f"(code, semantics, CFG)={trained_dims}; got "
                f"({d_scode}, {d_fsem}, {d_cfg})."
            )
        if actual != expected:
            raise ValueError(
                f"Fusion input has {actual} features, but the trained checkpoint "
                f"expects {expected} (code={d_scode}, semantics={d_fsem}, CFG={d_cfg})."
            )
        self._model.load_state_dict(state)
        self._model.eval()

    # Hàm _call_llm an toàn (Giữ nguyên)
    def _call_llm(self, prompt: str):
        if self._llm is None:
            raise RuntimeError("❌ No LLM provided to EmbeddingPredictorTool.")
        errors = []
        for method in ["invoke", "generate", "call"]:
            try:
                fn = getattr(self._llm, method, None)
                if callable(fn):
                    resp = fn(prompt)
                    if isinstance(resp, str):
                        return resp
                    if hasattr(resp, "content"):
                        return resp.content
                    if hasattr(resp, "text"):
                        return resp.text
                    return str(resp)
            except Exception as e:
                errors.append(f"{method} failed: {e}")
        raise RuntimeError("All LLM invocation attempts failed: " + " | ".join(errors))

    # ✅ Sửa hàm _run, không nhận 'path' nữa (bạn đã làm đúng)
    def _run(self) -> str:
        
        # --- 1. Phần Logic Model ---
        
        # Lấy data từ biến global
        data = safe_read_json(SOURCE_EMBEDDING_PATH)
        if not data:
             print(f"❌ Error: Biến global 'SOURCE_EMBEDDING_CONTENT' bị rỗng.")
             return "Error: Embedding data was not loaded globally."

        # ---- Extract 3 embedding vectors ----
        def extract(key):
            v = data.get(key, [])
            if isinstance(v, list) and len(v) > 0:
                v = v[0] if isinstance(v[0], list) else v
            return np.array(v, dtype=np.float32)

        fsem_vec  = extract("functional_semantic_embeddings")
        scode_vec = extract("code_embeddings")
        cfg_vec   = extract("cfg_embeddings")
        
        # Kiểm tra nếu vector rỗng
        if fsem_vec.size == 0 or scode_vec.size == 0 or cfg_vec.size == 0:
            error_msg = "Một hoặc nhiều vector embedding bị rỗng. Không thể dự đoán."
            print(f"❌ {error_msg}")
            return error_msg

        # dimensions
        d_fsem  = fsem_vec.shape[0]
        d_scode = scode_vec.shape[0]
        d_cfg   = cfg_vec.shape[0]

        # ---- Load model once ----
        if not hasattr(self, "_model"):
            self._load_model(d_scode, d_fsem, d_cfg)

        # ---- Convert to tensor ----
        fs = torch.tensor(fsem_vec, dtype=torch.float32).unsqueeze(0)
        sc = torch.tensor(scode_vec, dtype=torch.float32).unsqueeze(0)
        cf = torch.tensor(cfg_vec,  dtype=torch.float32).unsqueeze(0)

        # ---- Predict ----
        self._model.eval()  # Ensure model is in eval mode
        with torch.no_grad():
            logits = self._model(sc, fs, cf)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()

        label = self._label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(probs[0, pred_idx])

        print(f"✅ Fusion Predicted: {label} (Confidence: {confidence:.2%})")

        # --- 2. Phần Logic LLM ---
        
        # Lấy source code từ biến GLOBAL
        source_code = safe_read_text(SOURCE_CODE_PATH)
        
        if not source_code:
            print(f"❌ Error: Biến global 'SOURCE_CODE_CONTENT' bị rỗng.")
            return "Error: Source code was not loaded globally."

        prompt = f"""
        Here is a Solidity smart contract:
        ```solidity
        {source_code}
        ```
        An analysis model has predicted that this code contains a vulnerability of type: **{label}**.

        Your task is to analyze the source code and identify the specific line number(s) that are most likely responsible for this **{label}** vulnerability.

        Respond ONLY with the line number(s). For example: "Line 42" or "Lines 10-15". If you are unsure, respond "Unknown".
        """

        print(f"🤖 Calling LLM to find line number for {label}...")
        try:
            llm_response = self._call_llm(prompt) 
            predicted_line = llm_response.strip().replace("`", "")
        except Exception as e:
            print(f"❌ Error calling LLM: {e}")
            predicted_line = "Error calling LLM"
        
        print(f"✅ LLM Predicted Line: {predicted_line}")

        # --- 3. Lưu kết quả tổng hợp ---
        output = {
            "Predict": label, 
            "Confidence": confidence,
            "Predicted_Line_of_Vulnerability": predicted_line 
        }
        
        with open("fusion_output_agent.json", "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print("✅ Saved fusion_output_agent.json")

        # Trả về chuỗi kết quả đơn giản (bạn đã làm đúng)
        return f"Successfully predicted vulnerability: {label} (Confidence: {confidence:.2%}). Predicted Line: {predicted_line}. Output saved to fusion_output_agent.json."


def build_fusion_tool() -> EmbeddingPredictorTool:
    """Create the fusion node's classifier and line-localization tool."""
    return EmbeddingPredictorTool(llm=build_chat_model())
