# consensus_agent.py (Phiên bản đã sửa)
import json
import os
import re
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from llm_backend import build_chat_model

# --- Import các thư viện RAG ---
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ===========================
# --- KHU VỰC LOẠI BỎ LOGIC TRÍCH XUẤT MD ---
# ===========================
# ✅ Xóa toàn bộ logic trích xuất từ 'multimodal-Audit.md' thành 'multi_modal.md'
#    vì chúng ta sẽ đọc trực tiếp 'explainer_output.json'


# ===========================
# --- CONFIG GLOBAL PATH (ĐÃ CẬP NHẬT) ---
# ===========================

SOURCE_PATH = "contracts/sample.sol"
RAG_PATH = "rag_output.json"
EXPLAINER_PATH = "explainer_output.json" # <--- ĐÃ SỬA: Đọc file JSON trực tiếp

def safe_read_text(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"⚠️ Could not read text file {path}: {e}")
        return ""

def safe_read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Could not read JSON file {path}: {e}")
        return {}


# ===========================
# --- Input Schema (Giữ nguyên)
# ===========================
class ConsensusInput(BaseModel):
    pass

# ===========================
# --- Consensus Tool Definition
# ===========================
class ConsensusTool(BaseTool):
    name: str = "ConsensusTool"
    description: str = (
        "Compare two audit outputs (RAG and Explainer) plus the source code, "
        "then decide which audit is more accurate or merge their results."
    )
    args_schema: Type[BaseModel] = ConsensusInput

    def __init__(self, llm=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._llm = llm

    # ... (Hàm _call_llm giữ nguyên) ...
    def _call_llm(self, prompt: str):
        if self._llm is None:
            raise RuntimeError("❌ No LLM provided to ConsensusTool.")
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

    # ---------- Main Run (ĐÃ CẬP NHẬT LOGIC) ----------
    def _run(self, source_path: str, rag_path: str, explainer_path: str) -> dict:
        source_code = safe_read_text(source_path)
        rag_json = safe_read_json(rag_path)
        explainer_json = safe_read_json(explainer_path)
        if not source_code or not rag_json or not explainer_json:
            raise ValueError("Consensus requires source code, RAG output, and Reason output")

        # =================================================================
        # 2. (CẬP NHẬT) LỌC DỮ LIỆU: Chỉ lấy 'vuln_type' và 'Audit_report'
        # =================================================================
        
        # --- Lọc dữ liệu RAG ---
        # Ưu tiên lấy 'vuln_type', nếu không có thì thử 'Predict' (đề phòng format cũ)
        rag_vuln_type = rag_json.get("vuln_type") or rag_json.get("Predict", "Unknown")
        rag_filtered = {
            "vuln_type": rag_vuln_type,
            "Audit_report": rag_json.get("Audit_report", "No report provided.")
        }

        # --- Lọc dữ liệu Explainer ---
        explainer_vuln_type = explainer_json.get("vuln_type", "Unknown")
        explainer_filtered = {
            "vuln_type": explainer_vuln_type,
            "Audit_report": explainer_json.get("Audit_report", "No report provided.")
        }

        print(f"🔍 Đã xác định Vuln Types để truy vấn Knowledge: RAG='{rag_vuln_type}', Explainer='{explainer_vuln_type}'")

        # 4. (Giữ nguyên) Truy vấn Vectorstore TÁCH BIỆT
        
        # Lazily initialize the retriever once per audit, not once per branch.
        retriever = None

        # Hàm helper để truy vấn và gộp context (Giữ nguyên)
        def get_knowledge_context(query: str) -> str:
            nonlocal retriever
            if not query or query == "Unknown":
                return "No vulnerability type provided to search."
            try:
                if retriever is None:
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    vector_store = Chroma(
                        persist_directory=os.environ.get("CONVERA_CHROMA_DB", "knowledge_base/chroma_db"),
                        embedding_function=embeddings,
                    )
                    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
                print(f"📚 Truy vấn KB cho: '{query}'")
                docs = retriever.invoke(query) 
                return "\n\n---\n\n".join([doc.page_content for doc in docs])
            except Exception as e:
                print(f"⚠️ Lỗi truy vấn vectorstore: {e}")
                return f"Error retrieving knowledge: {e}"

        # Lấy context dựa trên vuln_type đã lọc
        rag_knowledge_context = get_knowledge_context(rag_vuln_type)
        explainer_knowledge_context = get_knowledge_context(explainer_vuln_type)

        print(f"📚 Đã truy xuất {len(rag_knowledge_context)} chars cho RAG.")
        print(f"📚 Đã truy xuất {len(explainer_knowledge_context)} chars cho Explainer.")

        prompt = f"""You are the Supreme Arbitrator and Final Judge for Smart Contract Security.
        You are reviewing a case where an "Auditor" (RAG Agent) flagged a potential vulnerability, and a "Critic" (Explainer Agent) verified it.

        Your duty is to review the Source Code, the Auditor's initial claim, the Critic's verification, and the historical Knowledge Context to make the FINAL, authoritative ruling.

        --- SOURCE CODE ---
        {source_code}
        --- END SOURCE CODE ---

        --- THE AUDITOR'S FINDINGS (RAG AGENT) ---
        (Focuses on finding all potential issues; may contain false positives)
        {json.dumps(rag_filtered, ensure_ascii=False, indent=2)}
        
        --- AUDITOR'S KNOWLEDGE CONTEXT ---
        {rag_knowledge_context}
        --- END AUDITOR KNOWLEDGE ---

        --- THE CRITIC'S VERIFICATION (EXPLAINER AGENT) ---
        (Strictly skeptical; verifies if the Auditor's claim is actually exploitable)
        {json.dumps(explainer_filtered, ensure_ascii=False, indent=2)}

        --- CRITIC'S KNOWLEDGE CONTEXT ---
        {explainer_knowledge_context}
        --- END CRITIC KNOWLEDGE ---

        TASK:
        1) Review the Auditor's claim: Did they find a legitimate attack vector?
        2) Review the Critic's verification: Was their confirmation or rejection logical and backed by the code?
        3) Make the FINAL DECISION: Which agent's conclusion is most accurate?
           - Choose "RAG" if the Auditor is correct (the vulnerability is real).
           - Choose "Explainer" if the Critic is correct (e.g., they successfully proved it's a False Positive, or gave a much better explanation).
           - Choose "Merged" if both have valid points that need to be combined.
        4) Output ONLY valid JSON:

        {{
            "decision": "RAG" | "Explainer" | "Merged",
            "reasoning": "Step-by-step justification for your final ruling based on the code and the two reports.",
            "final_vulnerability_summary": "One-sentence summary of the confirmed vulnerability, or 'No vulnerability' if it is a False Positive.",
            "confidence": float
        }}
        """

        # ... (Phần Call LLM và Parse JSON... giữ nguyên) ...
        # Call LLM
        try:
            llm_text = self._call_llm(prompt)
        except Exception as e:
            fallback = {
                "decision": "FAILED",
                "reasoning": f"LLM invocation failed: {e}",
                "final_vulnerability_summary": rag_json.get("vuln_type", "") or explainer_vuln_type,
                "confidence": 0.0
            }
            with open("consensus_output.json", "w", encoding="utf-8") as f:
                json.dump(fallback, f, ensure_ascii=False, indent=2)
            return fallback
        with open("consensus_raw.txt", "w", encoding="utf-8") as f:
            f.write(llm_text)

        # Parse JSON output
        parsed = None
        decoder = json.JSONDecoder(strict=False)
        for start in (match.start() for match in re.finditer(r"\{", llm_text)):
            try:
                candidate, _ = decoder.raw_decode(llm_text[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and "decision" in candidate:
                parsed = candidate
                break

        if not parsed:
            result = {
                "decision": "FAILED",
                "reasoning": "Consensus response was not parseable as a decision JSON object.",
                "final_vulnerability_summary": rag_json.get("Predict", "") or explainer_vuln_type,
                "confidence": 0.0
            }
        else:
            result = {
                "decision": parsed.get("decision", "Merged"),
                "reasoning": parsed.get("reasoning", ""),
                "final_vulnerability_summary": parsed.get("final_vulnerability_summary", ""),
                "confidence": float(parsed.get("confidence", 0.5))
            }

        with open("consensus_output.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print("✅ Saved consensus_output.json")
        return result


def build_consensus_tool() -> ConsensusTool:
    model = build_chat_model()
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if provider in ("", "mistral") and (os.environ.get("MISTRAL_API_KEY") or os.environ.get("mistral")):
        model = model.bind(response_format={"type": "json_object"})
    return ConsensusTool(llm=model)


if __name__ == "__main__":
    from langgraph_pipeline import build_audit_graph
    print(build_audit_graph(("consensus",)).invoke({}))
