# consensus_agent.py (Version: inline content)
import json
import os
import re
from typing import Type
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

# --- Import các thư viện RAG ---
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ===========================
# --- CONFIG GLOBAL PATH ---
# ===========================

# ===========================
# --- TRÍCH XUẤT MD MỚI ---
# ===========================

import json
import re

input_path = "multimodal-Audit.md"
output_path = "multi_modal.md" # <-- Đổi sang .md

# Đọc toàn bộ file .md
with open(input_path, "r", encoding="utf-8") as f:
    md_content = f.read()

# Lấy tất cả JSON trong raw='...'
json_matches = re.findall(r"raw='(\{.*?\})'", md_content, flags=re.DOTALL)

if not json_matches:
    raise ValueError("Không tìm thấy JSON nào trong file .md")

# Lấy chuỗi "raw" cuối cùng
raw_content = json_matches[-1]

# --- Helper function để trích xuất an toàn bằng Regex ---
def extract_field(content, key):
    # 1. Thử tìm giá trị là string: "key": "value"
    pattern_str = rf'"{key}":\s*"(.*?)"'
    match = re.search(pattern_str, content, flags=re.DOTALL)
    if match:
        # Dọn dẹp escape chars
        return match.group(1).replace("\\n", "\n").replace("\\t", "\t").strip()
    
    # 2. Thử tìm giá trị là list: "key": [...]
    pattern_list = rf'"{key}":\s*(\[.*?\])'
    match_list = re.search(pattern_list, content, flags=re.DOTALL)
    if match_list:
        return match_list.group(1).strip()

    # 3. Thử tìm giá trị không có quote (như 100.00%): "key": value,
    # (Tìm đến dấu phẩy của key tiếp theo, hoặc dấu } )
    pattern_other = rf'"{key}":\s*(.*?)(?:,\s*"\w+"|\s*\}})'
    match_other = re.search(pattern_other, content, flags=re.DOTALL)
    if match_other:
        return match_other.group(1).strip()
        
    return "N/A"
# --- End Helper ---

# 6. Lấy các trường quan trọng bằng Regex
important_fields = {
    "security_vulnerability": extract_field(raw_content, "security_vulnerability"),
    "confidence_score": extract_field(raw_content, "confidence_score"),
    "description": extract_field(raw_content, "description"),
    "vuln_type": extract_field(raw_content, "vuln_type"),
    "solutions": extract_field(raw_content, "solutions"),
    "context": extract_field(raw_content, "context")
}

# 7. Ghi ra file MD mới
with open(output_path, "w", encoding="utf-8") as f:
    f.write("# Extracted Audit Report\n\n")
    f.write(f"## security_vulnerability\n{important_fields['security_vulnerability']}\n\n")
    f.write(f"## confidence_score\n{important_fields['confidence_score']}\n\n")
    f.write(f"## description\n{important_fields['description']}\n\n")
    f.write(f"## vuln_type\n{important_fields['vuln_type']}\n\n")
    f.write(f"## solutions\n{important_fields['solutions']}\n\n")
    f.write(f"## context\n{important_fields['context']}\n\n")

print(f"✅ Đã tạo file MD: {output_path}")
print("ĐÃ GHI THÀNH CÔNG NỘI DUNG FILE MD SAU:", important_fields)
print("==========THỰC HIỆN CONSENSUS AGENT==========")



SOURCE_PATH = "contracts/sample.sol"
RAG_PATH = "rag_output.json"
EXPLAINER_PATH = "multi_modal.md" # <-- Sửa ở đây

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

# --- Read content once globally
SOURCE_CONTENT = safe_read_text(SOURCE_PATH)
RAG_CONTENT = safe_read_json(RAG_PATH)
EXPLAINER_CONTENT = safe_read_text(EXPLAINER_PATH)


# ===========================
# --- Local LLM (Ollama)
# ===========================
llm_local = LLM(
    model="ollama/llama3:8b-instruct-q8_0",
    base_url="http://localhost:11434"
)

# ===========================
# --- THIẾT LẬP VECTORSTORE (MỚI) ---
# ===========================
print("Đang tải mô hình embeddings (local)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Đang tải ChromaDB từ disk...")
vector_store = Chroma(
    persist_directory="knowledge_base/chroma_db", 
    embedding_function=embeddings
)
# Tạo retriever, k=2 nghĩa là lấy 2 kết quả liên quan nhất
retriever = vector_store.as_retriever(search_kwargs={"k": 1})
print("✅ Vectorstore đã sẵn sàng.")


# ===========================
# --- Input Schema (Giữ nguyên)
# ===========================
class ConsensusInput(BaseModel):
    source_path: str = Field(SOURCE_PATH, description="Path to the source Solidity file")
    rag_path: str = Field(RAG_PATH, description="Path to the RAG agent output JSON")
    explainer_path: str = Field(EXPLAINER_PATH, description="Path to the Explainer agent output JSON")


# ===========================
# --- Consensus Tool Definition (Cập nhật)
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

    # ---------- Main Run (CẬP NHẬT) ----------
# ---------- Main Run (CẬP NHẬT THEO MỤC TIÊU MỚI) ----------
    def _run(self, source_path: str, rag_path: str, explainer_path: str) -> dict:
        # 1. Lấy nội dung global
        source_code = SOURCE_CONTENT
        rag_json = RAG_CONTENT
        explainer_text = EXPLAINER_CONTENT # Đây là text từ file .md

        # 2. (MỚI) Trích xuất Vuln Types
        rag_vuln_type = rag_json.get("Predict", "") or rag_json.get("vuln_type", "")
        
        # Dùng regex để tìm vuln_type trong file explainer .md
        explainer_vuln_type = ""
        match = re.search(r"## vuln_type\n(.*?)\n", explainer_text, re.DOTALL | re.IGNORECASE)
        if match:
            explainer_vuln_type = match.group(1).strip()

        print(f"🔍 Đã xác định Vuln Types: RAG='{rag_vuln_type}', Explainer='{explainer_vuln_type}'")

        # 3. (MỚI) Truy vấn Vectorstore TÁCH BIỆT
        
        # Hàm helper để truy vấn và gộp context
        def get_knowledge_context(query: str) -> str:
            if not query:
                return "No vulnerability type provided."
            try:
                print(f"📚 Truy vấn KB cho: '{query}'")
                docs = retriever.invoke(query) # retriever đã được định nghĩa ở global (k=1)
                # Gộp nội dung của các tài liệu tìm được
                return "\n\n---\n\n".join([doc.page_content for doc in docs])
            except Exception as e:
                print(f"⚠️ Lỗi truy vấn vectorstore: {e}")
                return f"Error retrieving knowledge: {e}"

        # Lấy context riêng cho RAG
        rag_knowledge_context = get_knowledge_context(rag_vuln_type)
        
        # Lấy context riêng cho Explainer
        explainer_knowledge_context = get_knowledge_context(explainer_vuln_type)

        print(f"📚 Đã truy xuất {len(rag_knowledge_context)} chars cho RAG.")
        print(f"📚 Đã truy xuất {len(explainer_knowledge_context)} chars cho Explainer.")

        # 4. Build prompt với kiến thức TÁCH BIỆT
        prompt = f"""
        You are an expert smart contract auditor. You will compare two audit reports 
        and the original source code, using the specific knowledge context provided for each report.

        --- SOURCE CODE ---
        {source_code}
        --- END SOURCE CODE ---


        --- RAG AGENT OUTPUT ---
        {json.dumps(rag_json, ensure_ascii=False, indent=2)}
        
        --- RAG KNOWLEDGE CONTEXT (Kiến thức cho RAG) ---
        {rag_knowledge_context}
        --- END RAG KNOWLEDGE ---


        --- EXPLAINER AGENT OUTPUT (Nội dung file .md) ---
        {explainer_text}

        --- EXPLAINER KNOWLEDGE CONTEXT (Kiến thức cho Explainer) ---
        {explainer_knowledge_context}
        --- END EXPLAINER KNOWLEDGE ---


        TASK:
        1) Evaluate the RAG AGENT OUTPUT. Is it accurate based on the SOURCE CODE and the RAG KNOWLEDGE CONTEXT?
        2) Evaluate the EXPLAINER AGENT OUTPUT. Is it accurate based on the SOURCE CODE and the EXPLAINER KNOWLEDGE CONTEXT?
        3) Decide which audit is MORE ACCURATE: "RAG", "Explainer", or "Merged" (if both have complementary, valid findings).
        4) Provide concise reasoning for your decision.
        5) Output ONLY valid JSON with this exact schema (no surrounding text):

        {{
            "decision": "RAG" | "Explainer" | "Merged",
            "reasoning": "concise reasons why you chose this decision, comparing both agents.",
            "final_vulnerability_summary": "one-sentence summary of the confirmed vulnerability",
            "confidence": float
        }}
        """

        # ... (Phần còn lại của hàm _run, Call LLM, Parse JSON... giữ nguyên) ...
        # Call LLM
        try:
            llm_text = self._call_llm(prompt)
        except Exception as e:
            fallback = {
                "decision": "Merged",
                "reasoning": f"LLM invocation failed: {e}",
                "final_vulnerability_summary": rag_json.get("vuln_type", "") or explainer_vuln_type,
                "confidence": 0.0
            }
            with open("consensus_output.json", "w", encoding="utf-8") as f:
                json.dump(fallback, f, ensure_ascii=False, indent=2)
            return fallback

        # Parse JSON output
        try:
            parsed = json.loads(llm_text)
        except Exception:
            m = re.search(r"\{.*\}", llm_text, re.DOTALL)
            parsed = json.loads(m.group(0)) if m else None

        if not parsed:
            result = {
                "decision": "Merged",
                "reasoning": llm_text.strip(),
                "final_vulnerability_summary": rag_json.get("Predict", "") or explainer_vuln_type,
                "confidence": 0.5
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


# ===========================
# --- Build Consensus Agent & Task (Giữ nguyên)
# ===========================
consensus_tool = ConsensusTool(llm=llm_local)

consensus_agent = Agent(
    role="Consensus Agent",
    goal="Compare RAG and Explainer audit outputs with source code to produce a unified, authoritative audit decision.",
    backstory="An expert smart contract auditor combining multiple model outputs to achieve the most reliable result.",
    tools=[consensus_tool],
    verbose=True,
    llm=llm_local,
)

consensus_task = Task(
    # Mô tả này vẫn chính xác, tool tự xử lý việc lấy data
    description="Use the ConsensusTool to analyze and merge the audit results. This tool already has all the necessary data.",
    expected_output="JSON containing decision, reasoning, final_vulnerability_summary, confidence.",
    agent=consensus_agent,
)


# ===========================
# --- Optional: run standalone (Giữ nguyên)
# ===========================
if __name__ == "__main__":
    crew = Crew(
        agents=[consensus_agent],
        tasks=[consensus_task],
        process=Process.sequential,
        verbose=True,
    )
    result = crew.kickoff()
    print("\n✅ Final Consensus Result:")
    print(result)