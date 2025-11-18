# -*- coding: utf-8 -*-
"""
invoke.py - Run Sequential Crew Pipeline (Gemini Functional Semantic → RAG → Embedding)
"""

import os, json
from rag_agent import rag_agent, FunctionalSemantic, gemini_model # Xóa rag_task
from embedding_agent import embedding_agent # Xóa embedding_task
from crewai import Crew, Process, Task # Import Task

# ============================================================
# 1️⃣ Đọc code Solidity từ file sample.sol
# ============================================================

sample_path = os.path.join(os.getcwd(), "contracts", "sample.sol")

if not os.path.exists(sample_path):
    # Thử tìm ở thư mục gốc xem sao (fallback)
    root_path = os.path.join(os.getcwd(), "sample.sol")
    if os.path.exists(root_path):
        sample_path = root_path
        print(f"⚠️ Không tìm thấy trong contracts/, nhưng đã tìm thấy tại: {sample_path}")
    else:
        raise FileNotFoundError(f"⚠️ File sample.sol not found! Vui lòng kiểm tra tại: {sample_path}")

with open(sample_path, "r", encoding="utf-8") as f:
    code = f.read()

print("✅ Đã đọc code từ sample.sol")

# ============================================================
# 2️⃣ Phân tích Functional Semantic bằng Gemini
# ============================================================

# ... (Giữ nguyên) ...
print("🚀 Đang sinh functional_semantic bằng Gemini...")
fs = FunctionalSemantic(gemini_model)
functional_semantic = fs.analyze(code)

# ============================================================
# 3️⃣ Lưu input.jsonjson
# ============================================================
input_data = {
    "code": code,
    "functional_semantic": functional_semantic
}

input_path = os.path.join(os.getcwd(), "input.json")

# Ghi file để RAG Agent có thể tự đọc (Safe Read)
with open(input_path, "w", encoding="utf-8") as f:
    json.dump(input_data, f, indent=2, ensure_ascii=False)

print(f"✅ Đã lưu input.json tại: {input_path}")

# ============================================================
# 4️⃣ Khởi tạo Crew tuần tự (RAG → Embedding)
# ============================================================

# ✅ THAY ĐỔI 1: Định nghĩa lại RAG Task
rag_task = Task(
    name="rag_task",
    description="Analyze Solidity code and functional semantics and produce structured vulnerability report.",
    expected_output="A JSON object with vulnerability type, description, recommendation, and context.",
    agent=rag_agent,

)

# ✅ THAY ĐỔI 2: Định nghĩa lại Embedding Task
embedding_task = Task(
    name="embedding_task",
    description="Generate embeddings for CFG, Code, and Functional Semantics from rag_output.json.",
    expected_output="parser_output.json containing embeddings for the three data types.",
    agent=embedding_agent,
)

crew = Crew(
    agents=[rag_agent, embedding_agent],
    tasks=[rag_task, embedding_task],
    process=Process.sequential
)

# ============================================================
# 5️⃣ Chạy pipeline
# ============================================================

if __name__ == "__main__":
    print("\n🚀 Starting sequential pipeline: Gemini → RAG → Embedding\n")
    result = crew.kickoff()

    print("\n✅ FINAL PIPELINE RESULT:")
    try:
        # CrewOutput thường có thuộc tính .output (dict)
        print(json.dumps(result.output, indent=2, ensure_ascii=False))
    except Exception:
        print("⚠️ Không thể serialize CrewOutput — in dạng text:")
        print(str(result))
