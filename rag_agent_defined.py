# -*- coding: utf-8 -*-
"""
rag_agent.py
Purpose:
    - Analyze Solidity code semantics using Gemini
    - Query Neo4j knowledge graph for vulnerability context
    - Use Unsloth model for final reasoning
"""
import os, re, json
import torch
from typing import Type, Optional
from transformers import TextIteratorStreamer
from pydantic import BaseModel, Field
from typing import Union, Dict, Any
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from threading import Thread
# === Import các đối tượng khởi tạo từ file setup ===
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
max_seq_length = 2048

# ============================================================
# Helper: Format Context String
# ============================================================

def format_context_to_string(context_dict, score):
    """
    Chuẩn hóa dữ liệu context từ Neo4j, loại bỏ code dư thừa để LLM tập trung.
    """
    ret_type = context_dict.get('type', 'N/A')
    ret_vuln = str(context_dict.get('vulnerability', 'N/A')).replace('\\n', '\n')

    ret_sol = context_dict.get('solution')
    if ret_sol:
        ret_sol = str(ret_sol).replace('\\n', '\n')

    ret_fcode = context_dict.get('fixed_code')
    if ret_fcode:
        ret_fcode = str(ret_fcode).replace('\\n', '\n')

    context_str = f"--- BEGIN CONTEXT FROM KNOWLEDGE GRAPH (Similarity Score: {score:.4f}) ---\n"
    context_str += f"**Retrieved Type:** {ret_type}\n\n"
    context_str += f"**Retrieved Vulnerability:**\n{ret_vuln}\n\n"

    if ret_sol:
        context_str += f"**Retrieved Solution:**\n{ret_sol}\n\n"
    if ret_fcode:
        context_str += f"**Retrieved Fixed Code Example:**\n{ret_fcode}\n\n"

    context_str += "--- END CONTEXT FROM KNOWLEDGE GRAPH ---"
    return context_str


# ============================================================
# Main RAG Retrieve Tool Function
# ============================================================

def rag_retrieve_tool(code: str, functional_semantic: str) -> dict:
    print("🚀 [RAGRetrieveTool] Received functional semantic:")
    print(functional_semantic)

    print("🚀 [RAGRetrieveTool] Received Solidity code snippet:")
    print(code)
    """
    Thực hiện pipeline RAG:
    1. Semantic search trong Neo4j
    2. Format context
    3. Tạo prompt
    4. Gọi model sinh phản hồi
    """

    output_dict = {}
    output_dict["code"] = code
    output_dict["functional_semantic"] = functional_semantic

    # --- Step 1: Vector Search ---
    print("\n🔍 Thực hiện vector search trên Neo4j...")
    if not vcode_vector_store:
        print("❌ Neo4j vector store chưa được khởi tạo.")
        return {"error": "Neo4j not initialized"}

    try:
        results = vcode_vector_store.similarity_search_with_score(code, k=1)
    except Exception as e:
        print(f"❌ Lỗi khi truy vấn Neo4j: {e}")
        return {"error": f"Neo4j query failed: {e}"}

    if not results:
        print("❌ Không tìm thấy context tương đồng trong database.")
        return {"error": "No matching context found."}

    doc, score = results[0]
    retrieved_context_dict = doc.metadata
    print(f"🎯 Đã tìm thấy context. Score = {score:.4f}")

    # --- Step 2: Format Context ---
    context_str = format_context_to_string(retrieved_context_dict, score)

    # --- Step 3: Build Prompt ---
    prompt = f"""
You are a meticulous smart contract security analyst.
Your task is to compare the [Solidity Code (To be analyzed)] against the [Retrieved Context].

---
[Solidity Code (To be analyzed)]:
```solidity
{code}
[Retrieved Context]:
(This context describes a vulnerability found in a similar piece of code)
{context_str}

YOUR INSTRUCTIONS:

Analyze the [Retrieved Context] – what specific vulnerability does it describe?

Examine the [Solidity Code (To be analyzed)] carefully.

Decide if the code ALSO contains the same vulnerability.

Output format:

Vulnerability Name: <Name or Non-vulnerable>
Reasoning: <Explain why or why not vulnerable>
Recommendation: <Suggested fix>

Response:

Vulnerability Name:"""

    output_dict["llm_prompt"] = prompt
    # --- Step 4: Tokenization check ---
    try:
        inputs = unsloth_tok([prompt], return_tensors="pt")
    except Exception as e:
        print(f"❌ Lỗi tokenizing: {e}")
        return {"error": f"Tokenizing failed: {e}"}

    token_len = inputs["input_ids"].shape[1]
    print("-" * 50)
    print(f"ℹ️  Prompt Token Count: {token_len} / Max: {max_seq_length}")
    if token_len > max_seq_length:
        print("⚠️ Prompt vượt quá context window, có thể bị cắt.")
        output_dict["warning_llm"] = "Prompt truncated."
    print("-" * 50)

    # --- Step 5: Run model generate ---
    llm_output = ""
    try:
        try:
            inputs = {k: v.cuda() for k, v in inputs.items()}
            unsloth_model.cuda()
        except Exception:
            print("⚠️ Running on CPU mode (no CUDA).")

        streamer = TextIteratorStreamer(unsloth_tok, skip_prompt=True, skip_special_tokens=True)
        thread = Thread(
            target=unsloth_model.generate,
            kwargs=dict(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                streamer=streamer,
                repetition_penalty=1.15
            )
        )
        thread.start()

        result = ""
        print("🤖 LLM Response:\n**Vulnerability Status:**", end="", flush=True)
        for token in streamer:
            print(token, end="", flush=True)
            result += token
        thread.join()

        llm_output = "**Vulnerability Status:**" + result
        print("\n\n✅ Model output generated successfully.")
    except Exception as e:
        llm_output = f"Error generating response: {e}"
        print(f"\n❌ Lỗi khi generate: {e}")

    output_dict["raw_llm_output"] = llm_output
    output_dict["context_similarity_score"] = score
    output_dict["context_metadata"] = retrieved_context_dict
    output_dict["functional_semantic"] = functional_semantic

    output_path = "rag_agent_output.json"
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Output saved successfully → {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"❌ Failed to save JSON output: {e}")

    return output_dict
# ============================================================
# Define RAG Agent & Task
# ============================================================
from pydantic import BaseModel, Field
from typing import Type
from crewai.tools import BaseTool
import json

class RAGRetrieveToolInput(BaseModel):
    path: str = Field("input.json", description="Đường dẫn file JSON chứa code và functional semantic")

# Tool chuẩn
class RAGRetrieveTool(BaseTool):
    name: str = "RAG Agent Retrieval"
    description: str = (
        "Đọc JSON từ path, lấy 'code' và 'functional_semantic', "
        "chạy phân tích RAG và trả về JSON kết quả."
    )
    args_schema: Type[BaseModel] = RAGRetrieveToolInput

    def _run(self, path: str) -> dict:
        if not os.path.exists(path):
            return {"error": f"File not found: {path}"}

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        code = data.get("code", "")
        functional_semantic = data.get("functional_semantic", "")
        result = rag_retrieve_tool(code=code, functional_semantic=functional_semantic)

        return f"Thought: I now know the final answer\nFinal Answer:\n{result}"

rag_tool = RAGRetrieveTool()

rag_agent = Agent(
    role="RAG Agent",
    goal="Analyze the given Solidity code for security vulnerabilities. Use semantic analysis to extract the functional intent of the code and perform similarity search for known vulnerabilities based on the extracted semantics. Return a detailed vulnerability type and description.",
    backstory=(
        "You are an expert in Ethereum, Solidity, and decentralized finance (DeFi) security. "
        "Your job is to deeply analyze Solidity code, identify vulnerabilities, and provide a precise understanding of potential security issues. "
        "You use a combination of functional analysis and vulnerability databases to ensure a comprehensive analysis of any given contract code."
    ),
    tools=[rag_tool],   # Tool to retrieve vulnerability analysis based on semantic search
    allow_delegation=False,
    verbose=True,
    llm=llm_local
)

rag_task = Task(
    name="rag_task",
    description=(
        "Analyze the following Solidity code and Functional_semantic in input.json"
        "Return detailed vulnerability information including the type of vulnerability, "
        "a detailed explanation, and suggested mitigations."
        "IMPORTANT:\n"
        "- After receiving the tool output, DO NOT call any tool again.\n"
        "- The tool output IS the final answer.\n"
        "- Do not generate additional explanation.\n"
        "- Do not attempt to continue analyzing after tool output.\n"
    ),
    expected_output="The final vulnerability report produced by the tool.",
    agent=rag_agent,
    input={"path": "input.json"}
)


log("RAG Agent & Task initialized ✅")
