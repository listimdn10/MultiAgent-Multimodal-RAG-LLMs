# -*- coding: utf-8 -*-
"""
rag_agent.py
Purpose:
    - Analyze Solidity code semantics using Gemini
    - Query Neo4j knowledge graph for vulnerability context
    - Use Unsloth model for final reasoning
"""

import os, re, json, torch
from unsloth import FastLanguageModel
import google.generativeai as genai
from langchain_community.vectorstores import Neo4jVector
from neo4j import GraphDatabase
from transformers import TextIteratorStreamer
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
# ============================================================
# Utilities
# ============================================================

def log(msg, symbol="✅"):
    print(f"{symbol} {msg}")

# ============================================================
# API Keys and Models
# ============================================================

try:
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")
    log("Gemini model initialized ✅")
except Exception as e:
    gemini_model = None
    log(f"⚠️ Gemini not available: {e}", "⚠️")

try:
    unsloth_model, unsloth_tok = FastLanguageModel.from_pretrained(
        model_name="Nhudang/LLama-3B-Solidity",
        max_seq_length=1024,
        load_in_4bit=True
    )
    log("Unsloth model loaded ✅")
except Exception as e:
    unsloth_model, unsloth_tok = None, None
    log(f"⚠️ Unsloth model failed: {e}", "⚠️")

# ============================================================
# Functional Semantic Analyzer
# ============================================================

class FunctionalSemantic:
    """
    Analyze code functional semantics using Gemini only.
    """
    def __init__(self, gemini_model):
        self.gemini = gemini_model
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def _make_prompt(self, code: str) -> str:
        return f"""
        What is the purpose of the following code snippet?
        Please summarize the answer in one sentence with the following format:
        “Abstract purpose:” (one sentence).
        Then summarize the functions of the code in a list format without explanation:
        “Detail Behaviors: 1. 2. 3...”
        
        Here is the code:
        {code}
        """

    def analyze(self, code: str):
        """Generate a functional semantic summary using Gemini API."""
        prompt = self._make_prompt(code)
        if not self.gemini:
            raise ValueError("Gemini model not provided in FunctionalSemantic.")

        try:
            response = self.gemini.generate_content(prompt)
            print("=== [Gemini Output] ===")
            print(response.text)
            return response.text
        except Exception as e:
            print(f"❌ Gemini API error: {e}")
            return "Error generating functional semantic."

# ============================================================
# Neo4j Configuration
# ============================================================

NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")
NEO4J_VECTOR_INDEX = os.environ.get("NEO4J_VECTOR_INDEX")

if not all((NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_VECTOR_INDEX)):
    raise RuntimeError("Set the NEO4J_* environment variables before using the legacy RAG agent.")

fs = FunctionalSemantic(gemini_model)
embeddings = fs.embeddings

behaviors_index_rag_agent = Neo4jVector.from_existing_graph(
    embedding=embeddings,
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
    index_name=NEO4J_VECTOR_INDEX,
    node_label="Behavior",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)
driver_rag_agent = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
log("Connected to Neo4j Aura ✅")

# ============================================================
# RAGRetrieveTool
# ============================================================

def safe_read_json(path):
    """Đọc file JSON một cách an toàn."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Không thể đọc file JSON {path}: {e}")
        return {} # Trả về dict rỗng nếu lỗi


INPUT_FILE_PATH = "input.json"

# ✅ THAY ĐỔI : Dùng Input rỗng (giống DummyInput)
class RAGRetrieveToolInput(BaseModel):
    """Không nhận tham số, tool sẽ tự đọc file."""
    pass

class RAGRetrieveTool(BaseTool):
    name: str = "RAG Agent Retrieval"
    description: str = "Receive code and semantic data, query Neo4j, and return vulnerability information"
    args_schema: Type[BaseModel] = RAGRetrieveToolInput # Sửa schema

    # ✅ THAY ĐỔI 4: Sửa chữ ký hàm _run, không nhận tham số
    def _run(self):
        
        # ✅ THAY ĐỔI 5: Đọc file bên trong _run
        print(f"📂 [RAGRetrieveTool] Đang đọc file input cố định: {INPUT_FILE_PATH}")
        data = safe_read_json(INPUT_FILE_PATH)

        code = data.get("code", "")
        functional_semantic = data.get("functional_semantic", "")
        
        # In ra nội dung (như bạn yêu cầu)
        print("🚀 [RAGRetrieveTool] Received functional semantic:")
        print(functional_semantic)
        print("🚀 [RAGRetrieveTool] Received Solidity code snippet:")
        print(code)

        # Kiểm tra xem có đọc được file không
        if not code or not functional_semantic:
            error_msg = f"❌ [RAGRetrieveTool] File {INPUT_FILE_PATH} rỗng hoặc thiếu 'code'/'functional_semantic'."
            print(error_msg)
            # Return một lỗi để Agent biết
            return {"error": error_msg}
        
        output_path = "rag_output.json"

        behaviors_index = behaviors_index_rag_agent
        neo4j_driver = driver_rag_agent

        def clean_text(raw_text):
            clean = raw_text.strip()
            if clean.lower().startswith("text:"):
                clean = clean[5:].strip()
            return clean

        def get_related_vulnerabilities(behavior_text):
            query = """
            MATCH (b:Behavior)-[:EXPOSES]->(v:Vulnerability)
            WHERE b.text = $behavior_text
            RETURN v.text AS vuln_text
            """
            with neo4j_driver.session(database=NEO4J_DATABASE) as session:
                results = session.execute_read(
                    lambda tx: tx.run(query, behavior_text=behavior_text).data()
                )
            return results

        def build_train_prompt(context, code):
            return f"""
                You are a smart contract security assistant with deep expertise in Ethereum, Solidity, and DeFi security.
                You are given contextual information about a potential vulnerability:
                {context}

                And the following Solidity code:
                {code}

                Please follow these steps:
                **Step 1**: Summarize what the code is intended to do.
                **Step 2**: Identify and explain any suspicious or potentially unsafe logic patterns.
                **Step 3**: Based on your analysis, determine the type of vulnerability, describe the issue in detail, and recommend mitigation strategies.
                Provide your output in the following structured format:
                - **Type of Vulnerability**: <short name>
                - **Description**: <detailed explanation>
                - **Recommendation**: <fix or mitigation suggestion>
                ### Response:
                <think>
            """

        def parse_llm_output(output: str):
            """Tách 3 trường vuln_type, description, recommendation linh hoạt"""
            patterns = {
                "vuln_type": [
                    r"\*\*Type of Vulnerability\*\*:\s*(.+)",
                    r"[-\s]*type\s*:\s*(.+)",
                    r"Type\s*of\s*Vulnerability[:\- ]*(.+)"
                ],
                "description": [
                    r"\*\*Description\*\*:\s*(.+)",
                    r"[-\s]*description\s*:\s*(.+)",
                    r"Step\s*3[:\- ]*(.+)"  # fallback nếu có step 3
                ],
                "recommendation": [
                    r"\*\*Recommendation\*\*:\s*(.+)",
                    r"[-\s]*recommendation\s*:\s*(.+)",
                    r"Mitigation[:\- ]*(.+)"
                ]
            }

            def find_match(text, pats):
                for p in pats:
                    m = re.search(p, text, re.IGNORECASE)
                    if m:
                        return m.group(1).strip()
                return ""

            vuln_type = find_match(output, patterns["vuln_type"])
            description = find_match(output, patterns["description"])
            recommendation = find_match(output, patterns["recommendation"])

            if not description:
                # lấy vài dòng đầu tiên trong phần analysis
                desc_match = re.search(r"### Analysis(.+?)(\n\*\*Recommendation|\Z)", output, re.DOTALL | re.IGNORECASE)
                if desc_match:
                    description = desc_match.group(1).strip()

            return vuln_type, description, recommendation


        # 2. Vector search (behavior semantic)
        docs_with_score = behaviors_index.similarity_search_with_score(functional_semantic, k=1)
        contexts = []
        for doc, score in docs_with_score:
            behavior_text = clean_text(doc.page_content)
            related_vulns = get_related_vulnerabilities(behavior_text)
            for record in related_vulns:
                contexts.append(record['vuln_text'])

        # 3. Build prompt for the open-source LLM
        context_str = "\n".join(contexts)
        prompt = build_train_prompt(context_str, code)
        print("\n[INFO] Prompt to LLM:\n", prompt)

        # 4. Generate response using open-source HuggingFace model
        try:
            inputs = unsloth_tok([prompt], return_tensors="pt")
            try:
                inputs = {k: v.cuda() for k, v in inputs.items()}
                unsloth_model.cuda()
            except Exception as _:
                pass  # If no CUDA, fallback to CPU

            from transformers import TextIteratorStreamer
            streamer = TextIteratorStreamer(unsloth_tok, skip_prompt=True)
            _ = unsloth_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                streamer=streamer
            )

            result = ""
            for token in streamer:
                result += token
            llm_output = result

        except Exception as e:
            print(f"❌ Lỗi khi generate với transformers: {e}")
            llm_output = ""

        # Parse output
        vuln_type, description, recommendation = parse_llm_output(llm_output)
        output = {
            "type": "vuln_analysis_result",
            "vuln_type": vuln_type,
            "description": description,
            "recommendation": recommendation,
            "context": context_str,
            "raw_llm_output": llm_output,
            "code": code,
            "functional_semantic": functional_semantic
        }

# =======================================================
        # ✅ SỬA LỖI Ở ĐÂY
        # =======================================================
        
        output_path = "rag_output.json"

        try:
            dir_path = os.path.dirname(output_path)

            # Chỉ tạo folder nếu dir_path không phải chuỗi rỗng
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            print(f"✅ Đã ghi output vào {output_path}")

        except Exception as e:
            print(f"❌ Lỗi khi ghi file rag_output.json: {e}")
            
        vuln_name = output.get("vuln_type", "Unknown")
        return f"TASK COMPLETED. Vulnerability analysis saved to {output_path}. Identified vulnerability: {vuln_name}. You can stop now."

rag_tool = RAGRetrieveTool()
