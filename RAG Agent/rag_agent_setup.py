# -*- coding: utf-8 -*-
"""
rag_agent_setup.py
Purpose:
    - Initialize Gemini, Unsloth, Ollama
    - Setup Neo4j connection and vector index
    - Provide shared embedding + driver objects
"""

import os
import google.generativeai as genai
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from unsloth import FastLanguageModel
from crewai import LLM
from neo4j import GraphDatabase
import subprocess, time, requests
# ============================================================
# Helper
# ============================================================

def log(msg, symbol="✅"):
    print(f"{symbol} {msg}")

# ============================================================
# Gemini Init
# ============================================================

try:
    GEMINI_API_KEY = ''
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")
    log("Gemini model initialized ✅")
except Exception as e:
    gemini_model = None
    log(f"⚠️ Gemini init failed: {e}", "⚠️")

# ============================================================
# Unsloth Init
# ============================================================

try:
    unsloth_model, unsloth_tok = FastLanguageModel.from_pretrained(
        model_name = "Nhudang/DeepSeek-R1-Distill-Llama-8B",
        max_seq_length = 2048,
        load_in_4bit = True
    )
    log("Unsloth model loaded ✅")
except Exception as e:
    unsloth_model, unsloth_tok = None, None
    log(f"⚠️ Unsloth model failed: {e}", "⚠️")

# ============================================================
# Ollama Init
# ============================================================
def ensure_ollama():
    try:
        # kiểm tra nếu server đã chạy
        requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        print("✅ Ollama already running.")
    except:
        print("🚀 Starting Ollama server...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)

# Gọi hàm trước khi init LLM
ensure_ollama()

try:
    llm_local = LLM(model="ollama/llama3:8b-instruct-q8_0", base_url="http://127.0.0.1:11434", timeout=180)
    log("✅ Ollama connected")
except Exception as e:
    llm_local = None
    log(f"⚠️ Ollama not available: {e}")

# ============================================================
# Neo4j Config & Connection
# ============================================================

NEO4J_URI = ""
NEO4J_USERNAME = ""
NEO4J_PASSWORD = ""
NEO4J_DATABASE = ""

retrieval_query = """
WITH node, score
    MATCH (node)-[:CAUSES]->(v:Vulnerability)
    MATCH (v)-[:IS_OF_TYPE]->(vt:VulnType)
    OPTIONAL MATCH (node)-[:BELONGS_TO]->(f:Functionality)
    OPTIONAL MATCH (v)-[:FIXED_BY]->(fc:FCode)
    OPTIONAL MATCH (v)-[:HAS_SOLUTION]->(s:Solution)
RETURN {
    vulnerable_code: node.content,
    vulnerability: v.text,
    type: vt.name,
    functionality: f.text,
    fixed_code: fc.content,
    solution: s.text
} AS metadata,
node.content AS text,
score
"""

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vcode_vector_store = Neo4jVector(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
    embedding=embeddings,
    index_name="vcode_index",
    text_node_property="content",
    retrieval_query=retrieval_query
)

driver_rag_agent = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
log("Connected to Neo4j Aura ✅")

# ============================================================
# Exports
# ============================================================

__all__ = [
    "gemini_model",
    "unsloth_model",
    "unsloth_tok",
    "llm_local",
    "driver_rag_agent",
    "vcode_vector_store",
    "embeddings",
    "log"
]
