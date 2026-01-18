# -*- coding: utf-8 -*-
"""
single_agent_baseline.py - Monolithic Single-Agent Approach
Purpose: Demonstrate a baseline where ONE agent tries to do everything
         (Semantic Analysis + RAG + Embedding + Fusion + Explanation + Consensus)
         
Compare this with multi-agent CrewAI architecture to show benefits of specialization.
"""

import os, json, time
import torch
import numpy as np
import google.generativeai as genai
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings  # ✅ Dùng langchain_community thay vì langchain_huggingface
from langchain_community.vectorstores import Neo4jVector
from neo4j import GraphDatabase
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from tools.fusion_model import EarlyFusionModel

# ============================================================
# Configuration
# ============================================================

SAMPLE_PATH = "contracts/sample.sol"
OUTPUT_PATH = "single_agent_output.json"

# API Keys
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyDJLue1nRDsxrI_Nc2le3Si2wpE-U8wBzQ')
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-flash-latest")

# LLM for agent
llm_local = LLM(model="ollama/llama3:8b-instruct-q8_0", base_url="http://localhost:11434")

# Embeddings
embeddings_neo4j = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # 384 dim cho Neo4j (giống rag_agent.py)
embeddings_semantic = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")  # 768 dim cho functional semantic fusion
embeddings_chroma = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # 384 dim cho ChromaDB (giống consensus_agent.py)

# Load GraphCodeBERT for code embeddings (768 dim)
try:
    code_tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    code_model = AutoModel.from_pretrained("microsoft/graphcodebert-base").to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    code_model.eval()
    print("✅ Loaded GraphCodeBERT for code embeddings (768 dim)")
except Exception as e:
    code_tokenizer = None
    code_model = None
    print(f"⚠️ Could not load GraphCodeBERT: {e}")

# Fallback sentence model
sentence_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')  # 768 dim

# Vector Store - ChromaDB (dùng all-MiniLM-L6-v2 như consensus_agent.py)
vector_store = Chroma(
    persist_directory="knowledge_base/chroma_db", 
    embedding_function=embeddings_chroma
)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# Neo4j Configuration (giống y hệt rag_agent.py)
NEO4J_URI = "neo4j+s://2fb0c4a7.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "yIn7gf5LZOwTSAabAaQiOY0sgTdJK83iOkeaVf0nhBc"
AURA_INSTANCENAME = "Instance01"

try:
    behaviors_index = Neo4jVector.from_existing_graph(
        embedding=embeddings_neo4j,  # Dùng all-MiniLM-L6-v2 384 dim (giống rag_agent.py)
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name=AURA_INSTANCENAME,
        node_label="Behavior",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    print("✅ Connected to Neo4j Aura")
except Exception as e:
    behaviors_index = None
    neo4j_driver = None
    print(f"⚠️ Neo4j connection failed: {e}")

# Fusion Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fusion_model = EarlyFusionModel(
    d_sc=768,      # ✅ dimension of source code embedding (all-mpnet-base-v2)
    d_fs=768,      # ✅ dimension of functional semantic embedding (all-mpnet-base-v2)
    d_cfg=64,      # ✅ dimension of CFG embedding (adjusted to match checkpoint 1600 total)
    n_classes=4    # ✅ 4 loại vulnerability
).to(device)

# Load trained weights
checkpoint_path = "tools/early_fusion.pth"  # ✅ SỬA: Dùng đúng checkpoint như fusion_agent.py
if os.path.exists(checkpoint_path):
    fusion_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    fusion_model.eval()
    print(f"✅ Loaded fusion model from {checkpoint_path}")
else:
    print(f"⚠️ Fusion model checkpoint not found: {checkpoint_path}")

# Load label encoder to get correct vulnerability classes
import joblib
label_encoder_path = "tools/early_fusion_label_encoder.pkl"
try:
    label_encoder = joblib.load(label_encoder_path)
    LABELS = {i: label for i, label in enumerate(label_encoder.classes_)}
    print(f"✅ Loaded {len(LABELS)} vulnerability classes: {list(LABELS.values())}")
except Exception as e:
    print(f"⚠️ Could not load label encoder, using fallback labels: {e}")
    LABELS = {0: "arithmetic", 1: "block number dependency (BN)", 2: "external_call_vulnerability", 3: "time_manipulation"}

# ============================================================
# Helper Functions
# ============================================================

def safe_read_text(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"⚠️ Could not read {path}: {e}")
        return ""

def generate_embeddings(code, functional_semantic, cfg=""):
    """Generate embeddings for code, semantic, and CFG (giống embedding_agent.py)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Code embedding: GraphCodeBERT với sliding window (768 dim) - giống embedding_agent.py
    if code_model and code_tokenizer:
        try:
            # Sliding window approach
            tokens = code_tokenizer(code, add_special_tokens=True, return_tensors='pt', truncation=False, padding=False)
            input_ids = tokens['input_ids'][0]
            
            if len(input_ids) <= 512:
                # Short code - direct encoding
                input_ids = input_ids.unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = code_model(input_ids)
                    code_emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            else:
                # Long code - sliding window
                all_chunk_embeddings = []
                window_size = 512
                stride = 256
                
                for i in range(0, len(input_ids), stride):
                    chunk_ids = input_ids[i:i+512]
                    if len(chunk_ids) < 10:
                        continue
                    chunk_ids = chunk_ids.unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = code_model(chunk_ids)
                        chunk_emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                        all_chunk_embeddings.append(chunk_emb)
                
                if len(all_chunk_embeddings) > 0:
                    code_emb = np.mean(all_chunk_embeddings, axis=0)
                else:
                    code_emb = np.zeros(768)
        except Exception as e:
            print(f"⚠️ GraphCodeBERT failed: {e}, using fallback")
            code_emb = sentence_model.encode(code[:1000], convert_to_numpy=True)
    else:
        # Fallback to sentence transformer
        code_emb = sentence_model.encode(code[:1000], convert_to_numpy=True)
    
    # 2. Functional semantic: HuggingFaceEmbeddings all-mpnet-base-v2 (768 dim) - giống embedding_agent.py
    semantic_emb = embeddings_semantic.embed_query(functional_semantic)
    semantic_emb = np.array(semantic_emb)
    
    # 3. CFG: Simplified (64 dim) - vì không extract CFG graph thật
    cfg_base = sentence_model.encode(cfg if cfg else "No CFG available", convert_to_numpy=True)
    cfg_emb = cfg_base[:64]  # Truncate to 64 dim (768+768+64=1600)
    
    return code_emb, semantic_emb, cfg_emb

# ============================================================
# Monolithic Tool (Everything in one place)
# ============================================================

class MonolithicInput(BaseModel):
    """No input needed - reads from global state"""
    pass

class MonolithicAuditTool(BaseTool):
    name: str = "Monolithic Smart Contract Auditor"
    description: str = "Single tool that performs ALL audit tasks: semantic analysis, RAG, embedding, fusion, explanation, and consensus"
    args_schema: Type[BaseModel] = MonolithicInput

    def _run(self) -> dict:
        start_time = time.time()
        result = {
            "architecture": "single_agent_monolithic",
            "stages": {}
        }
        
        # ===== STAGE 1: Read Code =====
        print("\n📖 [STAGE 1/7] Reading Solidity Code...")
        code = safe_read_text(SAMPLE_PATH)
        if not code:
            return {"error": "Failed to read source code"}
        result["stages"]["code_reading"] = {"status": "completed", "lines": len(code.split('\n'))}
        
        # ===== STAGE 2: Functional Semantic Analysis (Gemini) =====
        print("\n🧠 [STAGE 2/7] Generating Functional Semantic Analysis...")
        try:
            prompt = f"""
What is the purpose of the following code snippet?
Please summarize the answer in one sentence with the following format:
"Abstract purpose:" (one sentence).
Then summarize the functions of the code in a list format without explanation:
"Detail Behaviors: 1. 2. 3..."

Here is the code:
{code}
"""
            response = gemini_model.generate_content(prompt)
            functional_semantic = response.text
            result["stages"]["semantic_analysis"] = {
                "status": "completed",
                "model": "Gemini Flash",
                "output_length": len(functional_semantic)
            }
        except Exception as e:
            functional_semantic = "Error in semantic analysis"
            result["stages"]["semantic_analysis"] = {"status": "failed", "error": str(e)}
        
        # ===== STAGE 3: RAG Retrieval (Neo4j only - giống rag_agent.py) =====
        print("\n🔍 [STAGE 3/7] Performing RAG Retrieval...")
        rag_context = ""
        neo4j_vulnerabilities = []
        
        try:
            # Neo4j retrieval - search behaviors và lấy related vulnerabilities như rag_agent.py
            if neo4j_driver and behaviors_index:
                query_text = functional_semantic[:500] if functional_semantic else "Smart contract vulnerability analysis"
                
                # Search behaviors using Neo4j vector index (giống rag_agent.py line 267)
                behavior_docs = behaviors_index.similarity_search_with_score(query_text, k=1)
                
                contexts = []
                for doc, score in behavior_docs:
                    behavior_text = doc.page_content.strip()
                    # Clean text như rag_agent.py
                    if behavior_text.lower().startswith("text:"):
                        behavior_text = behavior_text[5:].strip()
                    
                    # Get related vulnerabilities for this behavior (giống rag_agent.py line 192-199)
                    vuln_query = """
                    MATCH (b:Behavior)-[:EXPOSES]->(v:Vulnerability)
                    WHERE b.text = $behavior_text
                    RETURN v.text AS vuln_text
                    """
                    with neo4j_driver.session() as session:
                        results = session.run(vuln_query, behavior_text=behavior_text).data()
                        for record in results:
                            contexts.append(record['vuln_text'])
                            neo4j_vulnerabilities.append(record['vuln_text'])
                
                rag_context = "\n".join(contexts)
            else:
                rag_context = "Neo4j not available"
            
            result["stages"]["rag_retrieval"] = {
                "status": "completed",
                "neo4j_behaviors_found": len(behavior_docs) if neo4j_driver and behaviors_index else 0,
                "neo4j_vulnerabilities": len(neo4j_vulnerabilities),
                "context_length": len(rag_context)
            }
        except Exception as e:
            rag_context = ""
            result["stages"]["rag_retrieval"] = {"status": "failed", "error": str(e)}
        
        # ===== STAGE 4: Generate Embeddings =====
        print("\n🔢 [STAGE 4/7] Generating Embeddings...")
        try:
            code_emb, semantic_emb, cfg_emb = generate_embeddings(code, functional_semantic)
            result["stages"]["embedding_generation"] = {
                "status": "completed",
                "embedding_dim": code_emb.shape[0]
            }
        except Exception as e:
            result["stages"]["embedding_generation"] = {"status": "failed", "error": str(e)}
            return result
        
        # ===== STAGE 5: Fusion Model Prediction =====
        print("\n🔮 [STAGE 5/7] Running Fusion Model Prediction...")
        try:
            code_tensor = torch.tensor(code_emb, dtype=torch.float32).unsqueeze(0).to(device)
            semantic_tensor = torch.tensor(semantic_emb, dtype=torch.float32).unsqueeze(0).to(device)
            cfg_tensor = torch.tensor(cfg_emb, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = fusion_model(code_tensor, semantic_tensor, cfg_tensor)
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(logits, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()
            
            vulnerability_type = LABELS.get(predicted_class, "Unknown")
            
            result["stages"]["fusion_prediction"] = {
                "status": "completed",
                "vulnerability": vulnerability_type,
                "confidence": f"{confidence:.2%}",
                "model": "EarlyFusionTransformer"
            }
        except Exception as e:
            vulnerability_type = "Unknown"
            result["stages"]["fusion_prediction"] = {"status": "failed", "error": str(e)}
        
        # ===== STAGE 6: Generate Explanation =====
        print("\n📝 [STAGE 6/7] Generating Explanation...")
        try:
            explanation_prompt = f"""You are a smart contract security expert. Analyze this code and explain the vulnerability.

Code:
{code[:1000]}

Predicted Vulnerability: {vulnerability_type}
RAG Context: {rag_context[:500]}

Provide:
1. Root Cause of the vulnerability
2. Recommended Solution

Be concise and specific."""

            # Call LLM for explanation
            explanation = llm_local.call(explanation_prompt)
            if hasattr(explanation, 'content'):
                explanation = explanation.content
            elif not isinstance(explanation, str):
                explanation = str(explanation)
            
            result["stages"]["explanation"] = {
                "status": "completed",
                "length": len(explanation)
            }
        except Exception as e:
            explanation = "Failed to generate explanation"
            result["stages"]["explanation"] = {"status": "failed", "error": str(e)}
        
        # ===== STAGE 7: Consensus (Validation) - Dùng ChromaDB ở đây =====
        print("\n🤝 [STAGE 7/7] Performing Consensus Validation...")
        try:
            # Query ChromaDB để validate (ChromaDB chỉ dùng ở consensus như consensus_agent.py)
            consensus_context = ""
            try:
                kb_docs = vector_store.similarity_search_with_score(vulnerability_type, k=2)
                consensus_context = "\n\n".join([doc.page_content for doc, score in kb_docs])
            except Exception as e:
                print(f"⚠️ ChromaDB query failed in consensus: {e}")
                consensus_context = "ChromaDB not available"
            
            # Validate consistency between RAG context and Fusion prediction
            consensus_prompt = f"""You are a security audit validator. Compare the following outputs and determine if they are consistent:

1. RAG Neo4j Context:
{rag_context[:500]}

2. ML Model Prediction: {vulnerability_type} (confidence: {confidence:.2%})

3. Explanation:
{explanation[:500]}

4. Knowledge Base Validation:
{consensus_context[:300]}

Provide a brief consensus assessment:
- Are these outputs consistent?
- What is the final confidence level?
- Any conflicts or concerns?

Be concise (2-3 sentences)."""
            
            consensus_result = llm_local.call(consensus_prompt)
            if hasattr(consensus_result, 'content'):
                consensus_result = consensus_result.content
            elif not isinstance(consensus_result, str):
                consensus_result = str(consensus_result)
            
            result["stages"]["consensus"] = {
                "status": "completed",
                "assessment_length": len(consensus_result)
            }
            
            # Final output with consensus
            result["final_output"] = {
                "vulnerability_type": vulnerability_type,
                "confidence": f"{confidence:.2%}",
                "explanation": explanation,
                "consensus_assessment": consensus_result,
                "rag_context_preview": rag_context[:300] + "...",
                "neo4j_vulnerabilities_count": len(neo4j_vulnerabilities)
            }
        except Exception as e:
            result["stages"]["consensus"] = {"status": "failed", "error": str(e)}
            result["final_output"] = {
                "vulnerability_type": vulnerability_type,
                "confidence": f"{confidence:.2%}",
                "explanation": explanation,
                "error": "Consensus validation failed"
            }
        
        # ===== Final Stats =====
        total_time = time.time() - start_time
        result["execution_time_seconds"] = round(total_time, 2)
        result["total_stages"] = 7
        completed = sum(1 for s in result["stages"].values() if s.get("status") == "completed")
        result["completed_stages"] = completed
        
        # Print failed stages
        failed_stages = [name for name, info in result["stages"].items() if info.get("status") == "failed"]
        if failed_stages:
            print(f"\n⚠️ FAILED STAGES: {', '.join(failed_stages)}")
            for stage in failed_stages:
                error = result["stages"][stage].get("error", "Unknown error")
                print(f"   - {stage}: {error}")
        
        print(f"\n✅ Monolithic audit completed in {total_time:.2f}s")
        print(f"📊 Stages: {completed}/7 successful")
        
        # ===== Save output immediately =====
        try:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"💾 Results saved to: {OUTPUT_PATH}")
        except Exception as e:
            print(f"❌ Error saving output: {e}")
        
        return result

# ============================================================
# Single Agent Definition
# ============================================================

monolithic_tool = MonolithicAuditTool()

single_agent = Agent(
    role="Monolithic Smart Contract Auditor",
    goal="Perform complete end-to-end smart contract security audit in a single execution",
    backstory="""You are an all-in-one smart contract auditor that handles:
    - Semantic analysis
    - Knowledge retrieval (RAG)
    - Embedding generation
    - ML-based vulnerability prediction
    - Explanation generation
    - Final consensus
    
    You work alone without specialized sub-agents.""",
    tools=[monolithic_tool],
    llm=llm_local,
    verbose=True,
    allow_delegation=False  # No delegation - single agent does everything
)

single_task = Task(
    description="Analyze the Solidity smart contract and produce a complete security audit report including vulnerability detection and explanation.",
    expected_output="A comprehensive JSON report containing vulnerability type, confidence, explanation, and execution metrics.",
    agent=single_agent
)

# ============================================================
# Main Execution
# ============================================================

def main():
    print("="*60)
    print("🔬 SINGLE-AGENT BASELINE AUDIT")
    print("="*60)
    print("⚠️  This is a monolithic approach where ONE agent does EVERYTHING")
    print("    Compare with multi-agent CrewAI architecture to show benefits\n")
    
    overall_start = time.time()
    
    crew = Crew(
        agents=[single_agent],
        tasks=[single_task],
        process=Process.sequential,
        verbose=True
    )
    
    result = crew.kickoff()
    print("\n✅ SINGLE-AGENT AUDIT COMPLETED", result)
    
    # overall_time = time.time() - overall_start
    
    # # Save output
    # try:
    #     output_data = result.output if hasattr(result, 'output') else str(result)
    #     if isinstance(output_data, str):
    #         # Try to parse if it's JSON string
    #         try:
    #             output_data = json.loads(output_data)
    #         except:
    #             output_data = {"raw_output": output_data}
        
    #     output_data["total_execution_time"] = round(overall_time, 2)
        
    #     # with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    #     #     json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    #     # print(f"\n✅ Results saved to: {OUTPUT_PATH}")
    #     print(f"⏱️  Total execution time: {overall_time:.2f}s")
        
    # except Exception as e:
    #     print(f"❌ Error saving output: {e}")

if __name__ == "__main__":
    main()
