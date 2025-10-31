# explainer.py
import json
import torch
from pydantic import BaseModel, Field
from typing import Type
from crewai import Agent, Task, LLM
from crewai.tools import BaseTool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector
from neo4j import GraphDatabase
from init_unsloth_model import load_unsloth_model
from dotenv import load_dotenv
import os
from transformers import TextIteratorStreamer

load_dotenv()
FUSION_FILE_PATH = "fusion_output_agent.json"
print(f"🔍 Using fusion file path: {FUSION_FILE_PATH}")



url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
model, tokenizer, device = load_unsloth_model()

behaviors_index_explainer_agent = Neo4jVector.from_existing_graph(
    embedding=embeddings,
    url=url,
    username=username,
    password=password,
    index_name="Instance01",
    node_label="Vulnerability",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)
driver_explainer_agent = GraphDatabase.driver(url, auth=(username, password))

class DummyInput(BaseModel):
    pass

class ExplainerTool(BaseTool):
    name: str = "ExplainerTool"
    description: str = "Explain the root causes and solutions using the graph database."
    args_schema: Type[BaseModel] = DummyInput  

    def _run(self) -> dict:
        # ✅ Bỏ hoàn toàn try-except, chỉ đọc file cố định
        with open(FUSION_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        # ✅ Lấy trường Predict trong file
        vuln_type = data.get("Predict", "")
        print(f"🔍 Vulnerability: {vuln_type}")

        # Access các biến toàn cục bạn đã định nghĩa ở nơi khác
        behaviors_index = behaviors_index_explainer_agent
        neo4j_driver = driver_explainer_agent

        prompt_template = """You are a smart contract security assistant specializing in Ethereum, Solidity, and DeFi security.
    Below are several real-world vulnerabilities and their corresponding solutions, extracted from smart contract audits.


    ### Retreived Vulnerability-Solution Pairs:
    {pairs_str}

    Read the retrived vulnerability-solution pairs above carefully and 
    
    ### Task:
    For the following vulnerability, provide:
    - **Vulnerability Name:** The specific name/type of the vulnerability.
    - **Summary of Solution(s):** Key real-world solutions or best practices.
    - **Reason:** Why this vulnerability happens in smart contracts.
    ---
    Vulnerability: {vuln_type}
    ### Your Answer:
    """

        def clean_text(raw_text):
            clean = raw_text.strip()
            if clean.lower().startswith("text:"):
                clean = clean[5:].strip()
            return clean

        def get_solutions_for_vulnerability(vuln_text):
            query = """
            MATCH (v:Vulnerability)
            WHERE v.text = $vuln_text
            OPTIONAL MATCH (v)-[:HAS_SOLUTION]->(s:Solution)
            RETURN v.text AS vulnerability, collect(DISTINCT s.text) AS solutions
            """
            with neo4j_driver.session() as session:
                results = session.execute_read(
                    lambda tx: tx.run(query, vuln_text=vuln_text).data()
                )
            return results

        def build_vul_sol_list(vuln_type, k=1):
            docs_with_score = behaviors_index.similarity_search_with_score(vuln_type, k=k)
            vul_sol_list = []
            for doc, score in docs_with_score:
                vuln_text = clean_text(doc.page_content)
                results = get_solutions_for_vulnerability(vuln_text)
                for item in results:
                    vul_sol_list.append({
                        "vulnerability": item['vulnerability'],
                        "solutions": [s for s in item["solutions"] if s]
                    })
            return vul_sol_list

        def build_vul_sol_context(vul_sol_list):
            context = ""
            for idx, item in enumerate(vul_sol_list, 1):
                context += f"{idx}. Vulnerability: {item['vulnerability']}\n"
                if item['solutions']:
                    context += "   Solutions:\n"
                    for s in item['solutions']:
                        context += f"      - {s}\n"
                else:
                    context += "   Solutions: (No solution provided)\n"
                context += "\n"
            return context


        vul_sol_list = build_vul_sol_list(vuln_type)
        if not vul_sol_list:
            return {
                "type": "error",
                "vuln_type": vuln_type,
                "solutions": [],
                "context": "",
                "raw_llm_output": ""
            }

        pairs_str = build_vul_sol_context(vul_sol_list)
        prompt = prompt_template.format(pairs_str=pairs_str, vuln_type=vuln_type)


        # Generate with Open-source model
        inputs = tokenizer([prompt], return_tensors="pt")
        # If using GPU, pass to CUDA, else use CPU
        inputs = {k: v.cuda() for k, v in inputs.items()}  # GPU acceleration (or remove .cuda() for CPU)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
        _ = model.generate(
            **inputs,
            max_new_tokens=1024,  # Adjust based on the output length needed
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            streamer=streamer
        )

        result = ""
        for token in streamer:
            result += token

        # Prepare the final output
        output = {
            "type": "explanation_result",
            "vuln_type": vuln_type,
            "solutions": [item['solutions'] for item in vul_sol_list],
            "context": pairs_str,
            "raw_llm_output": result
        }

        with open("explainer_output.json", "w") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print("✅ Saved explainer_output.json")
        return output
    



def build_explainer_agent():
    llm_local = LLM(model="ollama/llama3:8b-instruct-q8_0", base_url="http://localhost:11434")
    tool = ExplainerTool()

    agent = Agent(
        role="Explainer Agent",
        goal="Use the ExplainerTool to explain vulnerabilities and suggest mitigations.",
        backstory="An expert in smart contract auditing who delivers security analysis and mitigation strategies.",
        tools=[tool],
        verbose=True,
        llm=llm_local
    )

    task = Task(
        description="Explain the root cause and remediation methods for the identified vulnerability in the file fusion_output_agent.json",
        expected_output="JSON object saved to explainer_output.json containing explanation result. No further tool calls required.",
        agent=agent,
    )


    return agent, task
