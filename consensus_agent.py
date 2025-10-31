# consensus_agent.py (Version: inline content)
import json
import os
import re
from typing import Type
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

# ===========================
# --- CONFIG GLOBAL PATH ---
# ===========================
SOURCE_PATH = "contracts/sample.sol"
RAG_PATH = "rag_output.json"
EXPLAINER_PATH = "explainer_output.json"

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
EXPLAINER_CONTENT = safe_read_json(EXPLAINER_PATH)


# ===========================
# --- Local LLM (Ollama)
# ===========================
llm_local = LLM(
    model="ollama/llama3:8b-instruct-q8_0",
    base_url="http://localhost:11434"
)


# ===========================
# --- Input Schema (Giữ để CrewAI không lỗi)
# ===========================
class ConsensusInput(BaseModel):
    source_path: str = Field(SOURCE_PATH, description="Path to the source Solidity file")
    rag_path: str = Field(RAG_PATH, description="Path to the RAG agent output JSON")
    explainer_path: str = Field(EXPLAINER_PATH, description="Path to the Explainer agent output JSON")


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

    # ---------- LLM call handling ----------
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

    # ---------- Main Run ----------
    def _run(self, source_path: str, rag_path: str, explainer_path: str) -> dict:
        # Không đọc file nữa → dùng nội dung global
        source_code = SOURCE_CONTENT
        rag_json = RAG_CONTENT
        explainer_json = EXPLAINER_CONTENT

        print(f"🔍 Loaded inline content (source: {len(source_code)} chars, RAG keys: {list(rag_json.keys())}, Explainer keys: {list(explainer_json.keys())})")

        # Build prompt với nội dung thực tế
        prompt = f"""
        You are an expert smart contract auditor. You will compare two audit reports and the original source code,
        then decide which audit is more accurate, or merge their conclusions reliably.

        SOURCE CODE:
        {source_code}

        RAG AGENT OUTPUT:
        {json.dumps(rag_json, ensure_ascii=False, indent=2)}

        EXPLAINER AGENT OUTPUT:
        {json.dumps(explainer_json, ensure_ascii=False, indent=2)}

        TASK:
        1) Evaluate whether the RAG output correctly reflects the vulnerability in the code.
        2) Evaluate whether the Explainer output correctly reflects the vulnerability in the code.
        3) Decide which audit is MORE ACCURATE: "RAG", "Explainer", or "Merged" (if both have complementary, valid findings).
        4) Provide concise reasoning and a final vulnerability summary that is directly grounded in the source code.
        5) Output ONLY valid JSON with this exact schema (no surrounding text):

        {{
            "decision": "RAG" | "Explainer" | "Merged",
            "reasoning": "concise reasons why you chose this decision (grounded in code).",
            "final_vulnerability_summary": "one-sentence summary of the confirmed vulnerability",
            "confidence": float
        }}
        """

        # Call LLM
        try:
            llm_text = self._call_llm(prompt)
        except Exception as e:
            fallback = {
                "decision": "Merged",
                "reasoning": f"LLM invocation failed: {e}",
                "final_vulnerability_summary": rag_json.get("vuln_type", "") or explainer_json.get("vuln_type", ""),
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
                "final_vulnerability_summary": rag_json.get("Predict", "") or explainer_json.get("vuln_type", ""),
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
# --- Build Consensus Agent & Task
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
    description="Read inline source, rag_output.json, and explainer_output.json; return the consensus decision as JSON.",
    expected_output="JSON containing decision, reasoning, final_vulnerability_summary, confidence.",
    agent=consensus_agent,
)


# ===========================
# --- Optional: run standalone
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
