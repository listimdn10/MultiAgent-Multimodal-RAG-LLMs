# explainer.py
import json
from pydantic import BaseModel
from typing import Type
from crewai import Agent, Task, LLM
from crewai.tools import BaseTool

FUSION_FILE_PATH = "fusion_output_agent.json"
print(f"🔍 Using fusion file path: {FUSION_FILE_PATH}")

# ✅ Đọc file rag_output.json để lấy nội dung code
with open("rag_output.json", "r", encoding="utf-8") as f:
    rag_data = json.load(f)
code_snippet = rag_data.get("code", "")

class DummyInput(BaseModel):
    pass

class ExplainerTool(BaseTool):
    name: str = "ExplainerTool"
    description: str = "Explain the root causes and solutions for predicted vulnerabilities."
    args_schema: Type[BaseModel] = DummyInput  

    _llm: LLM = None

    def __init__(self, llm: LLM = None, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm

    def _run(self) -> dict:
        # ✅ Đọc file fusion_output_agent.json
        with open(FUSION_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        # ✅ Lấy trường Predict và Predicted_Line_of_Vulnerability từ file
        vuln_type = data.get("Predict", "")
        predicted_line = data.get("Predicted_Line_of_Vulnerability", "Unknown")
        print(f"🔍 Vulnerability: {vuln_type}")
        print(f"📍 Predicted Line: {predicted_line}")

        prompt_template = """Below is an instruction describing a task, followed by a smart contract code snippet. Analyze the code and provide a structured response identifying security vulnerabilities and corresponding solutions.

### Instruction:

You are a smart contract security assistant with deep expertise in Ethereum, Solidity, and DeFi security. A machine learning model has predicted that this code contains a specific vulnerability. Your task is to:
1. Explain the root cause of this vulnerability
2. Provide concrete recommendations to fix or mitigate it

### Predicted Vulnerability:

Type: {vuln_type}
Predicted Location: {predicted_line}

### Code:

{code_snippet}

### Response:

Provide a detailed security analysis including:
1. **Root Cause**: Explain why this vulnerability exists in the code
2. **Solution**: Provide specific code changes or best practices to fix the issue

"""

        prompt = prompt_template.format(
            vuln_type=vuln_type,
            predicted_line=predicted_line,
            code_snippet=code_snippet
        )

        def _call_llm(prompt_text: str) -> str:
            if self._llm is None:
                raise RuntimeError("No LLM provided to ExplainerTool.")
            errors = []
            for method in ["invoke", "generate", "call"]:
                try:
                    fn = getattr(self._llm, method, None)
                    if callable(fn):
                        resp = fn(prompt_text)
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

        print("🧠 Calling agent LLM to generate explanation...")
        try:
            result = _call_llm(prompt)
        except Exception as e:
            result = f"Error calling LLM: {e}"

        # Prepare the final output
        output = {
            "type": "explanation_result",
            "vuln_type": vuln_type,
            "Audit_report": result
        }

        with open("explainer_output.json", "w") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print("✅ Saved explainer_output.json")
        return output
    

def build_explainer_agent():
    llm_local = LLM(model="ollama/llama3:8b-instruct-q8_0", base_url="http://localhost:11434")
    tool = ExplainerTool(llm=llm_local)

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