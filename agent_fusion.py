# agent_fusion.py
import json, pickle, torch, numpy as np
import torch.nn as nn
from pydantic import BaseModel, Field
from typing import Type
from crewai import Agent, Task, LLM
from crewai.tools import BaseTool

# ==== MLP MODEL ====
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# ==== TOOL ====
class DummyInput(BaseModel):
    path: str = Field("parser_output.json", description="Path to the JSON file containing embeddings.")

class EmbeddingPredictorTool(BaseTool):
    name: str = "MLP Embedding Vulnerability Predictor"
    description: str = "Predict the type of security vulnerability from the embedding JSON."
    args_schema: Type[BaseModel] = DummyInput

    def _load_model(self, input_size):
        with open("tools/label_encoder.pkl", "rb") as f:
            self._label_encoder = pickle.load(f)
        num_classes = len(self._label_encoder.classes_)
        self._model = MLP(input_size, num_classes)
        self._model.load_state_dict(torch.load("tools/mlp_model.pth", map_location="cpu"))
        self._model.eval()

    def _run(self, path: str) -> str:
        with open(path, "r") as f:
            data = json.load(f)
        combined_emb = []
        for key in ["cfg_embeddings", "source_code_embeddings", "functional_semantic_embeddings"]:
            if key in data:
                emb = data[key][0] if isinstance(data[key][0], list) else data[key]
                combined_emb.extend(emb)
        combined_emb = np.array(combined_emb, dtype=np.float32)

        if not hasattr(self, "_model"):
            self._load_model(input_size=len(combined_emb))

        tensor = torch.tensor(combined_emb.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            probs = torch.softmax(self._model(tensor), dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
        label = self._label_encoder.inverse_transform([pred_idx])[0]
        confidence = probs[0, pred_idx].item()

        output = {"Predict": label, "Confidence": confidence}
        with open("fusion_output_agent.json", "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print("✅ Saved fusion_output_agent.json")
        return f"🎯 {label} ({confidence:.2%})"


# ==== AGENT ====
def build_fusion_agent():
    llm_local = LLM(model="ollama/llama3:8b-instruct-q8_0", base_url="http://localhost:11434")
    tool = EmbeddingPredictorTool()

    agent = Agent(
        role="ML Security Analyzer",
        goal="Analyze embeddings and predict security vulnerabilities.",
        backstory="Uses embeddings to identify the type of security vulnerability in smart contracts.",
        tools=[tool],
        verbose=True,
        llm=llm_local
    )

    task = Task(
        description="Predict the type of security vulnerability present in the file parser_output.json.",
        expected_output="Name of the security vulnerability and its confidence score.",
        agent=agent,
        input={"path": "parser_output.json"}
    )

    return agent, task
