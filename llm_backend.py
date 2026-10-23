"""Chat model configuration shared by LangGraph audit nodes.

The returned model exposes ``invoke(prompt).content``, the interface used by
the existing reasoning and consensus tools. No API request is made here.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).resolve().parent / ".env")

def build_chat_model(default_ollama: str = "llama3:8b-instruct-q8_0", provider: str | None = None) -> ChatOpenAI:
    provider = (provider if provider is not None else os.environ.get("LLM_PROVIDER", "")).strip().lower()
    mistral_key = os.environ.get("MISTRAL_API_KEY") or os.environ.get("mistral")
    if provider == "mistral" or (mistral_key and not provider):
        if not mistral_key:
            raise ValueError("LLM_PROVIDER=mistral requires MISTRAL_API_KEY")
        return ChatOpenAI(
            model=os.environ.get("MISTRAL_MODEL", "ministral-8b-latest"),
            api_key=mistral_key,
            base_url=os.environ.get("MISTRAL_BASE_URL", "https://api.mistral.ai/v1"),
            timeout=float(os.environ.get("MISTRAL_TIMEOUT_SECONDS", "90")),
        )

    openai_key = os.environ.get("OPENAI_API_KEY")
    if provider == "openai" or (openai_key and not provider):
        if not openai_key:
            raise ValueError("LLM_PROVIDER=openai requires OPENAI_API_KEY")
        return ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), api_key=openai_key)

    if provider not in ("", "ollama"):
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
    model = os.environ.get("OLLAMA_MODEL", default_ollama).removeprefix("ollama/")
    return ChatOpenAI(
        model=model,
        api_key="ollama",
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    )
