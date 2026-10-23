"""Functional-semantic stage used before RAG retrieval in the audit graph."""

from __future__ import annotations

import os

def analyze_functional_semantics(code: str) -> str:
    """Preserve the original Bi-modal semantic prompt without loading RAG models."""
    prompt = f"""
        What is the purpose of the following code snippet?
        Please summarize the answer in one sentence with the following format:
        “Abstract purpose:” (one sentence).
        Then summarize the functions of the code in a list format without explanation:
        “Detail Behaviors: 1. 2. 3...”

        Here is the code:
        {code}
        """
    provider = os.environ.get("SEMANTIC_PROVIDER", "gemini").strip().lower()
    if provider == "mistral":
        from llm_backend import build_chat_model

        response = build_chat_model(provider="mistral").invoke(prompt)
        semantic = response.content
    elif provider == "gemini":
        import google.generativeai as genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required for the Gemini semantic stage")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-flash-latest")
        response = model.generate_content(prompt)
        semantic = getattr(response, "text", None)
    else:
        raise ValueError(f"Unsupported SEMANTIC_PROVIDER: {provider}")
    if not semantic or not semantic.strip():
        raise RuntimeError("Gemini returned no functional-semantic description")
    return semantic
