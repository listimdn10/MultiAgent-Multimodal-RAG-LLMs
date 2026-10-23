# CONVERA LangGraph pipeline

`main_pipeline.py` now runs the complete audit as a sequential LangGraph
`StateGraph`:

`semantic → rag → embedding → fusion → reason → consensus`

Each node writes the same JSON artifact used by the previous pipeline:
`input.json`, `rag_output.json`, `parser_output.json`,
`fusion_output_agent.json`, `explainer_output.json`, and
`consensus_output.json`. The graph also returns these artifacts in its state.
The Bi-modal path is `semantic → rag`; its output feeds the embedding node.

## Entry points

From the repository root, run:

```powershell
python main_pipeline.py
python invoke.py
python crew_run.py
python crew_run.py --benchmark business-bench-data/manifest.json --limit 1
```

`invoke.py` runs only `semantic → rag → embedding`.
`crew_run.py` keeps its historical filename for compatibility with existing
scripts, but no longer imports or executes CrewAI. In benchmark mode it
prepares the saved case inputs and runs the `reason` node; in its default mode
it runs `fusion → reason`. The monolithic baseline also uses a one-node
LangGraph wrapper.

## Dependencies and provider configuration

Install `requirements.txt` in the project environment. `langgraph==1.0.3`
and `langchain-openai==1.0.3` replace the CrewAI dependency. OpenAI-compatible
LLM nodes use `LLM_PROVIDER=mistral`, `openai`, or `ollama` with corresponding
`MISTRAL_API_KEY`, `OPENAI_API_KEY`, or `OLLAMA_BASE_URL`. By default the
semantic node needs `GEMINI_API_KEY` and the RAG generator needs the local
Unsloth model. An explicitly labeled rerun can set
`SEMANTIC_PROVIDER=mistral` and `RAG_GENERATION_PROVIDER=mistral` to use
Ministral for those two steps; this changes the model configuration, not the
six-stage graph. The RAG node still needs `NEO4J_*` and `behavior_index`.
Configure keys locally in `.env`; never commit that file.

## Offline verification

```powershell
python -m unittest discover -s tests -p test_langgraph_pipeline.py -v
```

These tests verify node order, subgraph selection, and rejection of stale
artifacts without contacting a model API. A full audit additionally requires
its model weights, databases, and provider access. Stage tools currently use
project-relative JSON paths, so run entry points from the repository root.
