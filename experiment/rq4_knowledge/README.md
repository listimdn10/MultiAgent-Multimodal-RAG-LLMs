# RQ4 — Knowledge-node and knowledge-graph effects

RQ4 compares structured knowledge with flat/no-knowledge representations in
two places:

1. `embedding_agent.py` and `bi-modal/embedding_agent.py` add the RAG-derived
   knowledge node to the CFG representation; and
2. `rag_agent.py`, `bi-modal/rag_agent.py`, and `knowledge_base/` implement
   knowledge-graph retrieval for the Bi-modal branch.

The paper's RQ4 figures/tables are derived from the controlled stored
predictions of these configurations.  The source needed to inspect how the
knowledge node and retrieval context are formed is included, but this package
does not currently expose a standalone command that regenerates all RQ4 model
training and inference artifacts.  Do not represent it as a fully rerunnable
training benchmark without publishing the corresponding checkpoints and data
split manifests.
