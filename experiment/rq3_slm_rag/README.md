# RQ3 — Fine-tuned SLM and RAG effectiveness

RQ3 reports the Bi-modal branch's SLM/RAG results.  The relevant implementation
artifacts are:

- `bi-modal/rag_agent.py`: retrieval-augmented Bi-modal inference;
- `bi-modal/embedding_agent.py`: embedding and CFG-knowledge-node preparation;
- `knowledge_base/`: vulnerability knowledge-graph assets; and
- `tools/fusion_model.py`: the multimodal fusion model used by the system.

The current replication package preserves the trained-model evaluation outputs
and the implementation used to produce them, but does not provide a standalone
one-command training runner or redistributable fine-tuned checkpoints.  RQ3 is
therefore inspectable from the implementation and reported outputs, not an
independent retraining claim.  This limitation should remain explicit in any
artifact or reviewer response.
