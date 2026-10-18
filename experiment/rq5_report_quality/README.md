# RQ5 — Audit-report quality

RQ5 evaluates generated reports with the criteria used in the manuscript:
Clarity, Completeness, and Readability.  The reports are produced by the
pipeline code in `explainer.py` and `consensus_agent.py`; the final benchmark
results are summarized in `sections/5-results.tex` of the manuscript project.

The current package documents the report-generation implementation and the
evaluation protocol, but it does not yet include a portable LLM-judge runner,
the complete PoPoC report set, or the raw judge outputs.  Consequently, this
RQ should be presented as a reported evaluation rather than a fully automated
reproduction.  Publishing a fixed judge prompt, anonymized reports, and raw
per-run scores is the next required step for a fully reproducible RQ5.
