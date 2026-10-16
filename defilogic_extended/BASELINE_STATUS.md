# Baseline execution status (2026-10-12)

This note separates executed results from baselines that cannot yet be run in
the local environment.  It must accompany any comparison built on
DeFiLogic-Extended.

| Method | Status | Measurement on source-ready subset | Interpretation |
|---|---|---|---|
| CONVERA | Executed | 3 Mistral (`ministral-8b-latest`) repeats over 5 verified-source incidents | IDR and strict property-family/evidence RCVR are available in `results/run_20261012_defilogic_ministral8b_all3/`. This is a pilot, not a replacement for VERITE. |
| GPTLens | Executed protocol reproduction | 1 Mistral run over the same 5 incidents; IDR = 4/5 | Uses prompts and 0.5/0.25/0.25 ranker from the official release. The backend differs from published GPT-4, so results must be labelled a Mistral reproduction. RCVR is N/A because critic scores are not independent validation evidence. |
| VulnHunt-GPT | Not executable reproducibly | N/A | The paper describes a trained GPT-3 vulnerability detector, but no author-maintained code/checkpoint was located. Community forks are not treated as an official baseline. |
| FinanceFuzz | Partial execution; not scoreable | 1/15 completed in a 30-second WSL reproduction; no detector signal | Upstream commit `103fff2` was run in documented single-file source mode with Petersburg EVM. The public `patches/financefuzz-python310.patch` only backports Python-3.12 type syntax and avoids a missing-timestamp assertion; it adds no oracle. Twelve cases could not resolve imports, two reverted during constructor deployment, and the only completed case was a proxy with no signal. These failures are excluded from IDR/RCVR rather than counted as negative findings. |

## Required setup before FinanceFuzz claims

1. Build per-incident source packages that compile with all imports, provide
   constructor arguments and protocol dependencies, and record the target
   contract, compiler version, EVM version, timeout, and random seed.
2. Use a VM/runtime that supports the incidents' required EVM revisions; the
   released FinanceFuzz runner exposes only Homestead, Byzantium, and Petersburg.
3. Treat FinanceFuzz outputs as property-test witnesses/detections, not as
   CONVERA RCVR.  A common root-cause/evidence annotation protocol would be
   needed before comparing validation recall.
