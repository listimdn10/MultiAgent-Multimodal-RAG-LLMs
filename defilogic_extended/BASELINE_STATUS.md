# Baseline execution status (2026-10-21)

This note separates executed results from baselines that cannot yet be run in
the local environment.  It must accompany any comparison built on
DeFiLogic-Extended.

| Method | Status | Measurement on source-ready subset | Interpretation |
|---|---|---|---|
| CONVERA | Executed | 3 Mistral (`ministral-8b-latest`) repeats over 15 verified-source incidents | IDR = 14/15 in every run; mechanism-level RCVR is reported separately after root-cause/evidence review. This external set complements, rather than replaces, VERITE. |
| GPTLens | Executed protocol reproduction | 1 Mistral run over 15 incidents; IDR = 11/15 | Uses the official auditor--critic--ranker prompts. The backend differs from published GPT-4, so it is a protocol reproduction. Post-hoc review found no DeFiLogic-Extended root-cause/evidence match. |
| VulnHunt-GPT | Executed local reproduction | 1 Mistral run over 15 incidents; IDR = 8/15 | The local reproduction is included as a prompt-based baseline, not as a rerun of the original GPT-3 system. Post-hoc review found no DeFiLogic-Extended root-cause/evidence match. |
| GPTScan | Executed official-rule reproduction | 3 Mistral (`ministral-8b-latest`) repeats over 15 incidents; IDR = 2/15 in every run | The official GPTScan rules were run at pinned commit `29a1747` using the public OpenAI-compatible patch. Post-hoc review credits both localized collateral-valuation findings, so RCVR = 2/15 in every run. |
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
