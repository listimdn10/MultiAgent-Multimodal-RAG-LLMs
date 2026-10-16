# Business Logic Benchmark

This folder contains a 15-case business-logic benchmark for the `crew_run.py` batch mode.

The contracts are **representative synthetic cases** inspired by recurring Ethereum business-logic failure modes documented in:

- OWASP Smart Contract Top 10 SC02 Business Logic Vulnerabilities: https://scs.owasp.org/sctop10/SC02-BusinessLogicVulnerabilities/
- FinanceFuzz: fuzzing smart contracts with financial properties: https://www.sciencedirect.com/science/article/pii/S2096720925000284
- Web3Bugs: https://github.com/ZhangZhuoSJTU/Web3Bugs
- DeFiHackLabs: https://github.com/SunWeb3Sec/DeFiHackLabs
- Verichains on self-transfer / TransferMint-style ERC20 flaws: https://blog.verichains.io/p/the-hidden-danger-of-self-transfers

These are not full protocol reproductions. They are minimized benchmark fixtures intended to stress:

- property-guided business-rule reasoning,
- validator-spec synthesis,
- lightweight validation,
- and final audit reliability under imprecise upstream labels.

Results are written under `business-bench-data/results/`.
