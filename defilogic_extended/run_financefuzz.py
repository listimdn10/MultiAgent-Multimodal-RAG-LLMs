"""Run the FinanceFuzz baseline on the 15-case DeFiLogic-Extended subset.

Run this script inside WSL/Linux.  FinanceFuzz only supports the Petersburg
EVM, accepts one Solidity source file per invocation, and writes its detector
signals to JSON.  The script records failures explicitly; they are not treated
as negative detections.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


CASE_IDS = (
    "setprotocol_unit_share_rounding",
    "ajna_self_controlled_liquidation",
    "aventa_reward_claim_bound",
    "popsicle_repeated_reward_claim",
    "compound_uni_price_feeder",
    "euler_donation_liquidation",
    "platypus_flashloan_collateral",
    "hundred_finance_exchange_rate",
    "indexed_finance_pool_weight",
    "cream_yusd_collateral_valuation",
    "qubit_bridge_zero_deposit",
    "roe_finance_delegation_collateral",
    "wise_lending_share_withdrawal",
    "harvest_vault_share_accounting",
    "yearn_ydai_share_accounting",
)


def compiler_version(value: str) -> str:
    match = re.search(r"\d+\.\d+\.\d+", value)
    if match is None:
        raise ValueError(f"Cannot parse Solidity version from {value!r}")
    return match.group(0)


def detector_signals(result_path: Path) -> dict[str, list[str]]:
    if not result_path.exists():
        return {}
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    signals: dict[str, list[str]] = {}
    for contract, report in payload.items():
        errors = report.get("errors", {})
        advanced = report.get("advanced_errors", {})
        signals[contract] = sorted(
            [f"swc:{key}" for key, value in errors.items() if value]
            + [f"advanced:{key}" for key, value in advanced.items() if value]
        )
    return signals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--financefuzz-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--seed", type=float, default=0.314159)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    ff_root = args.financefuzz_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    main_py = ff_root / "fuzzer" / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(main_py)

    summary: list[dict[str, Any]] = []
    for case_id in CASE_IDS:
        source_dir = dataset_root / "sources" / case_id
        metadata = json.loads((source_dir / "source_status.json").read_text(encoding="utf-8"))
        source = source_dir / "source" / "Contract.sol"
        case_dir = output / case_id
        case_dir.mkdir(exist_ok=True)
        result_path = case_dir / "result.json"
        log_path = case_dir / "run.log"
        version = compiler_version(metadata["compiler_version"])
        command = [
            sys.executable,
            str(main_py),
            "-s", str(source),
            "-c", metadata["contract_name"],
            "--solc", f"v{version}",
            "--evm", "petersburg",
            "-t", str(args.seconds),
            "--seed", str(args.seed),
            "-r", str(result_path),
        ]
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=ff_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.seconds + 60,
                check=False,
            )
            output_text = completed.stdout
            return_code: int | None = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            output_text = exc.stdout or ""
            return_code = None
            timed_out = True
        log_path.write_text(output_text, encoding="utf-8")
        signals = detector_signals(result_path)
        summary.append(
            {
                "case_id": case_id,
                "contract_name": metadata["contract_name"],
                "solc": version,
                "source_count": metadata["source_count"],
                "evm": "petersburg",
                "timeout_seconds": args.seconds,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "return_code": return_code,
                "timed_out": timed_out,
                "completed": return_code == 0 and result_path.exists(),
                "detector_signals": signals,
                "result_file": str(result_path),
                "log_file": str(log_path),
            }
        )
        print(f"{case_id}: code={return_code}, result={result_path.exists()}, signals={signals}", flush=True)

    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    completed_count = sum(row["completed"] for row in summary)
    print(f"Completed {completed_count}/{len(summary)} cases. Summary: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
