"""Retrieve verified deployed sources for the DeFiLogic-Extended manifest.

The exploit reproductions in DeFiHackLabs contain post-mortem comments.  They
must never become model input, because that would leak the answer.  This script
therefore retrieves only the vulnerable deployed contract sources from public
Blockscout APIs and stores the response separately from the hidden labels.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "manifest.json"
SOURCE_ROOT = ROOT / "sources"

BLOCKSCOUT_API = {
    "ethereum": "https://eth.blockscout.com/api",
    "base": "https://base.blockscout.com/api",
    "arbitrum": "https://arbitrum.blockscout.com/api",
    "optimism": "https://optimism.blockscout.com/api",
    "polygon": "https://polygon.blockscout.com/api",
    "gnosis": "https://gnosis.blockscout.com/api",
    # Routescan exposes an Etherscan-compatible public endpoint for Avalanche.
    "avalanche": "https://api.routescan.io/v2/network/mainnet/evm/43114/etherscan/api",
    "bsc": "https://api.routescan.io/v2/network/mainnet/evm/56/etherscan/api",
}


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def unwrap_source(source: str) -> dict[str, str]:
    """Normalize Etherscan-compatible source strings into individual files."""
    text = (source or "").strip()
    if text.startswith("{{") and text.endswith("}}"):
        text = text[1:-1]
    if text.startswith("{"):
        try:
            payload = json.loads(text)
            files = {
                name: item.get("content", "")
                for name, item in payload.get("sources", {}).items()
                if isinstance(item, dict) and item.get("content")
            }
            if files:
                return files
        except json.JSONDecodeError:
            pass
    return {"Contract.sol": source} if source else {}


def safe_relative_source_path(name: str) -> Path:
    parts = [slug(part) for part in Path(name.replace("\\", "/")).parts if part not in {".", "..", "/"}]
    output = Path(*parts) if parts else Path("Contract.sol")
    return output if output.suffix else output.with_suffix(".sol")


def fetch_case(case: dict[str, Any], refresh: bool) -> dict[str, Any]:
    case_dir = SOURCE_ROOT / case["id"]
    status_path = case_dir / "source_status.json"
    if status_path.exists() and not refresh:
        return load_json(status_path)
    if refresh:
        stale_source_dir = case_dir / "source"
        if stale_source_dir.exists():
            shutil.rmtree(stale_source_dir)

    endpoint = BLOCKSCOUT_API.get(case["chain"].lower())
    status: dict[str, Any] = {
        "case_id": case["id"],
        "chain": case["chain"],
        "address": case["address"],
        "ready": False,
        "source_count": 0,
        "error": "",
    }
    if not endpoint:
        status["error"] = f"No public Blockscout endpoint configured for chain {case['chain']}."
        write_json(status_path, status)
        return status

    try:
        def fetch_payload(address: str) -> dict[str, Any]:
            try:
                response = requests.get(
                    endpoint,
                    params={"module": "contract", "action": "getsourcecode", "address": address},
                    timeout=45,
                )
                response.raise_for_status()
                payload = response.json()
                records = payload.get("result") or []
                if records and records[0].get("SourceCode"):
                    return payload
            except requests.RequestException:
                # The legacy Blockscout compatibility endpoint may throttle
                # repeated source requests.  Its v2 endpoint serves the same
                # verified-source record and is used only as a provenance-
                # preserving fallback (not for Routescan-backed networks).
                pass

            if "blockscout.com/api" not in endpoint:
                raise RuntimeError("Explorer returned no verified Solidity source.")
            v2_endpoint = endpoint.removesuffix("/api") + f"/api/v2/smart-contracts/{address}"
            v2_response = requests.get(v2_endpoint, timeout=45)
            v2_response.raise_for_status()
            verified = v2_response.json()
            source_code = verified.get("source_code") or ""
            if not verified.get("is_verified") or not source_code:
                raise RuntimeError("Explorer returned no verified Solidity source.")
            implementations = verified.get("implementations") or []
            implementation = ""
            if implementations and isinstance(implementations[0], dict):
                implementation = implementations[0].get("address_hash") or ""
            record = {
                "SourceCode": source_code,
                "ContractName": verified.get("name") or "",
                "CompilerVersion": verified.get("compiler_version") or "",
                "ABI": json.dumps(verified.get("abi") or []),
                "IsProxy": "1" if implementation else "0",
                "Implementation": implementation,
            }
            return {"status": "1", "result": [record], "source_api": "blockscout_v2"}

        payload = fetch_payload(case["address"])
        write_json(case_dir / "explorer_response.json", payload)
        records = payload.get("result") or []
        record = records[0] if isinstance(records, list) and records else {}
        implementation = record.get("ImplementationAddress") or record.get("Implementation")
        is_proxy = str(record.get("IsProxy") or record.get("Proxy") or "").lower() in {"1", "true"}
        analysis_address = case["address"]
        if is_proxy and implementation and re.fullmatch(r"0x[a-fA-F0-9]{40}", implementation):
            implementation_payload = fetch_payload(implementation)
            implementation_records = implementation_payload.get("result") or []
            implementation_record = implementation_records[0] if isinstance(implementation_records, list) and implementation_records else {}
            if implementation_record.get("SourceCode"):
                record = implementation_record
                analysis_address = implementation
                write_json(case_dir / "implementation_response.json", implementation_payload)
        write_json(case_dir / "analysis_response.json", {"status": "1", "result": [record]})
        source_files = unwrap_source(record.get("SourceCode", ""))
        if not source_files:
            status["error"] = "Explorer returned no verified Solidity source."
            write_json(status_path, status)
            return status

        source_dir = case_dir / "source"
        if source_dir.exists():
            shutil.rmtree(source_dir)
        for file_name, content in source_files.items():
            target = source_dir / safe_relative_source_path(file_name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        status.update(
            {
                "ready": True,
                "source_count": len(source_files),
                "contract_name": record.get("ContractName", ""),
                "compiler_version": record.get("CompilerVersion", ""),
                "source_type": "standard_json" if len(source_files) > 1 else "single_file",
                "analysis_address": analysis_address,
                "implementation_resolved": analysis_address.lower() != case["address"].lower(),
            }
        )
    except Exception as exc:  # Preserve retrieval failures as benchmark provenance.
        status["error"] = f"{type(exc).__name__}: {exc}"

    write_json(status_path, status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch verified source code for DeFiLogic-Extended.")
    parser.add_argument("--case", action="append", default=[], help="Case id to fetch; repeat for multiple cases.")
    parser.add_argument("--refresh", action="store_true", help="Refetch cases that already have a status file.")
    args = parser.parse_args()

    manifest = load_json(MANIFEST_PATH)
    cases = manifest["cases"]
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case["id"] in wanted]
        unknown = wanted - {case["id"] for case in cases}
        if unknown:
            raise ValueError(f"Unknown case id(s): {', '.join(sorted(unknown))}")

    summaries = []
    for index, case in enumerate(cases, start=1):
        status = fetch_case(case, args.refresh)
        summaries.append(status)
        state = "READY" if status["ready"] else "SKIP"
        print(f"[{index}/{len(cases)}] {state} {case['id']}: {status.get('source_count', 0)} source file(s) {status.get('error', '')}")

    # Keep the summary representative of the complete manifest even when this
    # invocation refreshed just a selected subset of cases.
    all_statuses = []
    for case in manifest["cases"]:
        status_path = SOURCE_ROOT / case["id"] / "source_status.json"
        if status_path.exists():
            all_statuses.append(load_json(status_path))
        else:
            all_statuses.append({"case_id": case["id"], "ready": False, "error": "not collected"})
    write_json(SOURCE_ROOT / "collection_summary.json", {"cases": all_statuses})


if __name__ == "__main__":
    main()
