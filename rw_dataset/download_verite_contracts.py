import os
import re
import json
import time
import argparse
import requests
import pandas as pd
from pathlib import Path

ETHERSCAN_V2_API = "https://api.etherscan.io/v2/api"

CHAIN_ID_MAP = {
    "ethereum": "1",
    "eth": "1",
    "bsc": "56",
    "bnb": "56",
    "binance smart chain": "56",
    "polygon": "137",
    "matic": "137",
    "arbitrum": "42161",
    "optimism": "10",
    "fantom": "250",
    "avalanche": "43114",
    "avax": "43114",
    "base": "8453",
    "cronos": "25",
    "gnosis": "100",
}

def clean_name(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-zA-Z0-9_\-]+", "_", s)
    return s.strip("_") or "unknown"

def split_addresses(value: str):
    if pd.isna(value):
        return []
    candidates = re.split(r"[,;\s]+", str(value))
    return [x.strip() for x in candidates if re.fullmatch(r"0x[a-fA-F0-9]{40}", x.strip())]

def normalize_chain(chain: str):
    chain_clean = str(chain).strip().lower()
    return CHAIN_ID_MAP.get(chain_clean)

def fetch_source(address: str, chainid: str, api_key: str, timeout=30):
    params = {
        "chainid": chainid,
        "module": "contract",
        "action": "getsourcecode",
        "address": address,
        "apikey": api_key,
    }

    r = requests.get(ETHERSCAN_V2_API, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def unwrap_source_code(source_code: str):
    """
    Etherscan source may be:
    1. plain Solidity string
    2. JSON string
    3. double-braced JSON: {{ ... }}
    """
    if not source_code:
        return {"type": "empty", "files": {}}

    src = source_code.strip()

    if src.startswith("{{") and src.endswith("}}"):
        src = src[1:-1]

    if src.startswith("{"):
        try:
            obj = json.loads(src)
            files = {}

            sources = obj.get("sources", {})
            for filename, content_obj in sources.items():
                content = content_obj.get("content", "")
                files[filename] = content

            if files:
                return {"type": "standard_json", "files": files}

            return {"type": "json_unknown", "files": {"source.json": json.dumps(obj, indent=2)}}
        except Exception:
            pass

    return {"type": "single_file", "files": {"Contract.sol": source_code}}

def save_contract(out_dir: Path, row_info: dict, address: str, chainid: str, result: dict):
    project = clean_name(row_info.get("Name", "unknown"))
    bug_type = clean_name(row_info.get("Bug Type", "unknown"))
    chain = clean_name(row_info.get("Chain", "unknown"))

    contract_dir = out_dir / chain / project / address
    contract_dir.mkdir(parents=True, exist_ok=True)

    raw_path = contract_dir / "etherscan_raw.json"
    raw_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    status = result.get("status")
    message = result.get("message")
    records = result.get("result", [])

    meta = {
        "project": row_info.get("Name"),
        "date": row_info.get("Date"),
        "chain": row_info.get("Chain"),
        "chainid": chainid,
        "height": row_info.get("Height"),
        "bug_type": row_info.get("Bug Type"),
        "address": address,
        "status": status,
        "message": message,
        "downloaded": False,
        "verified": False,
        "contract_name": None,
        "compiler_version": None,
        "proxy": None,
        "implementation": None,
        "error": None,
    }

    if not isinstance(records, list) or len(records) == 0:
        meta["error"] = "No result records"
        (contract_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    rec = records[0]
    source_code = rec.get("SourceCode", "")

    meta.update({
        "contract_name": rec.get("ContractName"),
        "compiler_version": rec.get("CompilerVersion"),
        "optimization_used": rec.get("OptimizationUsed"),
        "runs": rec.get("Runs"),
        "license_type": rec.get("LicenseType"),
        "proxy": rec.get("Proxy"),
        "implementation": rec.get("Implementation"),
        "abi_available": bool(rec.get("ABI") and rec.get("ABI") != "Contract source code not verified"),
    })

    if rec.get("ABI"):
        (contract_dir / "abi.json").write_text(rec.get("ABI"), encoding="utf-8")

    if not source_code or source_code == "Contract source code not verified":
        meta["error"] = "Source code not verified or empty"
        (contract_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    unpacked = unwrap_source_code(source_code)

    src_dir = contract_dir / "sources"
    src_dir.mkdir(exist_ok=True)

    for filename, content in unpacked["files"].items():
        safe_parts = [clean_name(p) for p in Path(filename).parts]
        safe_file = src_dir.joinpath(*safe_parts)

        if safe_file.suffix == "":
            safe_file = safe_file.with_suffix(".sol")

        safe_file.parent.mkdir(parents=True, exist_ok=True)
        safe_file.write_text(content, encoding="utf-8", errors="ignore")

    meta["downloaded"] = True
    meta["verified"] = True
    meta["source_type"] = unpacked["type"]
    meta["num_source_files"] = len(unpacked["files"])

    (contract_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="dataset.csv")
    parser.add_argument("--out", default="verite_contracts")
    parser.add_argument("--api-key", default=os.getenv("ETHERSCAN_API_KEY"))
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Missing API key. Set ETHERSCAN_API_KEY or pass --api-key YOUR_KEY")

    df = pd.read_csv(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    seen = set()
    count = 0

    for idx, row in df.iterrows():
        row_info = row.to_dict()
        chainid = normalize_chain(row_info.get("Chain", ""))

        if not chainid:
            print(f"[SKIP] Unknown chain: {row_info.get('Chain')} | project={row_info.get('Name')}")
            continue

        addresses = split_addresses(row_info.get("Addresses", ""))

        for address in addresses:
            key = (chainid, address.lower())

            if key in seen:
                continue
            seen.add(key)

            if args.limit is not None and count >= args.limit:
                break

            print(f"[FETCH] {row_info.get('Name')} | {row_info.get('Chain')} | {address}")

            try:
                result = fetch_source(address, chainid, args.api_key)
                meta = save_contract(out_dir, row_info, address, chainid, result)
                summary.append(meta)

                if meta["downloaded"]:
                    print(f"  -> OK: {meta['contract_name']}")
                else:
                    print(f"  -> NO SOURCE: {meta.get('error')}")

            except Exception as e:
                print(f"  -> ERROR: {e}")
                summary.append({
                    "project": row_info.get("Name"),
                    "chain": row_info.get("Chain"),
                    "chainid": chainid,
                    "address": address,
                    "downloaded": False,
                    "verified": False,
                    "error": str(e),
                })

            count += 1
            time.sleep(args.sleep)

        if args.limit is not None and count >= args.limit:
            break

    summary_df = pd.DataFrame(summary)
    summary_path = out_dir / "download_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nDone.")
    print(f"Saved to: {out_dir}")
    print(f"Summary: {summary_path}")
    print(summary_df["downloaded"].value_counts(dropna=False))

if __name__ == "__main__":
    main()