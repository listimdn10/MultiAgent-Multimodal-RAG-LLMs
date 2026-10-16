import argparse
import csv
import importlib
import io
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import END, START, StateGraph

from business_bench_utils import generate_fusion_seed, generate_parser_output


try:
    if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if sys.stderr.encoding is None or sys.stderr.encoding.lower() != "utf-8":
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass


ROOT = Path(__file__).resolve().parent
RW_ROOT = ROOT / "rw_dataset"
DATASET_CSV = RW_ROOT / "dataset.csv"
VERITE_ROOT = RW_ROOT / "verite_contracts"
GENERATED_ROOT = RW_ROOT / "generated"
MANIFEST_ROOT = RW_ROOT / "manifests"
RESULTS_ROOT = RW_ROOT / "results"

RAG_OUTPUT_PATH = ROOT / "rag_output.json"
PARSER_OUTPUT_PATH = ROOT / "parser_output.json"
FUSION_OUTPUT_PATH = ROOT / "fusion_output_agent.json"
EXPLAINER_OUTPUT_PATH = ROOT / "explainer_output.json"
SAMPLE_CONTRACT_PATH = ROOT / "contracts" / "sample.sol"


BUSINESS_LOGIC_TYPES = {
    "Logical Bug",
    "Miscalculation",
    "Insufficient Input Validaton",
}

GENERIC_TOKEN_NAMES = {
    "wbnb",
    "weth",
    "busd",
    "usdt",
    "usdc",
    "dai",
    "cake",
    "bep20token",
    "erc20",
    "token",
}

GENERIC_INFRA_PATTERNS = (
    "pancakepair",
    "pair",
    "router",
    "factory",
)

REAL_WORLD_PROFITS = {
    "uranium": 40814877.9,
    "zeed": 1042284.8,
    "hackdao": 56973.5,
    "shadowfi": 299006.4,
    "pltd": 24493.0,
    "hpay": 31415.7,
    "health": 4539.8,
    "bego": 3235.2,
    "seaman": 7775.6,
    "mbc": 5904.4,
    "rfb": 3526.2,
    "aes": 61608.0,
    "dfs": 1458.1,
    "upswing": 590.1,
    "thoreumfinance": 1776.7,
    "bevo": 44377.3,
    "safemoon": 8574004.4,
    "swapos": 278903.0,
    "olife": 9966.9,
    "axioma": 6904.9,
    "melo": 90607.3,
    "fapen": 635.8,
    "cellframe": 75208.6,
    "depusdt": 69786.6,
    "bunn": 12699.8,
    "bamboo": 50210.1,
    "sut": 8033.7,
    "uwerx": 321442.1,
    "gss": 24883.4,
}


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_rows() -> List[Dict[str, str]]:
    with DATASET_CSV.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_addresses(raw: str) -> List[str]:
    return [address.strip() for address in raw.split(",") if address.strip()]


def normalize_bug_type(bug_type: str) -> str:
    return bug_type.strip()


def split_rows(rows: List[Dict[str, str]], split: str) -> List[Dict[str, str]]:
    if split == "all":
        return rows
    if split == "business_logic":
        return [row for row in rows if normalize_bug_type(row["Bug Type"]) in BUSINESS_LOGIC_TYPES]
    raise ValueError(f"Unsupported split: {split}")


def clean_source_code_blob(raw_source: str) -> Optional[Dict[str, Any]]:
    text = (raw_source or "").strip()
    if not text:
        return None
    if text.startswith("{{") and text.endswith("}}"):
        text = text[1:-1]
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    return None


def extract_abi_function_names(abi: Any) -> List[str]:
    names: List[str] = []
    if isinstance(abi, str):
        try:
            abi = json.loads(abi)
        except json.JSONDecodeError:
            return names
    if not isinstance(abi, list):
        return names
    for entry in abi:
        if isinstance(entry, dict) and entry.get("type") == "function" and entry.get("name"):
            names.append(entry["name"])
    return names


def load_contract_bundle(address_dir: Path) -> Dict[str, Any]:
    metadata = load_json(address_dir / "metadata.json")
    raw = load_json(address_dir / "etherscan_raw.json")
    result = raw["result"][0]
    source_blob = clean_source_code_blob(result.get("SourceCode", ""))
    abi_names = extract_abi_function_names(result.get("ABI"))

    contract_name = metadata.get("contract_name") or result.get("ContractName") or "UnknownContract"
    contract_file_name = result.get("ContractFileName", "")

    files: List[Tuple[str, str]] = []
    if source_blob and isinstance(source_blob.get("sources"), dict):
        for file_name, file_payload in source_blob["sources"].items():
            if isinstance(file_payload, dict):
                content = file_payload.get("content", "")
                if content:
                    files.append((file_name, content))
    else:
        for path in sorted((address_dir / "sources").rglob("*.sol")):
            try:
                files.append((str(path.relative_to(address_dir / "sources")), path.read_text(encoding="utf-8")))
            except Exception:
                continue

    main_file = ""
    main_content = ""
    if contract_file_name:
        for file_name, content in files:
            if file_name == contract_file_name or file_name.endswith(contract_file_name):
                main_file = file_name
                main_content = content
                break
    if not main_content:
        pattern = re.compile(rf"\bcontract\s+{re.escape(contract_name)}\b")
        for file_name, content in files:
            if pattern.search(content):
                main_file = file_name
                main_content = content
                break
    if not main_content and files:
        main_file, main_content = files[0]

    ordered_files: List[Tuple[str, str]] = []
    if main_content:
        ordered_files.append((main_file or contract_name, main_content))
    for file_name, content in files:
        if file_name != main_file:
            ordered_files.append((file_name, content))

    parts: List[str] = []
    for idx, (file_name, content) in enumerate(ordered_files):
        header = f"// FILE: {file_name}"
        parts.append(header)
        parts.append(content)
        if idx < len(ordered_files) - 1:
            parts.append("\n")

    combined_code = "\n".join(parts).strip()
    function_summary = ", ".join(abi_names[:24]) if abi_names else "unavailable"
    functional_semantic = (
        f"Real-world incident contract {contract_name} on {metadata.get('chain', 'unknown')} chain. "
        f"Primary verified file: {main_file or 'unknown'}. "
        f"ABI functions include: {function_summary}."
    )

    return {
        "contract_name": contract_name,
        "contract_file_name": contract_file_name,
        "main_file": main_file,
        "compiler_version": metadata.get("compiler_version", ""),
        "abi_functions": abi_names,
        "code": combined_code,
        "functional_semantic": functional_semantic,
        "num_files": len(files),
        "source_type": metadata.get("source_type", ""),
    }


def incident_dir(row: Dict[str, str]) -> Path:
    return VERITE_ROOT / row["Chain"] / row["Name"]


def profit_for_incident(name: str) -> Optional[float]:
    return REAL_WORLD_PROFITS.get(name.lower())


def map_gold_bug_family(bug_type: str) -> str:
    bug_type = normalize_bug_type(bug_type)
    mapping = {
        "Logical Bug": "logical_bug",
        "Miscalculation": "miscalculation",
        "Insufficient Input Validaton": "input_validation",
        "Access Control": "access_control",
        "Price Manipulation": "price_manipulation",
        "Reflection Token": "reflection_token",
        "Reentrancy": "reentrancy",
        "Bad Random": "bad_random",
    }
    return mapping.get(bug_type, "unknown")


def incident_tokens(name: str) -> List[str]:
    return [token for token in re.split(r"[^a-z0-9]+", name.lower()) if token]


def load_address_identity(address_dir: Path) -> Dict[str, Any]:
    metadata = load_json(address_dir / "metadata.json")
    contract_name = (metadata.get("contract_name") or "").strip()
    return {
        "contract_name": contract_name,
        "num_source_files": int(metadata.get("num_source_files") or 0),
        "source_type": metadata.get("source_type", ""),
        "compiler_version": metadata.get("compiler_version", ""),
        "verified": bool(metadata.get("verified")),
    }


def score_address_candidate(incident_name: str, identity: Dict[str, Any]) -> Tuple[float, List[str]]:
    score = 0.0
    reasons: List[str] = []
    contract_name = (identity.get("contract_name") or "").strip()
    lower_name = contract_name.lower()
    tokens = incident_tokens(incident_name)
    num_files = int(identity.get("num_source_files") or 0)
    source_type = identity.get("source_type") or ""

    if num_files:
        delta = min(num_files, 8) * 1.5
        score += delta
        reasons.append(f"+{delta:.1f} source-file depth ({num_files})")
    if source_type == "standard_json":
        score += 2.0
        reasons.append("+2.0 verified multi-file standard_json source")
    elif source_type == "single_file":
        score -= 1.0
        reasons.append("-1.0 single-file source")

    if any(token and token in lower_name for token in tokens):
        score += 3.0
        reasons.append("+3.0 contract name aligned with incident name")

    if lower_name in GENERIC_TOKEN_NAMES:
        score -= 5.0
        reasons.append("-5.0 generic token contract")
    elif any(pat in lower_name for pat in GENERIC_INFRA_PATTERNS) and not any(token in lower_name for token in tokens):
        score -= 3.5
        reasons.append("-3.5 generic AMM/pair infrastructure name")

    if num_files <= 1 and lower_name not in GENERIC_TOKEN_NAMES and not any(token in lower_name for token in tokens):
        score -= 1.0
        reasons.append("-1.0 shallow single-file contract")

    for keyword in ("vault", "minter", "migration", "network", "farm", "staking", "reward", "govern", "pool"):
        if keyword in lower_name:
            score += 1.0
            reasons.append(f"+1.0 semantic keyword `{keyword}`")
            break

    return score, reasons


def select_culprit_addresses(addr_entries: List[Dict[str, Any]], max_selected: int = 2) -> List[str]:
    scored = sorted(addr_entries, key=lambda item: item.get("culprit_score", -9999.0), reverse=True)
    if not scored:
        return []

    top_score = float(scored[0].get("culprit_score", 0.0))
    threshold = max(top_score - 2.0, 0.5)
    selected = [entry["address"] for entry in scored if float(entry.get("culprit_score", 0.0)) >= threshold][:max_selected]
    if not selected:
        selected = [scored[0]["address"]]
    return selected


def build_rw_manifest(split: str = "business_logic") -> Dict[str, Any]:
    rows = split_rows(load_rows(), split)
    incidents = []
    for row in rows:
        addresses = parse_addresses(row["Addresses"])
        addr_entries = []
        for address in addresses:
            address_dir = incident_dir(row) / address
            identity: Dict[str, Any] = {}
            culprit_score = -9999.0
            culprit_reasons: List[str] = []
            if address_dir.exists():
                try:
                    identity = load_address_identity(address_dir)
                    culprit_score, culprit_reasons = score_address_candidate(row["Name"], identity)
                except Exception as exc:
                    culprit_reasons = [f"identity load failed: {exc}"]
            addr_entries.append(
                {
                    "address": address,
                    "address_dir": str(address_dir.relative_to(ROOT)),
                    "exists": address_dir.exists(),
                    "contract_name": identity.get("contract_name", ""),
                    "num_source_files": identity.get("num_source_files", 0),
                    "source_type": identity.get("source_type", ""),
                    "culprit_score": culprit_score,
                    "culprit_reasons": culprit_reasons,
                }
            )

        selected_addresses = select_culprit_addresses(addr_entries)
        for entry in addr_entries:
            entry["selected"] = entry["address"] in selected_addresses

        incidents.append(
            {
                "incident_name": row["Name"],
                "chain": row["Chain"],
                "date": row["Date"],
                "height": row["Height"],
                "bug_type": normalize_bug_type(row["Bug Type"]),
                "gold_bug_family": map_gold_bug_family(row["Bug Type"]),
                "addresses": addr_entries,
                "selected_addresses": selected_addresses,
                "profit": profit_for_incident(row["Name"]),
            }
        )

    return {
        "split": split,
        "dataset_csv": str(DATASET_CSV.relative_to(ROOT)),
        "results_dir": str((RESULTS_ROOT / split).relative_to(ROOT)),
        "incidents": incidents,
    }


def write_manifests() -> List[Path]:
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for split in ("business_logic", "all"):
        payload = build_rw_manifest(split=split)
        path = MANIFEST_ROOT / f"rw_manifest_{split}.json"
        save_json(path, payload)
        paths.append(path)
    return paths


class ExplainerGraphState(TypedDict, total=False):
    """State passed through the single-node real-world audit graph."""

    output: Dict[str, Any]


def run_explainer_node(state: ExplainerGraphState) -> ExplainerGraphState:
    """Run one complete property-guided audit without CrewAI orchestration."""
    import explainer

    importlib.reload(explainer)
    tool = explainer.build_explainer_tool()
    return {"output": tool._run()}


def build_explainer_graph():
    graph = StateGraph(ExplainerGraphState)
    graph.add_node("explainer", run_explainer_node)
    graph.add_edge(START, "explainer")
    graph.add_edge("explainer", END)
    return graph.compile()


def build_rag_payload(case: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "contract_name": case["contract_name"],
        "code": case["code"],
        "functional_semantic": case["functional_semantic"],
        "Audit_report": case.get("seed_audit_report", ""),
        "metadata": {
            "incident_name": case["incident_name"],
            "chain": case["chain"],
            "address": case["address"],
            "bug_type": case["bug_type"],
            "gold_bug_family": case["gold_bug_family"],
            "compiler_version": case["compiler_version"],
            "main_file": case["main_file"],
            "source_type": case["source_type"],
            "profit": case.get("profit"),
        },
    }


def prepare_case_inputs(case: Dict[str, Any]) -> None:
    generated_contract = GENERATED_ROOT / case["generated_contract_relpath"]
    generated_contract.parent.mkdir(parents=True, exist_ok=True)
    generated_contract.write_text(case["code"], encoding="utf-8")
    shutil.copyfile(generated_contract, SAMPLE_CONTRACT_PATH)

    save_json(RAG_OUTPUT_PATH, build_rag_payload(case))
    generate_parser_output(str(RAG_OUTPUT_PATH), str(PARSER_OUTPUT_PATH))
    generate_fusion_seed(str(PARSER_OUTPUT_PATH), str(FUSION_OUTPUT_PATH))


def parse_audit_report(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def confidence_from_output(output: Dict[str, Any]) -> float:
    report = parse_audit_report(output.get("Audit_report"))
    confidence = report.get("confidence")
    if isinstance(confidence, (int, float)):
        return float(confidence)
    return 0.0


def validation_rank(status: str) -> int:
    return {"violated": 3, "supported": 2, "inconclusive": 1}.get((status or "").lower(), 0)


def infer_predicted_family(output: Dict[str, Any]) -> str:
    report = parse_audit_report(output.get("Audit_report"))
    haystack = " ".join(
        [
            str(output.get("vuln_type", "")),
            str(output.get("validated_bug_type", "")),
            str(report.get("root_cause_analysis", "")),
            str(report.get("violated_property", "")),
            str(output.get("validation_bundle", {}).get("summary", "")),
        ]
    ).lower()

    if any(token in haystack for token in ("onlyowner", "owner", "admin", "forbidden", "unauthor", "permission", "access")):
        return "access_control"
    if any(token in haystack for token in ("input validation", "parameter", "sanit", "validation", "bounds check", "require")):
        return "input_validation"
    if any(token in haystack for token in ("calculation", "rounding", "price", "fee", "math", "accounting", "conservation", "equivalence")):
        return "miscalculation"
    relation_types = []
    for spec in output.get("validator_spec_bundle", {}).get("validator_specs", []):
        relation_type = spec.get("assertion", {}).get("relation_type", "")
        if relation_type:
            relation_types.append(relation_type)
    if relation_types:
        return "logical_bug"
    return "unknown"


def persist_case_outputs(case: Dict[str, Any], run_dir: Path) -> None:
    case_dir = run_dir / "addresses" / case["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)

    generated_contract = GENERATED_ROOT / case["generated_contract_relpath"]
    if generated_contract.exists():
        shutil.copyfile(generated_contract, case_dir / generated_contract.name)

    for src, dst_name in [
        (RAG_OUTPUT_PATH, "rag_output.json"),
        (PARSER_OUTPUT_PATH, "parser_output.json"),
        (FUSION_OUTPUT_PATH, "fusion_output_agent.json"),
        (EXPLAINER_OUTPUT_PATH, "explainer_output.json"),
    ]:
        if src.exists():
            shutil.copyfile(src, case_dir / dst_name)

    save_json(
        case_dir / "case_metadata.json",
        {
            "incident_name": case["incident_name"],
            "chain": case["chain"],
            "address": case["address"],
            "bug_type": case["bug_type"],
            "gold_bug_family": case["gold_bug_family"],
            "profit": case.get("profit"),
            "contract_name": case["contract_name"],
            "compiler_version": case["compiler_version"],
            "main_file": case["main_file"],
            "source_type": case["source_type"],
            "num_files": case["num_files"],
            "culprit_score": case.get("culprit_score"),
            "culprit_reasons": case.get("culprit_reasons", []),
            "selected_by_heuristic": case.get("selected_by_heuristic", False),
        },
    )


def build_cases_from_manifest(manifest: Dict[str, Any], all_addresses: bool = False) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for incident in manifest["incidents"]:
        for entry in incident["addresses"]:
            if not all_addresses and not entry.get("selected", False):
                continue
            address_dir = ROOT / entry["address_dir"]
            if not entry.get("exists") or not address_dir.exists():
                continue
            try:
                bundle = load_contract_bundle(address_dir)
            except Exception as exc:
                print(f"Skipping {incident['incident_name']} {entry['address']}: {exc}")
                continue

            case_id = f"{incident['incident_name']}__{entry['address']}"
            generated_relpath = Path(manifest["split"]) / incident["incident_name"] / f"{entry['address']}.sol"
            cases.append(
                {
                    "case_id": case_id,
                    "incident_name": incident["incident_name"],
                    "chain": incident["chain"],
                    "address": entry["address"],
                    "bug_type": incident["bug_type"],
                    "gold_bug_family": incident["gold_bug_family"],
                    "profit": incident.get("profit"),
                    "culprit_score": entry.get("culprit_score"),
                    "culprit_reasons": entry.get("culprit_reasons", []),
                    "selected_by_heuristic": entry.get("selected", False),
                    "generated_contract_relpath": str(generated_relpath),
                    **bundle,
                }
            )
    return cases


def run_manifest(
    manifest_path: Path,
    limit_incidents: int = 0,
    incident_name: str = "",
    all_addresses: bool = False,
    resume_run_dir: Optional[Path] = None,
) -> Path:
    manifest = load_json(manifest_path)
    if incident_name:
        manifest["incidents"] = [item for item in manifest["incidents"] if item["incident_name"] == incident_name]
    if limit_incidents and limit_incidents > 0:
        manifest["incidents"] = manifest["incidents"][:limit_incidents]

    cases = build_cases_from_manifest(manifest, all_addresses=all_addresses)
    if resume_run_dir is not None:
        run_dir = resume_run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = ROOT / manifest["results_dir"] / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "benchmark_summary.json"
    summary_rows: List[Dict[str, Any]] = []
    completed_case_ids = set()
    if summary_path.exists():
        existing_summary = load_json(summary_path)
        summary_rows = list(existing_summary.get("cases", []))
        completed_case_ids = {row.get("case_id", "") for row in summary_rows if row.get("case_id")}

    pending_cases = [case for case in cases if case["case_id"] not in completed_case_ids]
    total_cases = len(cases)
    audit_graph = build_explainer_graph()

    for idx, case in enumerate(pending_cases, start=1):
        completed_so_far = len(completed_case_ids) + idx - 1
        print(f"\n=== [{idx}/{len(cases)}] Running {case['incident_name']} :: {case['address']} ===")
        status = "ok"
        error_message = ""
        try:
            prepare_case_inputs(case)
            graph_result = audit_graph.invoke({})
            result = graph_result.get("output", {})
            print(json.dumps(result, ensure_ascii=False, indent=2))
            persist_case_outputs(case, run_dir)
        except Exception as exc:
            status = "failed"
            error_message = str(exc)
            print(f"Case failed: {case['case_id']} -> {exc}")
            persist_case_outputs(case, run_dir)

        summary_rows.append(
            {
                "case_id": case["case_id"],
                "incident_name": case["incident_name"],
                "address": case["address"],
                "status": status,
                "error": error_message,
                "result_dir": str((run_dir / "addresses" / case["case_id"]).relative_to(ROOT)),
            }
        )
        print(f"Completed {completed_so_far + 1}/{total_cases} cases in {run_dir.name}")

    save_json(
        summary_path,
        {
            "manifest": str(manifest_path.relative_to(ROOT)),
            "split": manifest["split"],
            "all_addresses": all_addresses,
            "run_dir": str(run_dir.relative_to(ROOT)),
            "cases": summary_rows,
        },
    )
    return run_dir


def incident_case_rows(run_dir: Path) -> List[Dict[str, Any]]:
    summary_path = run_dir / "benchmark_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing benchmark summary: {summary_path}")
    summary = load_json(summary_path)

    rows: List[Dict[str, Any]] = []
    for case in summary.get("cases", []):
        case_dir = ROOT / case["result_dir"]
        meta = load_json(case_dir / "case_metadata.json")
        output = load_json(case_dir / "explainer_output.json") if (case_dir / "explainer_output.json").exists() else {}
        audit = parse_audit_report(output.get("Audit_report"))
        validation = output.get("validation_bundle", {})

        row = {
            "incident_name": meta["incident_name"],
            "chain": meta["chain"],
            "address": meta["address"],
            "bug_type": meta["bug_type"],
            "gold_bug_family": meta["gold_bug_family"],
            "profit": meta.get("profit"),
            "status": case["status"],
            "verdict": audit.get("verdict", ""),
            "confidence": confidence_from_output(output),
            "validation_status": validation.get("aggregate_status", ""),
            "validation_mode": validation.get("check_mode", ""),
            "vuln_type": output.get("vuln_type", ""),
            "validated_bug_type": output.get("validated_bug_type", ""),
            "violated_property": audit.get("violated_property", ""),
            "predicted_family": infer_predicted_family(output),
            "result_dir": case["result_dir"],
        }
        rows.append(row)
    return rows


def is_reasoning_positive(row: Dict[str, Any]) -> bool:
    """Return whether Final Reasoning retains an incident as a vulnerability candidate.

    ``Needs More Evidence`` is counted for IDR because the final LLM has not
    rejected the candidate; it denotes an unresolved incident that requires
    additional evidence rather than a confirmed-safe contract.
    """
    return row.get("verdict") in {"True Positive", "Needs More Evidence"}


def is_validation_positive(row: Dict[str, Any]) -> bool:
    return row.get("validation_status") == "violated"


def verdict_rank(verdict: str) -> int:
    """Rank incident-level audit outcomes before considering confidence."""
    return {
        "True Positive": 3,
        "Needs More Evidence": 2,
        "False Positive": 1,
    }.get((verdict or "").strip(), 0)


def choose_strongest(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None

    def sort_key(row: Dict[str, Any]):
        return (
            verdict_rank(row.get("verdict", "")),
            validation_rank(row.get("validation_status", "")),
            float(row.get("confidence") or 0.0),
        )

    return max(rows, key=sort_key)


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    rows = incident_case_rows(run_dir)
    incidents: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        incidents.setdefault(row["incident_name"], []).append(row)

    incident_rows = []
    reasoning_preds: List[int] = []
    validation_preds: List[int] = []
    root_matches_reasoning = 0
    root_matches_validation = 0
    detected_reasoning = 0
    detected_validation = 0
    profit_total = 0.0
    profit_reasoning = 0.0
    profit_validation = 0.0
    profit_annotated_incidents = 0

    for incident_name, addr_rows in sorted(incidents.items()):
        strongest = choose_strongest(addr_rows)
        if strongest is None:
            continue
        reasoning_hit = 1 if any(is_reasoning_positive(r) for r in addr_rows) else 0
        validation_hit = 1 if any(is_validation_positive(r) for r in addr_rows) else 0

        reasoning_preds.append(reasoning_hit)
        validation_preds.append(validation_hit)

        predicted_family = strongest.get("predicted_family", "unknown")
        gold_family = strongest.get("gold_bug_family", "unknown")
        root_match_reasoning = 1 if (reasoning_hit and predicted_family == gold_family) else 0
        root_match_validation = 1 if (validation_hit and predicted_family == gold_family) else 0

        detected_reasoning += reasoning_hit
        detected_validation += validation_hit
        root_matches_reasoning += root_match_reasoning
        root_matches_validation += root_match_validation

        profit = strongest.get("profit")
        if isinstance(profit, (int, float)):
            profit_annotated_incidents += 1
            profit_total += float(profit)
            if reasoning_hit:
                profit_reasoning += float(profit)
            if validation_hit:
                profit_validation += float(profit)

        incident_rows.append(
            {
                "incident_name": incident_name,
                "bug_type": strongest["bug_type"],
                "gold_bug_family": gold_family,
                "profit": profit,
                "num_addresses": len(addr_rows),
                "reasoning_detected": reasoning_hit,
                "validation_detected": validation_hit,
                "strongest_address": strongest["address"],
                "strongest_verdict": strongest["verdict"],
                "strongest_validation_status": strongest["validation_status"],
                "strongest_confidence": strongest["confidence"],
                "predicted_family": predicted_family,
                "root_match_reasoning": root_match_reasoning,
                "root_match_validation": root_match_validation,
                "result_dir": strongest["result_dir"],
            }
        )

    total_incidents = len(incident_rows)
    reasoning_root_match_rate = root_matches_reasoning / detected_reasoning if detected_reasoning else 0.0
    validation_root_match_rate = root_matches_validation / detected_validation if detected_validation else 0.0
    reasoning_profit_recall = profit_reasoning / profit_total if profit_total else None
    validation_profit_recall = profit_validation / profit_total if profit_total else None

    summary = {
        "run_dir": str(run_dir.relative_to(ROOT)),
        "num_address_cases": len(rows),
        "num_incidents": total_incidents,
        "profit_annotated_incidents": profit_annotated_incidents,
        "reasoning_level": {
            "detected_incidents": detected_reasoning,
            "total_incidents": total_incidents,
            "incident_recall": detected_reasoning / total_incidents if total_incidents else 0.0,
            "root_cause_match_rate": reasoning_root_match_rate,
        },
        "validation_grounded": {
            "detected_incidents": detected_validation,
            "total_incidents": total_incidents,
            "incident_recall": detected_validation / total_incidents if total_incidents else 0.0,
            "root_cause_match_rate": validation_root_match_rate,
        },
        "supplementary": {
            "profit_weighted_recall_reasoning": reasoning_profit_recall,
            "profit_weighted_recall_validation": validation_profit_recall,
        },
        "incidents": incident_rows,
        "address_rows": rows,
    }

    save_json(run_dir / "rw_benchmark_summary.json", summary)
    return summary


def write_csv(run_dir: Path, summary: Dict[str, Any]) -> None:
    address_rows = summary["address_rows"]
    incident_rows = summary["incidents"]

    address_csv = run_dir / "rw_address_results.csv"
    with address_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "incident_name",
                "chain",
                "address",
                "bug_type",
                "gold_bug_family",
                "profit",
                "status",
                "verdict",
                "confidence",
                "validation_status",
                "validation_mode",
                "vuln_type",
                "validated_bug_type",
                "violated_property",
                "predicted_family",
                "result_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(address_rows)

    incident_csv = run_dir / "rw_incident_results.csv"
    with incident_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "incident_name",
                "bug_type",
                "gold_bug_family",
                "profit",
                "num_addresses",
                "reasoning_detected",
                "validation_detected",
                "strongest_address",
                "strongest_verdict",
                "strongest_validation_status",
                "strongest_confidence",
                "predicted_family",
                "root_match_reasoning",
                "root_match_validation",
                "result_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(incident_rows)


def main():
    parser = argparse.ArgumentParser(description="Real-world benchmark runner for rw_dataset.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build-manifests", help="Build rw_dataset manifest files.")

    run_parser = sub.add_parser("run", help="Run a manifest and persist outputs.")
    run_parser.add_argument("--manifest", default="rw_dataset/manifests/rw_manifest_business_logic.json")
    run_parser.add_argument("--limit-incidents", type=int, default=0)
    run_parser.add_argument("--incident-name", default="")
    run_parser.add_argument("--all-addresses", action="store_true", help="Disable culprit-address selection and run every address in each incident.")
    run_parser.add_argument("--resume-run-dir", default="", help="Resume an existing run directory and skip completed cases.")

    sum_parser = sub.add_parser("summarize", help="Summarize an existing run.")
    sum_parser.add_argument("--run-dir", required=True)

    args = parser.parse_args()

    if args.cmd == "build-manifests":
        paths = write_manifests()
        for path in paths:
            print(path.relative_to(ROOT))
        return

    if args.cmd == "run":
        manifest_path = (ROOT / args.manifest).resolve() if not os.path.isabs(args.manifest) else Path(args.manifest)
        resume_run_dir = None
        if args.resume_run_dir:
            resume_run_dir = (ROOT / args.resume_run_dir).resolve() if not os.path.isabs(args.resume_run_dir) else Path(args.resume_run_dir)
        run_dir = run_manifest(
            manifest_path,
            limit_incidents=args.limit_incidents,
            incident_name=args.incident_name,
            all_addresses=args.all_addresses,
            resume_run_dir=resume_run_dir,
        )
        summary = summarize_run(run_dir)
        write_csv(run_dir, summary)
        print(json.dumps(summary["reasoning_level"], indent=2))
        print(json.dumps(summary["validation_grounded"], indent=2))
        print(f"Saved run to: {run_dir}")
        return

    if args.cmd == "summarize":
        run_dir = (ROOT / args.run_dir).resolve() if not os.path.isabs(args.run_dir) else Path(args.run_dir)
        summary = summarize_run(run_dir)
        write_csv(run_dir, summary)
        print(json.dumps(summary["reasoning_level"], indent=2))
        print(json.dumps(summary["validation_grounded"], indent=2))


if __name__ == "__main__":
    main()
