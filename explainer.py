import json
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Type

from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from llm_backend import build_chat_model

from dotenv import load_dotenv
load_dotenv()

FUSION_FILE_PATH = "fusion_output_agent.json"
RAG_FILE_PATH = "rag_output.json"
OUTPUT_FILE_PATH = "explainer_output.json"


def safe_read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Could not read JSON file {path}: {exc}")
        return {}


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        return None
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None


def find_function_blocks(code: str) -> List[Dict[str, Any]]:
    functions: List[Dict[str, Any]] = []
    pattern = re.compile(r"function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)[^{;]*\{", re.DOTALL)

    for match in pattern.finditer(code):
        name = match.group(1)
        params = match.group(2)
        body_start = match.end() - 1
        depth = 0
        end = body_start
        while end < len(code):
            if code[end] == "{":
                depth += 1
            elif code[end] == "}":
                depth -= 1
                if depth == 0:
                    break
            end += 1

        body = code[body_start + 1 : end]
        functions.append(
            {
                "name": name,
                "params": params,
                "body": body,
                "signature": match.group(0),
            }
        )

    return functions


def parse_params(params: str) -> List[Dict[str, str]]:
    parsed: List[Dict[str, str]] = []
    for raw in params.split(","):
        raw = raw.strip()
        if not raw:
            continue
        tokens = raw.split()
        if not tokens:
            continue
        name = tokens[-1].replace(",", "").strip()
        param_type = " ".join(tokens[:-1]).strip()
        parsed.append({"type": param_type, "name": name})
    return parsed


def infer_balance_mappings(code: str) -> List[str]:
    pattern = re.compile(
        r"mapping\s*\(\s*address\s*=>\s*uint(?:8|16|32|64|128|256)?\s*\)\s*(?:public|private|internal|external)?\s*([A-Za-z_][A-Za-z0-9_]*)",
        re.IGNORECASE,
    )
    return list(dict.fromkeys(pattern.findall(code)))


def infer_protocol_family(code: str) -> str:
    lowered = code.lower()
    if "mapping(address => uint256) public balances" in lowered or "transfer" in lowered or "mint" in lowered:
        return "token"
    if "deposit" in lowered and "withdraw" in lowered:
        return "vault"
    if "borrow" in lowered or "collateral" in lowered or "liquidat" in lowered:
        return "lending"
    if "reward" in lowered or "stake" in lowered:
        return "staking"
    return "unknown"


def build_property_context_block(context: Optional[Dict[str, Any]]) -> str:
    if not context:
        return ""

    lines: List[str] = []
    for label, key in [
        ("Incident name", "incident_name"),
        ("Incident bug type", "bug_type"),
        ("Coarse bug family", "gold_bug_family"),
        ("Contract name", "contract_name"),
        ("Main file", "main_file"),
        ("Source type", "source_type"),
        ("Compiler version", "compiler_version"),
    ]:
        value = context.get(key)
        if value:
            lines.append(f"- {label}: {value}")

    if not lines:
        return ""

    return "Incident and contract context:\n" + "\n".join(lines)


def has_self_transfer_guard(body: str, from_var: str, to_var: str) -> bool:
    guard_patterns = [
        rf"require\s*\(\s*{re.escape(from_var)}\s*!=\s*{re.escape(to_var)}\s*[,)]",
        rf"require\s*\(\s*{re.escape(to_var)}\s*!=\s*{re.escape(from_var)}\s*[,)]",
        rf"if\s*\(\s*{re.escape(from_var)}\s*==\s*{re.escape(to_var)}\s*\)\s*(?:return|revert)",
        rf"if\s*\(\s*{re.escape(to_var)}\s*==\s*{re.escape(from_var)}\s*\)\s*(?:return|revert)",
    ]
    return any(re.search(pattern, body) for pattern in guard_patterns)


def detect_transfermint(code: str) -> Optional[Dict[str, Any]]:
    mapping_names = infer_balance_mappings(code)
    if not mapping_names:
        return None

    functions = find_function_blocks(code)
    for fn in functions:
        params = parse_params(fn["params"])
        address_params = [p["name"] for p in params if "address" in p["type"]]
        amount_params = [p["name"] for p in params if "uint" in p["type"]]
        if len(address_params) < 2 or not amount_params:
            continue

        body = fn["body"]
        for mapping_name in mapping_names:
            for from_var in address_params:
                for to_var in address_params:
                    if from_var == to_var:
                        continue

                    read_from = re.search(
                        rf"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*{re.escape(mapping_name)}\s*\[\s*{re.escape(from_var)}\s*\]\s*;",
                        body,
                    )
                    read_to = re.search(
                        rf"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*{re.escape(mapping_name)}\s*\[\s*{re.escape(to_var)}\s*\]\s*;",
                        body,
                    )
                    if not (read_from and read_to):
                        continue

                    if has_self_transfer_guard(body, from_var, to_var):
                        continue

                    from_balance_var = read_from.group(1)
                    to_balance_var = read_to.group(1)

                    temp_assignments = {}
                    for temp_name, expr in re.findall(
                        r"(?:uint(?:8|16|32|64|128|256)?|int(?:8|16|32|64|128|256)?|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);",
                        body,
                    ):
                        temp_assignments[temp_name] = expr.strip()

                    write_from = re.search(
                        rf"{re.escape(mapping_name)}\s*\[\s*{re.escape(from_var)}\s*\]\s*=\s*([^;]+);",
                        body,
                    )
                    write_to = re.search(
                        rf"{re.escape(mapping_name)}\s*\[\s*{re.escape(to_var)}\s*\]\s*=\s*([^;]+);",
                        body,
                    )
                    if not (write_from and write_to):
                        continue

                    from_expr = write_from.group(1).strip()
                    to_expr = write_to.group(1).strip()
                    resolved_from_expr = temp_assignments.get(from_expr, from_expr)
                    resolved_to_expr = temp_assignments.get(to_expr, to_expr)

                    matched_amount = None
                    for amount_var in amount_params:
                        subtract_like = (
                            re.search(rf"\b{re.escape(from_balance_var)}\b\s*-\s*\b{re.escape(amount_var)}\b", resolved_from_expr)
                            or re.search(rf"\b{re.escape(from_balance_var)}\b\s*-=\s*\b{re.escape(amount_var)}\b", resolved_from_expr)
                            or re.search(rf"\b{re.escape(amount_var)}\b\s*<=?\s*\b{re.escape(from_balance_var)}\b", body)
                        )
                        add_like = (
                            re.search(rf"\b{re.escape(to_balance_var)}\b\s*\+\s*\b{re.escape(amount_var)}\b", resolved_to_expr)
                            or re.search(rf"\b{re.escape(to_balance_var)}\b\s*\+=\s*\b{re.escape(amount_var)}\b", resolved_to_expr)
                        )
                        if subtract_like and add_like:
                            matched_amount = amount_var
                            break

                    if not matched_amount:
                        continue

                    if write_from.start() > write_to.start():
                        continue

                    return {
                        "validator": "transfermint_alias_sensitive_check",
                        "status": "violated",
                        "bug_type": "TransferMint",
                        "function": fn["name"],
                        "mapping": mapping_name,
                        "property_name": "self_transfer_balance_conservation",
                        "property_statement": (
                            "If sender and receiver are the same address, the transfer should preserve both "
                            "the account balance and total supply."
                        ),
                        "scenario": {
                            "assumption": f"{from_var} == {to_var}",
                            "preconditions": [f"{mapping_name}[{from_var}] >= {matched_amount}", f"{matched_amount} > 0"],
                            "steps": [
                                f"Read {mapping_name}[{from_var}] into {from_balance_var}",
                                f"Read {mapping_name}[{to_var}] into {to_balance_var}",
                                f"Compute {from_expr} -> {resolved_from_expr}",
                                f"Write {mapping_name}[{from_var}] = {from_expr}",
                                f"Compute {to_expr} -> {resolved_to_expr}",
                                f"Write {mapping_name}[{to_var}] = {to_expr}",
                            ],
                        },
                        "witness": {
                            "symbolic_initial_state": f"{from_balance_var} = {to_balance_var} = B",
                            "symbolic_final_state": f"{mapping_name}[{from_var}] = B + {matched_amount}",
                            "violation": f"Final balance increases by {matched_amount} under self-transfer aliasing.",
                        },
                        "confidence": 0.98,
                    }

    return None


def infer_check_mode(validator_specs: List[Dict[str, Any]]) -> str:
    modes = {spec.get("validation_mode", "").strip().lower() for spec in validator_specs if spec.get("validation_mode")}
    if not modes:
        return "none"
    if len(modes) == 1:
        return next(iter(modes))
    return "mixed"


def build_inconclusive_check_result(
    spec: Dict[str, Any],
    note: str,
    evidence_type: str = "none",
    evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "validator_name": spec.get("validator_name", "unknown_validator"),
        "validation_mode": spec.get("validation_mode", "invariant"),
        "status": "inconclusive",
        "evidence_type": evidence_type,
        "evidence": evidence or {},
        "notes": note,
    }


def build_supported_check_result(
    spec: Dict[str, Any],
    note: str,
    evidence_type: str = "structural_support",
    evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "validator_name": spec.get("validator_name", "unknown_validator"),
        "validation_mode": spec.get("validation_mode", "invariant"),
        "status": "supported",
        "evidence_type": evidence_type,
        "evidence": evidence or {},
        "notes": note,
    }


def find_function_by_name(code: str, name: str) -> Optional[Dict[str, Any]]:
    for fn in find_function_blocks(code):
        if fn["name"] == name:
            return fn
    return None


def extract_require_conditions(body: str) -> List[str]:
    return [cond.strip() for cond in re.findall(r"require\s*\(\s*(.+?)\s*(?:,\s*[^)]*)?\)\s*;", body, re.DOTALL)]


def contains_any(text: str, needles: List[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def extract_identifier_hints(values: List[str]) -> List[str]:
    stopwords = {
        "true",
        "false",
        "before",
        "after",
        "state",
        "contract",
        "balance",
        "amount",
        "value",
        "sender",
        "receiver",
        "owner",
        "caller",
        "user",
        "msg",
        "block",
        "timestamp",
        "this",
        "address",
        "uint",
        "int",
        "bool",
        "returns",
        "must",
        "remain",
        "remains",
        "unchanged",
        "equal",
        "same",
    }
    hints: List[str] = []
    for value in values:
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", value or ""):
            lowered = token.lower()
            if lowered not in stopwords and len(lowered) > 1:
                hints.append(lowered)
    return list(dict.fromkeys(hints))


def infer_relation_intent(spec: Dict[str, Any]) -> str:
    assertion = spec.get("assertion", {}) if isinstance(spec.get("assertion"), dict) else {}
    relation_type = str(assertion.get("relation_type", "")).strip().lower()
    text = " ".join(
        [
            str(spec.get("property_name", "")),
            str(spec.get("focus_condition", "")),
            str(assertion.get("expected_relation", "")),
            str(spec.get("failure_interpretation", "")),
        ]
    ).lower()

    if contains_any(text, ["owner", "authorized", "authorize", "permission", "admin", "forbidden", "access control"]):
        return "authorization"
    if contains_any(text, ["equivalent", "same outcome", "same result", "split", "merge", "round-trip", "round trip"]):
        return "equivalence"
    if contains_any(text, ["once", "one-time", "consumed", "claim", "redeem", "refund", "nonce", "vote", "replay"]):
        return "single_use"
    if contains_any(text, ["before", "after", "release", "unlock", "maturity", "phase", "finalize", "commit", "order"]):
        return "ordering"
    if contains_any(text, ["bound", "cap", "limit", "threshold", "collateral", "debt", "ratio", "exceed", "<=", ">="]):
        return "bound"
    if contains_any(text, ["unchanged", "constant", "preserve", "preservation", "remain", "same"]):
        return "preservation"
    if relation_type in {"preservation", "bound", "single_use", "ordering", "equivalence"}:
        return relation_type
    return "custom"


def extract_modifiers(signature: str) -> List[str]:
    return re.findall(r"\)\s*(.*?)\{", signature, re.DOTALL)


def extract_state_writes(body: str, tracked_state: List[str]) -> List[Dict[str, str]]:
    hints = extract_identifier_hints(tracked_state)
    writes: List[Dict[str, str]] = []
    for lhs, op, rhs in re.findall(r"([A-Za-z_][A-Za-z0-9_\.\[\]]*)\s*(\+=|-=|\*=|/=|=)\s*([^;]+);", body):
        lowered_lhs = lhs.lower()
        if any(hint in lowered_lhs for hint in hints):
            writes.append({"lhs": lhs.strip(), "op": op.strip(), "rhs": rhs.strip()})
    return writes


def analyze_functions(code: str, target_functions: List[str], tracked_state: List[str]) -> List[Dict[str, Any]]:
    analyses: List[Dict[str, Any]] = []
    for fn_name in target_functions:
        fn = find_function_by_name(code, fn_name)
        if not fn:
            continue
        body = fn["body"]
        requires = extract_require_conditions(body)
        modifiers = extract_modifiers(fn["signature"])
        writes = extract_state_writes(body, tracked_state)
        analyses.append(
            {
                "name": fn["name"],
                "signature": fn["signature"],
                "body": body,
                "requires": requires,
                "modifiers": modifiers,
                "writes": writes,
            }
        )
    return analyses


def gather_relation_hints(spec: Dict[str, Any]) -> Dict[str, Any]:
    assertion = spec.get("assertion", {}) if isinstance(spec.get("assertion"), dict) else {}
    text_fields = [
        str(spec.get("focus_condition", "")),
        str(assertion.get("expected_relation", "")),
        *[str(x) for x in spec.get("tracked_state", []) if isinstance(x, str)],
        *[str(x) for x in spec.get("preconditions", []) if isinstance(x, str)],
    ]
    identifiers = extract_identifier_hints(text_fields)
    comparators = [token for token in ["<=", ">=", "<", ">", "==", "!="] if any(token in field for field in text_fields)]
    return {"identifiers": identifiers, "comparators": comparators}


def has_guard_evidence(analyses: List[Dict[str, Any]], hint_tokens: List[str]) -> bool:
    for analysis in analyses:
        haystack = " ".join(analysis.get("requires", []) + analysis.get("modifiers", [])).lower()
        if all(token in haystack for token in hint_tokens if token):
            return True
        if any(token in haystack for token in hint_tokens if token):
            return True
    return False


def summarize_writes(analyses: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
    summary: Dict[str, List[Dict[str, str]]] = {}
    for analysis in analyses:
        for write in analysis.get("writes", []):
            summary.setdefault(write["lhs"].lower(), []).append(write)
    return summary


def parse_sequence_function_names(sequence: List[str]) -> List[str]:
    names: List[str] = []
    for step in sequence:
        match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", step)
        if match:
            names.append(match.group(1))
    return names


class DummyInput(BaseModel):
    pass


class ExplainerTool(BaseTool):
    name: str = "ExplainerTool"
    description: str = (
        "Verify predicted vulnerabilities via LLM-guided property synthesis, adversarial scenario generation, "
        "and lightweight validation before producing the final audit report."
    )
    args_schema: Type[BaseModel] = DummyInput
    _llm: Optional[ChatOpenAI] = None

    def __init__(self, llm: Optional[ChatOpenAI] = None, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm

    def _call_llm(self, prompt_text: str) -> str:
        if self._llm is None:
            raise RuntimeError("No LLM provided to ExplainerTool.")
        errors = []
        for method in ["invoke", "generate", "call"]:
            try:
                fn = getattr(self._llm, method, None)
                if callable(fn):
                    resp = fn(prompt_text)
                    if isinstance(resp, str):
                        return resp
                    if hasattr(resp, "content"):
                        return resp.content
                    if hasattr(resp, "text"):
                        return resp.text
                    return str(resp)
            except Exception as exc:
                errors.append(f"{method} failed: {exc}")
        raise RuntimeError("All LLM invocation attempts failed: " + " | ".join(errors))

    def _synthesize_properties(
        self,
        code: str,
        vuln_type: str,
        predicted_line: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        protocol_hint = infer_protocol_family(code)
        context_block = build_property_context_block(context)
        prompt = f"""
You are a smart contract security analyst specialized in protocol intent extraction.
Infer the financial or business properties that should hold for the contract below.

Return ONLY valid JSON with this schema:
{{
  "protocol_family": string,
  "critical_state": [string],
  "critical_functions": [string],
  "properties": [
    {{
      "name": string,
      "type": "invariant" | "equivalence",
      "statement": string,
      "scope": [string],
      "actors": [string],
      "preconditions": [string],
      "postcondition": string
    }}
  ],
  "reasoning_summary": string
}}

Prior detector signal:
- Predicted vulnerability: {vuln_type}
- Predicted location: {predicted_line}
- Protocol family hint: {protocol_hint}

{context_block}

Instructions:
- Derive the properties from the contract semantics and the incident context above, not from generic ERC20 templates alone.
- Do not default to self-transfer, total-supply, or generic balance-preservation properties unless the contract logic directly centers on those behaviors.
- Prefer the smallest set of high-relevance properties over a longer generic list.
- Each property should describe a protocol rule that would be materially relevant to the likely failure mode of this contract.
- Do not invent arbitrary thresholds, actors, or transaction sequences that are not supported by the code or the provided context.

Contract:
```solidity
{code}
```
"""
        raw = self._call_llm(prompt)
        parsed = extract_json_object(raw)
        if parsed:
            return parsed
        return {
            "protocol_family": "unknown",
            "critical_state": [],
            "critical_functions": [],
            "properties": [],
            "reasoning_summary": raw.strip(),
        }

    def _generate_scenarios(
        self,
        code: str,
        property_bundle: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        protocol_hint = infer_protocol_family(code)
        context_block = build_property_context_block(context)
        prompt = f"""
You are an adversarial scenario generator for smart contract security testing.
Given the contract and candidate protocol properties, synthesize minimal transaction scenarios that could falsify them.

Return ONLY valid JSON with this schema:
{{
  "scenarios": [
    {{
      "property_name": string,
      "target_function": string,
      "goal": string,
      "steps": [string],
      "expected_postcondition": string,
      "focus_condition": string
    }}
  ]
}}

Properties:
{json.dumps(property_bundle, ensure_ascii=False, indent=2)}

Protocol family hint: {protocol_hint}
{context_block}

Instructions:
- Generate scenarios only for the inferred properties above; do not introduce unrelated generic token misuse patterns.
- Prefer short, executable, contract-relevant transaction sequences over broad hypothetical exploit stories.
- Use concrete but valid values and actors. Never invent impossible numeric domains such as negative uint inputs.
- In `expected_postcondition`, describe the SAFE behavior that should hold if the property is respected.
- Let the inferred properties drive the scenario shape. Do not default to self-transfer or ERC20-style templates unless the synthesized property explicitly calls for them.

Contract:
```solidity
{code}
```
"""
        raw = self._call_llm(prompt)
        parsed = extract_json_object(raw)
        if parsed:
            return parsed
        return {"scenarios": [], "raw_generation": raw.strip()}

    def _generate_validator_specs(
        self,
        code: str,
        property_bundle: Dict[str, Any],
        scenario_bundle: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        protocol_hint = infer_protocol_family(code)
        context_block = build_property_context_block(context)
        prompt = f"""
You are a validator-spec synthesis engine for smart contract security analysis.
Given a contract, a set of inferred properties, and adversarial scenarios, synthesize validator specifications.

Return ONLY valid JSON with this schema:
{{
  "validator_specs": [
    {{
      "validator_name": string,
      "validation_mode": "invariant" | "equivalence",
      "property_name": string,
      "target_functions": [string],
      "tracked_state": [string],
      "preconditions": [string],
      "focus_condition": string,
      "reference_sequence": [string],
      "comparison_sequence": [string],
      "assertion": {{
        "relation_type": "preservation" | "bound" | "single_use" | "ordering" | "equivalence" | "custom",
        "expected_relation": string,
        "before_state": [string],
        "after_state": [string]
      }},
      "failure_interpretation": string
    }}
  ],
  "selection_summary": string
}}

Properties:
{json.dumps(property_bundle, ensure_ascii=False, indent=2)}

Scenarios:
{json.dumps(scenario_bundle, ensure_ascii=False, indent=2)}

Protocol family hint: {protocol_hint}
{context_block}

Instructions:
- Synthesize validator specifications strictly from the inferred properties and generated scenarios above.
- Do not inject generic validator templates that are not clearly supported by the property bundle.
- Use `validation_mode = equivalence` only when the property truly compares two semantically equivalent sequences; otherwise use `invariant`.
- The assertion must describe the SAFE expected relation, not the buggy outcome.
- Prefer tracked state and preconditions that are directly visible in the contract or explicitly named by the synthesized properties.

Contract:
```solidity
{code}
```
"""
        raw = self._call_llm(prompt)
        parsed = extract_json_object(raw)
        if parsed:
            return parsed
        return {"validator_specs": [], "selection_summary": raw.strip()}

    def _run_invariant_check(self, code: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        spec_text = json.dumps(spec, ensure_ascii=False).lower()
        target_functions = [fn for fn in spec.get("target_functions", []) if isinstance(fn, str)]
        tracked_state = spec.get("tracked_state", [])
        assertion = spec.get("assertion", {}) if isinstance(spec.get("assertion"), dict) else {}
        relation_intent = infer_relation_intent(spec)
        focus_condition = str(spec.get("focus_condition", ""))
        analyses = analyze_functions(code, target_functions, tracked_state)
        relation_hints = gather_relation_hints(spec)
        hint_tokens = relation_hints["identifiers"]
        write_summary = summarize_writes(analyses)

        is_self_transfer_conservation = any(
            token in spec_text
            for token in [
                "self_transfer",
                "self-transfer",
                "transfermint",
                "from == to",
                "remains unchanged",
                "preserve the account balance",
            ]
        )

        if is_self_transfer_conservation:
            finding = detect_transfermint(code)
            if finding:
                return {
                    "validator_name": spec.get("validator_name", "self_transfer_conservation"),
                    "validation_mode": "invariant",
                    "status": "violated",
                    "evidence_type": "symbolic_witness",
                    "evidence": finding,
                    "notes": "Invariant violated under alias-sensitive self-transfer checking.",
                }
            transfer_fn = None
            for fn_name in target_functions:
                transfer_fn = find_function_by_name(code, fn_name)
                if transfer_fn:
                    break
            if transfer_fn and has_self_transfer_guard(transfer_fn["body"], "from", "to"):
                return build_supported_check_result(
                    spec,
                    note="Self-transfer is explicitly guarded or short-circuited, preserving the invariant.",
                    evidence={"function": transfer_fn["name"]},
                )
            if transfer_fn and contains_any(transfer_fn["body"], ["if (from == to)", "if(from == to)", "return true"]):
                return build_supported_check_result(
                    spec,
                    note="Self-transfer path is handled as a no-op in the transfer function.",
                    evidence={"function": transfer_fn["name"]},
                )

        if relation_intent == "authorization":
            auth_tokens = ["owner", "authorized", "permission", "admin"]
            has_auth_guard = has_guard_evidence(analyses, auth_tokens)
            if has_auth_guard:
                return build_supported_check_result(
                    spec,
                    note="Sensitive function appears to be protected by ownership or authorization checks.",
                    evidence={
                        "require_conditions": [req for analysis in analyses for req in analysis.get("requires", [])],
                        "modifiers": [mod for analysis in analyses for mod in analysis.get("modifiers", [])],
                    },
                )
            if any(analysis.get("writes") for analysis in analyses):
                return {
                    "validator_name": spec.get("validator_name", "authorization_validator"),
                    "validation_mode": "invariant",
                    "status": "violated",
                    "evidence_type": "structural_witness",
                    "evidence": {
                        "target_functions": target_functions,
                        "writes": [write for analysis in analyses for write in analysis.get("writes", [])],
                        "require_conditions": [req for analysis in analyses for req in analysis.get("requires", [])],
                    },
                    "notes": "Sensitive state updates are present without an ownership or authorization guard.",
                }
        elif relation_intent == "bound":
            requires = [req.lower() for analysis in analyses for req in analysis.get("requires", [])]
            has_comparator = any(op in req for req in requires for op in ["<=", ">=", "<", ">"])
            has_bound_refs = any(any(token in req for token in hint_tokens) for req in requires)
            if has_comparator and has_bound_refs:
                return build_supported_check_result(
                    spec,
                    note="State-dependent bound guard is present around the target function.",
                    evidence={"require_conditions": requires},
                )
            if any(analysis.get("writes") for analysis in analyses):
                return {
                    "validator_name": spec.get("validator_name", "bound_validator"),
                    "validation_mode": "invariant",
                    "status": "violated",
                    "evidence_type": "structural_witness",
                    "evidence": {
                        "require_conditions": requires,
                        "writes": [write for analysis in analyses for write in analysis.get("writes", [])],
                    },
                    "notes": "Tracked state is updated without a clear state-dependent bound check tied to the synthesized relation.",
                }
        elif relation_intent == "single_use":
            writes = [write for analysis in analyses for write in analysis.get("writes", [])]
            has_consume = any(
                write["op"] in {"-=", "="}
                and (
                    re.search(r"\b0\b", write["rhs"])
                    or re.search(r"\btrue\b", write["rhs"].lower())
                    or any(token in write["rhs"].lower() for token in ["- 1", "-1", "sub("])
                )
                for write in writes
            )
            if has_consume:
                return build_supported_check_result(
                    spec,
                    note="Tracked entitlement state appears to be consumed or marked on use.",
                    evidence={"writes": writes},
                )
            if writes or target_functions:
                return {
                    "validator_name": spec.get("validator_name", "single_use_validator"),
                    "validation_mode": "invariant",
                    "status": "violated",
                    "evidence_type": "structural_witness",
                    "evidence": {"writes": writes, "target_functions": target_functions},
                    "notes": "The target flow does not show a clear one-time-consumption update for the tracked entitlement state.",
                }
        elif relation_intent == "ordering":
            requires = [req.lower() for analysis in analyses for req in analysis.get("requires", [])]
            ordering_tokens = [token for token in hint_tokens if token in {"unlocktime", "releasetime", "highestbid", "commit", "phase", "timestamp", "maturity"}]
            has_phase_guard = any(any(token in req for token in ordering_tokens) and any(op in req for op in ["<=", ">=", "<", ">", "==", "!="]) for req in requires)
            if has_phase_guard:
                return build_supported_check_result(
                    spec,
                    note="Target function includes a phase/time/order guard aligned with the synthesized relation.",
                    evidence={"require_conditions": requires},
                )
            if any(analysis.get("writes") for analysis in analyses):
                return {
                    "validator_name": spec.get("validator_name", "ordering_validator"),
                    "validation_mode": "invariant",
                    "status": "violated",
                    "evidence_type": "structural_witness",
                    "evidence": {"require_conditions": requires, "target_functions": target_functions},
                    "notes": "State transition appears writable without a relevant phase or time guard.",
                }
        elif relation_intent == "preservation":
            if not any(analysis.get("writes") for analysis in analyses):
                return build_supported_check_result(
                    spec,
                    note="Tracked state is not directly mutated in the target function, which structurally supports preservation.",
                    evidence={"target_functions": target_functions},
                )
            if contains_any(focus_condition, ["=="]) and any(len(writes) > 1 for writes in write_summary.values()):
                return {
                    "validator_name": spec.get("validator_name", "preservation_validator"),
                    "validation_mode": "invariant",
                    "status": "violated",
                    "evidence_type": "structural_witness",
                    "evidence": {
                        "writes": [write for analysis in analyses for write in analysis.get("writes", [])],
                        "focus_condition": focus_condition,
                    },
                    "notes": "Tracked state receives multiple writes under an aliasing-like focus condition, contradicting a preservation relation.",
                }

        if target_functions or tracked_state or assertion:
            return build_inconclusive_check_result(
                spec,
                note=(
                    "Invariant spec synthesized successfully, but relation-guided checking could not yet derive a decisive "
                    "supported or violated conclusion from the target guards and state updates."
                ),
                evidence_type="spec_only",
                evidence={
                    "target_functions": spec.get("target_functions", []),
                    "tracked_state": tracked_state,
                    "relation_type": relation_intent,
                    "expected_relation": assertion.get("expected_relation", ""),
                },
            )

        return build_inconclusive_check_result(
            spec,
            note="Invariant spec was too underspecified for lightweight checking.",
        )

    def _run_bound_check(self, code: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        spec_text = json.dumps(spec, ensure_ascii=False).lower()
        target_functions = [fn for fn in spec.get("target_functions", []) if isinstance(fn, str)]
        body = ""
        fn = None
        for fn_name in target_functions:
            fn = find_function_by_name(code, fn_name)
            if fn:
                body = fn["body"]
                break
        lowered_body = body.lower()
        requires = [r.lower() for r in extract_require_conditions(body)]

        if contains_any(" ".join(target_functions), ["borrow"]) or contains_any(spec_text, ["borrow", "debt", "oracle"]):
            if "updatedat" in code.lower() and "oracleprice" in code.lower():
                has_staleness_guard = any("updatedat" in req and ("block.timestamp" in req or "maxoracledelay" in req) for req in requires)
                if not has_staleness_guard:
                    return {
                        "validator_name": spec.get("validator_name", "oracle_freshness_bound"),
                        "validation_mode": "invariant",
                        "status": "violated",
                        "evidence_type": "structural_witness",
                        "evidence": {
                            "relation_type": "bound",
                            "issue": "Borrow path depends on oracle price without freshness guard.",
                            "target_functions": target_functions,
                            "require_conditions": requires,
                        },
                        "notes": "State-dependent financial bound violated because stale oracle protection is absent.",
                    }
                return build_supported_check_result(
                    spec,
                    note="Oracle freshness guard is present before the borrowing bound is enforced.",
                    evidence={"require_conditions": requires},
                )

            has_allowed_bound = any("allowed" in req and ("amount" in req or "debt" in req) for req in requires)
            has_allowed_from_collateral = bool(re.search(r"\ballowed\b\s*=\s*[^;]*collateral", lowered_body))
            has_collateral_guard = any("collateral" in req and ("amount" in req or "debt" in req) for req in requires) or (
                has_allowed_bound and has_allowed_from_collateral
            )
            if not has_collateral_guard:
                return {
                    "validator_name": spec.get("validator_name", "financial_bound_validator"),
                    "validation_mode": "invariant",
                    "status": "violated",
                    "evidence_type": "structural_witness",
                    "evidence": {
                        "relation_type": "bound",
                        "issue": "Financial bound check does not reference collateral state.",
                        "target_functions": target_functions,
                        "require_conditions": requires,
                    },
                    "notes": "Bound relation violated because the authorization check is disconnected from borrower collateral state.",
                }
            if any("allowed >= amount" in req for req in requires) and not has_allowed_from_collateral:
                return {
                    "validator_name": spec.get("validator_name", "financial_bound_validator"),
                    "validation_mode": "invariant",
                    "status": "violated",
                    "evidence_type": "structural_witness",
                    "evidence": {
                        "relation_type": "bound",
                        "issue": "Borrow bound is derived from requested amount rather than collateral state.",
                        "require_conditions": requires,
                    },
                    "notes": "Bound relation violated due to state-disconnected borrowing capacity logic.",
                }
            return build_supported_check_result(
                spec,
                note="Borrowing or liquidation path includes a state-dependent financial bound guard.",
                evidence={"require_conditions": requires},
            )

        if contains_any(spec_text, ["liquidat", "liquidation"]):
            comparison_text = " ".join(requires)
            if any(op in comparison_text for op in [">=", "<="]) and "collateral" in comparison_text and "debt" in comparison_text:
                if ">=" in comparison_text or "<=" in comparison_text:
                    return {
                        "validator_name": spec.get("validator_name", "liquidation_bound_validator"),
                        "validation_mode": "invariant",
                        "status": "violated",
                        "evidence_type": "structural_witness",
                        "evidence": {"require_conditions": requires},
                        "notes": "Liquidation condition appears to allow liquidation of healthy positions.",
                    }
            if any("< debt" in req or "debt" in req and "<" in req for req in requires):
                return build_supported_check_result(
                    spec,
                    note="Liquidation is guarded by an undercollateralization check.",
                    evidence={"require_conditions": requires},
                )

        if contains_any(spec_text, ["cap", "totaldeposits"]):
            if any("totaldeposits + msg.value" in req and "<= cap" in req for req in requires):
                return build_supported_check_result(
                    spec,
                    note="Deposit cap is checked against cumulative total deposits.",
                    evidence={"require_conditions": requires},
                )
            if any("msg.value <= cap" in req for req in requires):
                return {
                    "validator_name": spec.get("validator_name", "cap_bound_validator"),
                    "validation_mode": "invariant",
                    "status": "violated",
                    "evidence_type": "structural_witness",
                    "evidence": {"require_conditions": requires},
                    "notes": "Cap check constrains only the incoming deposit and ignores cumulative protocol state.",
                }

        return build_inconclusive_check_result(
            spec,
            note="Financial bound spec was recognized, but no decisive structural relation could be established.",
            evidence_type="spec_only",
            evidence={"require_conditions": requires},
        )

    def _run_single_use_check(self, code: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        spec_text = json.dumps(spec, ensure_ascii=False).lower()
        target_functions = [fn for fn in spec.get("target_functions", []) if isinstance(fn, str)]
        body = ""
        for fn_name in target_functions:
            fn = find_function_by_name(code, fn_name)
            if fn:
                body = fn["body"]
                break
        lowered_body = body.lower()
        requires = [r.lower() for r in extract_require_conditions(body)]

        if "reward" in spec_text and "claim" in "".join(target_functions).lower():
            if "rewardaccrued[msg.sender] = 0" in lowered_body or "rewardaccrued[msg.sender]=0" in lowered_body:
                return build_supported_check_result(spec, "Reward entitlement is consumed after claim.", evidence={"body": body})
            return {
                "validator_name": spec.get("validator_name", "single_use_entitlement_validator"),
                "validation_mode": "invariant",
                "status": "violated",
                "evidence_type": "structural_witness",
                "evidence": {"issue": "Reward claim path does not consume accrued reward.", "body": body},
                "notes": "Single-use entitlement violated because reward state remains reusable after claim.",
            }

        if "redeem" in "".join(target_functions).lower() or "credit" in spec_text:
            if "credits[msg.sender] -=" in lowered_body or "credits[msg.sender]=" in lowered_body:
                return build_supported_check_result(spec, "Credits are reduced on redemption.", evidence={"body": body})
            return {
                "validator_name": spec.get("validator_name", "single_use_entitlement_validator"),
                "validation_mode": "invariant",
                "status": "violated",
                "evidence_type": "structural_witness",
                "evidence": {"issue": "Redeem path does not consume credits.", "body": body},
                "notes": "Single-use entitlement violated because redemption does not consume the remaining credits.",
            }

        if "refund" in "".join(target_functions).lower() or "deposit" in spec_text:
            if "deposits[msg.sender] = 0" in lowered_body or "deposits[msg.sender]=0" in lowered_body:
                return build_supported_check_result(spec, "Refund path clears the refundable balance.", evidence={"body": body})
            return {
                "validator_name": spec.get("validator_name", "single_use_entitlement_validator"),
                "validation_mode": "invariant",
                "status": "violated",
                "evidence_type": "structural_witness",
                "evidence": {"issue": "Refund path leaves refundable balance reusable.", "body": body},
                "notes": "Single-use entitlement violated because the refundable balance is not consumed.",
            }

        if "vote" in "".join(target_functions).lower() or "hasvoted" in code.lower():
            has_guard = any("hasvoted" in req and "!" in req for req in requires)
            has_consume = "hasvoted[" in lowered_body and "= true" in lowered_body
            if has_guard and has_consume:
                return build_supported_check_result(spec, "Voting path prevents duplicate votes through guard and consumption.", evidence={"require_conditions": requires})
            return {
                "validator_name": spec.get("validator_name", "single_use_entitlement_validator"),
                "validation_mode": "invariant",
                "status": "violated",
                "evidence_type": "structural_witness",
                "evidence": {"require_conditions": requires, "body": body},
                "notes": "Single-use entitlement violated because the voting path does not reliably prevent double voting.",
            }

        if "message" in spec_text or "nonce" in spec_text or "bridge" in spec_text:
            has_guard = any("consumedmessage" in req and "!" in req for req in requires)
            has_consume = "consumedmessage[" in lowered_body and "= true" in lowered_body
            if has_guard and has_consume:
                return build_supported_check_result(spec, "Bridge message is consumed on first finalization.", evidence={"require_conditions": requires})
            return {
                "validator_name": spec.get("validator_name", "single_use_entitlement_validator"),
                "validation_mode": "invariant",
                "status": "violated",
                "evidence_type": "structural_witness",
                "evidence": {"require_conditions": requires, "body": body},
                "notes": "Single-use entitlement violated because the bridge message can be finalized repeatedly.",
            }

        return build_inconclusive_check_result(
            spec,
            note="Single-use entitlement spec was recognized, but the current checker could not map it to a supported state-consumption pattern.",
            evidence_type="spec_only",
            evidence={"require_conditions": requires},
        )

    def _run_ordering_check(self, code: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        spec_text = json.dumps(spec, ensure_ascii=False).lower()
        target_functions = [fn for fn in spec.get("target_functions", []) if isinstance(fn, str)]
        body = ""
        for fn_name in target_functions:
            fn = find_function_by_name(code, fn_name)
            if fn:
                body = fn["body"]
                break
        lowered_body = body.lower()
        requires = [r.lower() for r in extract_require_conditions(body)]

        if "finalize" in "".join(target_functions).lower():
            if any("highestbid > 0" in req or "no bids" in req for req in requires):
                return build_supported_check_result(spec, "Finalize path is gated by prior bidding state.", evidence={"require_conditions": requires})
            return {
                "validator_name": spec.get("validator_name", "ordering_validator"),
                "validation_mode": "invariant",
                "status": "violated",
                "evidence_type": "structural_witness",
                "evidence": {"require_conditions": requires, "body": body},
                "notes": "Ordering relation violated because finalization is not gated by a valid prior phase.",
            }

        if "releasetime" in spec_text or "vesting" in spec_text or "release" in "".join(target_functions).lower():
            if any("block.timestamp >=" in req and "releasetime" in req for req in requires):
                return build_supported_check_result(spec, "Release path is time-gated by releaseTime.", evidence={"require_conditions": requires})
            return {
                "validator_name": spec.get("validator_name", "ordering_validator"),
                "validation_mode": "invariant",
                "status": "violated",
                "evidence_type": "structural_witness",
                "evidence": {"require_conditions": requires, "body": body},
                "notes": "Ordering relation violated because release is not gated by vesting maturity.",
            }

        if "unlocktime" in spec_text or "locked" in spec_text or "withdraw" in "".join(target_functions).lower():
            if any("block.timestamp >=" in req and "unlocktime" in req for req in requires):
                return build_supported_check_result(spec, "Withdraw path is time-gated by unlockTime.", evidence={"require_conditions": requires})
            return {
                "validator_name": spec.get("validator_name", "ordering_validator"),
                "validation_mode": "invariant",
                "status": "violated",
                "evidence_type": "structural_witness",
                "evidence": {"require_conditions": requires, "body": body},
                "notes": "Ordering relation violated because withdrawal is not gated by the lock expiry condition.",
            }

        return build_inconclusive_check_result(
            spec,
            note="Ordering spec was recognized, but no decisive phase or time guard could be extracted.",
            evidence_type="spec_only",
            evidence={"require_conditions": requires},
        )

    def _run_equivalence_check(self, code: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        reference_sequence = spec.get("reference_sequence", [])
        comparison_sequence = spec.get("comparison_sequence", [])
        assertion = spec.get("assertion", {}) if isinstance(spec.get("assertion"), dict) else {}
        tracked_state = spec.get("tracked_state", [])

        reference_functions = parse_sequence_function_names(reference_sequence)
        comparison_functions = parse_sequence_function_names(comparison_sequence)
        reference_analyses = analyze_functions(code, reference_functions, tracked_state)
        comparison_analyses = analyze_functions(code, comparison_functions, tracked_state)

        reference_writes = [write for analysis in reference_analyses for write in analysis.get("writes", [])]
        comparison_writes = [write for analysis in comparison_analyses for write in analysis.get("writes", [])]

        def effect_signature(writes: List[Dict[str, str]]) -> List[Tuple[str, str, str]]:
            return sorted((write["lhs"].lower(), write["op"], re.sub(r"\s+", "", write["rhs"].lower())) for write in writes)

        ref_sig = effect_signature(reference_writes)
        cmp_sig = effect_signature(comparison_writes)
        ref_text = " ".join(write["rhs"].lower() for write in reference_writes)
        cmp_text = " ".join(write["rhs"].lower() for write in comparison_writes)

        bonus_like = any(token in (ref_text + " " + cmp_text) for token in ["bonus", "?0:5", "?5:0", "+1", "+ 1"])
        asymmetry = ref_sig != cmp_sig

        if reference_sequence and comparison_sequence and reference_writes and comparison_writes:
            if bonus_like or asymmetry:
                return {
                    "validator_name": spec.get("validator_name", "equivalence_validator"),
                    "validation_mode": "equivalence",
                    "status": "violated",
                    "evidence_type": "structural_witness",
                    "evidence": {
                        "reference_sequence": reference_sequence,
                        "comparison_sequence": comparison_sequence,
                        "reference_effects": ref_sig,
                        "comparison_effects": cmp_sig,
                        "expected_relation": assertion.get("expected_relation", ""),
                    },
                    "notes": "Compared sequences induce structurally different tracked-state effects, contradicting the equivalence relation.",
                }
            return build_supported_check_result(
                spec,
                "Compared sequences induce matching tracked-state effects under relation-guided checking.",
                evidence={
                    "reference_sequence": reference_sequence,
                    "comparison_sequence": comparison_sequence,
                    "effect_signature": ref_sig,
                },
            )

        if reference_sequence and comparison_sequence:
            return build_inconclusive_check_result(
                spec,
                note=(
                    "Equivalence spec synthesized successfully, but relation-guided checking could not derive comparable "
                    "tracked-state effects for both sequences."
                ),
                evidence_type="spec_only",
                evidence={
                    "reference_sequence": reference_sequence,
                    "comparison_sequence": comparison_sequence,
                    "expected_relation": assertion.get("expected_relation", ""),
                },
            )

        return build_inconclusive_check_result(
            spec,
            note="Equivalence spec was too underspecified for lightweight checking.",
        )

    def _lightweight_validate(
        self,
        code: str,
        property_bundle: Dict[str, Any],
        scenario_bundle: Dict[str, Any],
        validator_spec_bundle: Dict[str, Any],
        vuln_type: str,
    ) -> Dict[str, Any]:
        validator_specs = validator_spec_bundle.get("validator_specs", [])
        check_results: List[Dict[str, Any]] = []
        executors_used: List[str] = []

        for spec in validator_specs:
            mode = str(spec.get("validation_mode", "invariant")).strip().lower()
            if mode == "invariant":
                executors_used.append("invariant_check")
                check_results.append(self._run_invariant_check(code, spec))
            elif mode == "equivalence":
                executors_used.append("equivalence_check")
                check_results.append(self._run_equivalence_check(code, spec))
            else:
                check_results.append(
                    build_inconclusive_check_result(
                        spec,
                        note=f"Unsupported validation_mode: {mode}",
                    )
                )

        aggregate_status = "inconclusive"
        strongest_evidence: Dict[str, Any] = {}
        if any(result.get("status") == "violated" for result in check_results):
            aggregate_status = "violated"
            strongest_evidence = next(
                (result for result in check_results if result.get("status") == "violated"),
                {},
            )
        elif any(result.get("status") == "supported" for result in check_results):
            aggregate_status = "supported"
            strongest_evidence = next(
                (result for result in check_results if result.get("status") == "supported"),
                {},
            )

        return {
            "aggregate_status": aggregate_status,
            "check_mode": infer_check_mode(validator_specs),
            "executors_used": sorted(set(executors_used)),
            "check_results": check_results,
            "strongest_evidence": strongest_evidence,
            "summary": (
                "At least one validation spec was grounded with concrete violation evidence."
                if aggregate_status == "violated"
                else "Validator specifications were synthesized, but lightweight checking remained partial or inconclusive."
            ),
        }

    def _finalize_report(
        self,
        code: str,
        vuln_type: str,
        predicted_line: str,
        final_bug_type: str,
        property_bundle: Dict[str, Any],
        scenario_bundle: Dict[str, Any],
        validator_spec_bundle: Dict[str, Any],
        validation_bundle: Dict[str, Any],
    ) -> str:
        aggregate_status = (validation_bundle or {}).get("aggregate_status")
        prompt = f"""
You are a rigorous Smart Contract Security Critic.
Your job is to produce the final audit verdict after a multi-stage verification process.

Inputs:
1. Detector signal
   - Predicted type: {vuln_type}
   - Predicted location: {predicted_line}
   - Validation-prioritized bug type: {final_bug_type}
2. LLM-inferred protocol properties
{json.dumps(property_bundle, ensure_ascii=False, indent=2)}
3. LLM-generated adversarial scenarios
{json.dumps(scenario_bundle, ensure_ascii=False, indent=2)}
4. Synthesized validator specifications
{json.dumps(validator_spec_bundle, ensure_ascii=False, indent=2)}
5. Lightweight validation evidence
{json.dumps(validation_bundle, ensure_ascii=False, indent=2)}

Contract:
```solidity
{code}
```

Instructions:
- Decide whether the finding is a True Positive, False Positive, or Needs More Evidence.
- Treat `aggregate_status = violated` as strong evidence of a property-level business-logic flaw.
- Treat `aggregate_status = supported` as local evidence against the suspected violation, not conclusive proof that the incident or the broader protocol is safe. A guard or modifier can protect the inspected function while the exploit exists in another function, contract implementation, proxy target, or transaction sequence.
- When `aggregate_status = supported`, return `False Positive` only if the available source code and structural evidence directly address the candidate exploit path. Otherwise return `Needs More Evidence` and identify the missing contract, path, state transition, or execution evidence.
- Treat `aggregate_status = inconclusive` as insufficient structural evidence. Do not convert an inconclusive result into a confirmed finding unless the supplied code and scenario independently establish a concrete exploit path.
- Treat `aggregate_status = not_run` as unavailable validation evidence. Do not describe the finding as evidence-validated; state that the validation stage was disabled when this is relevant.
- Explicitly state the violated property, the adversarial condition, and the exploit mechanism.
- If evidence is incomplete, explain what additional source code, execution trace, or testing would be needed.

Respond ONLY with valid JSON:
{{
  "verdict": "True Positive" | "False Positive" | "Needs More Evidence",
  "violated_property": string,
  "root_cause_analysis": string,
  "exploit_scenario": string,
  "recommendation": string,
  "confidence": float
}}
"""
        raw = self._call_llm(prompt)
        parsed = extract_json_object(raw)
        if parsed:
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        return raw

    def run_pipeline(
        self,
        code_snippet: str,
        rag_metadata: Optional[Dict[str, Any]],
        vuln_type: str,
        predicted_line: str = "Unknown",
        stage_config: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """Run the Reason-Agent stages with an explicit, serializable ablation config.

        Disabled stages receive an empty, labelled bundle rather than being silently
        replaced by an LLM call.  This makes counterfactual ablations reproducible
        and keeps downstream stages aware that an input is unavailable.
        """
        config = {
            "property_synthesis": True,
            "scenario_generation": True,
            "validator_spec_synthesis": True,
            "lightweight_validation": True,
        }
        if stage_config:
            for name, enabled in stage_config.items():
                if name not in config:
                    raise ValueError(f"Unknown pipeline stage: {name}")
                config[name] = bool(enabled)

        rag_metadata = rag_metadata or {}
        if config["property_synthesis"]:
            property_bundle = self._synthesize_properties(code_snippet, vuln_type, predicted_line, rag_metadata)
        else:
            property_bundle = {
                "protocol_family": "not_inferred",
                "critical_state": [],
                "critical_functions": [],
                "properties": [],
                "reasoning_summary": "Property Synthesis disabled by the ablation configuration.",
                "stage_status": "disabled",
            }

        if config["scenario_generation"]:
            scenario_bundle = self._generate_scenarios(code_snippet, property_bundle, rag_metadata)
        else:
            scenario_bundle = {
                "scenarios": [],
                "stage_status": "disabled",
                "reasoning_summary": "Scenario Generation disabled by the ablation configuration.",
            }

        if config["validator_spec_synthesis"]:
            validator_spec_bundle = self._generate_validator_specs(
                code_snippet, property_bundle, scenario_bundle, rag_metadata
            )
        else:
            validator_spec_bundle = {
                "validator_specs": [],
                "stage_status": "disabled",
                "selection_summary": "Validator-Spec Synthesis disabled by the ablation configuration.",
            }

        if config["lightweight_validation"]:
            validation_bundle = self._lightweight_validate(
                code_snippet,
                property_bundle,
                scenario_bundle,
                validator_spec_bundle,
                vuln_type,
            )
        else:
            validation_bundle = {
                "aggregate_status": "not_run",
                "check_mode": "not_run",
                "executors_used": [],
                "check_results": [],
                "strongest_evidence": {},
                "summary": "Lightweight Validation disabled by the ablation configuration.",
                "stage_status": "disabled",
            }

        validated_bug_type = vuln_type
        strongest_evidence = validation_bundle.get("strongest_evidence", {})
        if strongest_evidence.get("evidence_type") == "symbolic_witness":
            evidence_payload = strongest_evidence.get("evidence", {})
            validated_bug_type = evidence_payload.get("bug_type", vuln_type)
        final_report = self._finalize_report(
            code_snippet,
            vuln_type,
            predicted_line,
            validated_bug_type,
            property_bundle,
            scenario_bundle,
            validator_spec_bundle,
            validation_bundle,
        )

        output = {
            "type": "explanation_result",
            "vuln_type": vuln_type,
            "validated_bug_type": validated_bug_type,
            "property_bundle": property_bundle,
            "scenario_bundle": scenario_bundle,
            "validator_spec_bundle": validator_spec_bundle,
            "validation_bundle": validation_bundle,
            "Audit_report": final_report,
            "ablation_config": config,
        }

        return output

    def _run(self) -> Dict[str, Any]:
        fusion_data = safe_read_json(FUSION_FILE_PATH)
        rag_data = safe_read_json(RAG_FILE_PATH)

        code_snippet = rag_data.get("code", "")
        rag_metadata = rag_data.get("metadata", {})
        vuln_type = fusion_data.get("Predict", "")
        predicted_line = fusion_data.get("Predicted_Line_of_Vulnerability", "Unknown")

        print(f"Using fusion file path: {FUSION_FILE_PATH}")
        print(f"Vulnerability: {vuln_type}")
        print(f"Predicted line: {predicted_line}")

        output = self.run_pipeline(code_snippet, rag_metadata, vuln_type, predicted_line)

        with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Saved {OUTPUT_FILE_PATH}")
        return output


def build_explainer_llm() -> ChatOpenAI:
    """Build the configured OpenAI-compatible chat model without an API call."""
    return build_chat_model()


def build_explainer_tool() -> ExplainerTool:
    """Create the explainer tool for direct or graph-based execution."""
    return ExplainerTool(llm=build_explainer_llm())
