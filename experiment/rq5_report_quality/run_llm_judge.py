"""Evaluate saved audit reports with the appendix RQ5 rubric.

Input is a JSONL file with one row per (case_id, method). This runner does not
generate reports or reconstruct the original PoPoC split. It never fabricates
missing reports/scores. The default requires CONVERA/GPTLens pairs; other
method sets and explicit missing-report handling are opt-in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any


METHODS = ("CONVERA", "GPTLens")
CRITERIA = ("clarity", "completeness", "readability")
OPENAI_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        key: ({"type": "integer"} if key in CRITERIA else {"type": "string"})
        for criterion in CRITERIA
        for key in (criterion, f"{criterion}_justification")
    },
    "required": [key for criterion in CRITERIA
                 for key in (criterion, f"{criterion}_justification")],
    "additionalProperties": False,
}
PROMPT_TEMPLATE = """You are an expert smart contract security reviewer evaluating the quality of an automatically generated audit report. You will be given the Solidity contract, the vulnerability that was identified, and the generated audit report. Your task is to score the report according to three criteria, using only the information provided.
### Contract
{code_snippet}
### Identified Vulnerability
{vulnerability_type}
### Generated Audit Report
{generated_report}
### Scoring Criteria
1. Clarity (1-10): Is the vulnerability explanation understandable and technically clear? Does it avoid ambiguous descriptions and clearly connect the issue to the relevant part of the code?
2. Completeness (1-10): Does the report cover the vulnerability type, the affected code location, why the issue is dangerous, its security impact, and a suitable mitigation suggestion?
3. Readability (1-10): Is the report coherent, well organized, and free of redundancy or fragmented explanations?
### Instructions
1. Base your scores only on the contract and report provided.
2. Do not assume information that is not present in the supplied context.
3. Apply the same standard consistently regardless of which system produced the report.
4. Provide a short justification for each score, referencing specific parts of the report.
### Output Format
Return only a valid JSON object:
{
"clarity": "<1-10>",
"clarity_justification": "...",
"completeness": "<1-10>",
"completeness_justification": "...",
"readability": "<1-10>",
"readability_justification": "..."
}"""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render_prompt(row: dict[str, str]) -> str:
    return (PROMPT_TEMPLATE.replace("{code_snippet}", row["contract_context"])
            .replace("{vulnerability_type}", row["identified_vulnerability"])
            .replace("{generated_report}", row["report"]))


def load_manifest(path: Path, methods: tuple[str, ...] = METHODS,
                  allow_missing: bool = False) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Input manifest not found: {path}")
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    by_case: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Line {number}: expected a JSON object")
        fields = ("case_id", "method", "contract_context", "identified_vulnerability", "report")
        for field in fields:
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Line {number}: {field} must be a nonempty string")
        row = {field: row[field].strip() for field in fields}
        if row["method"] not in methods:
            raise ValueError(f"Line {number}: method must be one of {methods}")
        key = (row["case_id"], row["method"])
        if key in seen:
            raise ValueError(f"Line {number}: duplicate case/method {key}")
        seen.add(key)
        by_case[row["case_id"]][row["method"]] = row
        rows.append(row)
    if not rows:
        raise ValueError("Input manifest has no reports")
    for case_id, pair in by_case.items():
        if not allow_missing and set(pair) != set(methods):
            raise ValueError(f"Case {case_id!r} must contain exactly {', '.join(methods)}")
        reference = next(iter(pair.values()))
        for other in pair.values():
            for field in ("contract_context", "identified_vulnerability"):
                if reference[field] != other[field]:
                    raise ValueError(f"Case {case_id!r} has different {field} between methods")
    return sorted(rows, key=lambda r: (r["case_id"], methods.index(r["method"])))


def parse_score(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge response is not JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Judge response must be a JSON object")
    result: dict[str, Any] = {}
    for criterion in CRITERIA:
        score = value.get(criterion)
        if isinstance(score, bool):
            raise ValueError(f"{criterion} cannot be boolean")
        try:
            score = float(score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{criterion} must be numeric") from exc
        if not math.isfinite(score) or not 1 <= score <= 10:
            raise ValueError(f"{criterion} must be between 1 and 10")
        justification = value.get(f"{criterion}_justification")
        if not isinstance(justification, str) or not justification.strip():
            raise ValueError(f"{criterion}_justification must be nonempty")
        result[criterion] = score
        result[f"{criterion}_justification"] = justification.strip()
    result["overall"] = statistics.mean(result[c] for c in CRITERIA)
    return result


class ProviderConfigurationError(RuntimeError):
    """Non-retryable API configuration/authentication failure."""


def openai_generate(prompt: str, model: str, api_key: str, reasoning_effort: str,
                    max_output_tokens: int, timeout: int,
                    max_retries: int) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": model,
        "input": [{"role": "user", "content": prompt}],
        "text": {"format": {"type": "json_schema", "name": "report_quality_score",
                            "strict": True, "schema": OPENAI_SCORE_SCHEMA}},
        "reasoning": {"effort": reasoning_effort},
        "max_output_tokens": max_output_tokens,
        "store": False,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.load(response)
            if body.get("status") not in (None, "completed"):
                raise ValueError(f"OpenAI response status: {body.get('status')}")
            parts = [part.get("text", "") for item in body.get("output", [])
                     if item.get("type") == "message"
                     for part in item.get("content", []) if part.get("type") == "output_text"]
            answer = "".join(parts)
            if not answer:
                raise ValueError("OpenAI returned no output_text (possibly a refusal)")
            return answer, {"response_id": body.get("id"), "usage": body.get("usage", {})}
        except urllib.error.HTTPError as exc:
            detail = exc.read(1500).decode("utf-8", errors="replace")
            if exc.code in (400, 401, 403, 404):
                raise ProviderConfigurationError(f"OpenAI HTTP {exc.code}: {detail}") from exc
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == max_retries:
                raise RuntimeError(f"OpenAI HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == max_retries:
                raise RuntimeError(f"OpenAI connection failure: {exc}") from exc
        time.sleep(min(2 ** attempt, 30))
    raise RuntimeError("Unreachable retry state")


def gemini_generate(prompt: str, model: str, api_key: str, temperature: float,
                    timeout: int, max_retries: int) -> tuple[str, dict[str, Any]]:
    endpoint = ("https://generativelanguage.googleapis.com/v1beta/models/"
                + urllib.parse.quote(model, safe="") + ":generateContent")
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"},
    }
    request = urllib.request.Request(
        endpoint, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.load(response)
            parts = body["candidates"][0]["content"]["parts"]
            answer = "".join(part.get("text", "") for part in parts)
            if not answer:
                raise ValueError("Gemini returned no text")
            return answer, body.get("usageMetadata", {})
        except urllib.error.HTTPError as exc:
            detail = exc.read(1500).decode("utf-8", errors="replace")
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == max_retries:
                raise RuntimeError(f"Gemini HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == max_retries:
                raise RuntimeError(f"Gemini connection failure: {exc}") from exc
        time.sleep(min(2 ** attempt, 30))
    raise RuntimeError("Unreachable retry state")


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in ("GEMINI_API_KEY", "OPENAI_API_KEY") or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"\'')


def summarize(records: list[dict[str, Any]], rows: list[dict[str, str]],
              repetitions: int, methods: tuple[str, ...] = METHODS) -> dict[str, Any]:
    successful = {(r["case_id"], r["method"], r["repeat"]): r for r in records
                  if r.get("status") == "ok"}
    case_ids = sorted({r["case_id"] for r in rows})
    complete = [case_id for case_id in case_ids if all(
        (case_id, method, repeat) in successful
        for method in methods for repeat in range(1, repetitions + 1))]
    metrics: dict[str, Any] = {}
    for method in methods:
        by_repeat: list[dict[str, float]] = []
        for repeat in range(1, repetitions + 1):
            subset = [successful[(case_id, method, repeat)]["scores"] for case_id in complete]
            by_repeat.append({criterion: statistics.mean(s[criterion] for s in subset)
                              for criterion in (*CRITERIA, "overall")} if subset else {})
        metrics[method] = {
            "per_repeat_means": by_repeat,
            "mean": {criterion: statistics.mean(r[criterion] for r in by_repeat)
                     for criterion in (*CRITERIA, "overall")} if complete else {},
            "std_population_across_repeats": {
                criterion: statistics.pstdev(r[criterion] for r in by_repeat)
                for criterion in (*CRITERIA, "overall")
            } if complete else {},
        }
    available: dict[str, Any] = {}
    for method in methods:
        method_cases = [case_id for case_id in case_ids if all(
            (case_id, method, repeat) in successful for repeat in range(1, repetitions + 1))]
        available[method] = {
            "complete_case_count": len(method_cases),
            "complete_case_ids": method_cases,
            "mean_not_paired_across_methods": {
                criterion: statistics.mean(
                    successful[(case_id, method, repeat)]["scores"][criterion]
                    for case_id in method_cases for repeat in range(1, repetitions + 1))
                for criterion in (*CRITERIA, "overall")
            } if method_cases else {},
        }
    pairwise: dict[str, Any] = {}
    for first, second in combinations(methods, 2):
        pair_cases = [case_id for case_id in case_ids if all(
            (case_id, method, repeat) in successful
            for method in (first, second) for repeat in range(1, repetitions + 1))]
        means = {
            method: {criterion: statistics.mean(
                successful[(case_id, method, repeat)]["scores"][criterion]
                for case_id in pair_cases for repeat in range(1, repetitions + 1))
                for criterion in (*CRITERIA, "overall")}
            for method in (first, second)
        } if pair_cases else {}
        repeat_means = {
            method: [
                {criterion: statistics.mean(
                    successful[(case_id, method, repeat)]["scores"][criterion]
                    for case_id in pair_cases)
                    for criterion in (*CRITERIA, "overall")}
                for repeat in range(1, repetitions + 1)
            ] for method in (first, second)
        } if pair_cases else {}
        pairwise[f"{first}__vs__{second}"] = {
            "paired_case_count": len(pair_cases), "paired_case_ids": pair_cases,
            "mean_scores": means,
            "std_population_across_repeats": {
                method: {criterion: statistics.pstdev(
                    scores[criterion] for scores in repeat_means[method])
                    for criterion in (*CRITERIA, "overall")}
                for method in (first, second)
            } if pair_cases else {},
            "difference_first_minus_second": {
                criterion: means[first][criterion] - means[second][criterion]
                for criterion in (*CRITERIA, "overall")
            } if pair_cases else {},
        }
    return {
        "total_input_cases": len(case_ids),
        "complete_requested_method_cases": len(complete),
        "complete_case_ids": complete,
        "expected_judgments": len(rows) * repetitions,
        "successful_judgments": len(successful),
        "scores_on_complete_method_intersection": metrics,
        "per_method_available_reports_descriptive_only": available,
        "pairwise_comparisons_on_matched_cases": pairwise,
        "note": "Use pairwise_comparisons_on_matched_cases for each two-method comparison. The complete-method intersection uses only cases with every requested method; per-method available-report means can have different denominators and are descriptive only. These are not the original PoPoC results.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Paired report JSONL manifest")
    parser.add_argument("--output-dir", type=Path, default=Path("experiment/rq5_report_quality/results/run"))
    parser.add_argument("--provider", choices=("gemini", "openai"), default="gemini")
    parser.add_argument("--model", default=None,
                        help="Judge model (default: gemini-2.5-flash-lite or gpt-6-luna)")
    parser.add_argument("--methods", nargs="+", default=list(METHODS),
                        help="Method labels expected in the manifest (default: CONVERA GPTLens)")
    parser.add_argument("--allow-missing", action="store_true",
                        help="Allow absent reports, but compare only the complete method intersection")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=None,
                        help="Gemini only; default 0.2. OpenAI uses reasoning effort instead")
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high"),
                        default="low", help="OpenAI only")
    parser.add_argument("--max-output-tokens", type=int, default=1200,
                        help="OpenAI only")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-judgments", type=int, default=0,
                        help="Stop after this many new API attempts; 0 means all (resume with the same output-dir)")
    parser.add_argument("--dry-run", action="store_true", help="Validate pairs without API calls or writes")
    args = parser.parse_args(argv)
    if (args.repetitions < 1 or args.timeout < 1 or args.max_retries < 0
            or args.max_output_tokens < 1 or args.max_judgments < 0):
        parser.error("Invalid repetitions, timeout, retries, output tokens, or judgment limit")
    if args.temperature is not None and not 0 <= args.temperature <= 2:
        parser.error("--temperature must be between 0 and 2")
    if args.provider == "openai" and args.temperature is not None:
        parser.error("--temperature is not used with OpenAI; use --reasoning-effort")
    model = args.model or ("gpt-6-luna" if args.provider == "openai" else "gemini-2.5-flash-lite")
    temperature = 0.2 if args.temperature is None else args.temperature
    methods = tuple(args.methods)
    if not methods or len(set(methods)) != len(methods):
        parser.error("--methods must contain unique method names")
    try:
        rows = load_manifest(args.input, methods, args.allow_missing)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Validated {len({r['case_id'] for r in rows})} cases and {len(rows)} reports; "
          f"{len(rows) * args.repetitions} judgments expected.")
    if args.dry_run:
        print("Dry run: no API call and no files written.")
        return 0

    load_env_file(Path(__file__).resolve().parents[2] / ".env")
    key_name = "OPENAI_API_KEY" if args.provider == "openai" else "GEMINI_API_KEY"
    api_key = os.environ.get(key_name, "")
    if not api_key:
        parser.error(f"{key_name} is missing (environment or root .env)")
    config = {
        "input_sha256": sha256(args.input.read_bytes()),
        "prompt_sha256": sha256(PROMPT_TEMPLATE.encode("utf-8")),
        "provider": args.provider, "model": model, "repetitions": args.repetitions,
        "temperature": temperature if args.provider == "gemini" else None,
        "reasoning_effort": args.reasoning_effort if args.provider == "openai" else None,
        "max_output_tokens": args.max_output_tokens if args.provider == "openai" else None,
        "methods": list(methods),
        "allow_missing": args.allow_missing,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        parser.error("Output directory has different config/input; choose a new --output-dir")
    if not config_path.exists():
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    judgments_path = args.output_dir / "judgments.jsonl"
    records = [json.loads(line) for line in judgments_path.read_text(encoding="utf-8").splitlines()
               if line.strip()] if judgments_path.exists() else []
    completed = {(r["case_id"], r["method"], r["repeat"]) for r in records
                 if r.get("status") == "ok"}
    fatal_error = False
    attempts = 0
    with judgments_path.open("a", encoding="utf-8") as stream:
        for repeat in range(1, args.repetitions + 1):
            for row in rows:
                key = (row["case_id"], row["method"], repeat)
                if key in completed:
                    continue
                print(f"Judging case={row['case_id']} method={row['method']} repeat={repeat}", flush=True)
                try:
                    prompt = render_prompt(row)
                    if args.provider == "openai":
                        raw, usage = openai_generate(prompt, model, api_key,
                                                     args.reasoning_effort, args.max_output_tokens,
                                                     args.timeout, args.max_retries)
                    else:
                        raw, usage = gemini_generate(prompt, model, api_key,
                                                     temperature, args.timeout, args.max_retries)
                    scores = parse_score(raw)
                    record = {"case_id": key[0], "method": key[1], "repeat": repeat,
                              "status": "ok", "scores": scores, "raw_response": raw,
                              "usage": usage, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
                    completed.add(key)
                except (RuntimeError, ValueError, KeyError, IndexError) as exc:
                    fatal_error = isinstance(exc, ProviderConfigurationError)
                    record = {"case_id": key[0], "method": key[1], "repeat": repeat,
                              "status": "error", "error": str(exc),
                              "timestamp_utc": datetime.now(timezone.utc).isoformat()}
                    print(f"  ERROR: {exc}", file=sys.stderr, flush=True)
                records.append(record)
                attempts += 1
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                if fatal_error or (args.max_judgments and attempts >= args.max_judgments):
                    if fatal_error:
                        print("Stopping after non-retryable provider error.", file=sys.stderr)
                    break
            if fatal_error or (args.max_judgments and attempts >= args.max_judgments):
                break
    summary = summarize(records, rows, args.repetitions, methods)
    summary["config"] = config
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Complete requested-method cases: {summary['complete_requested_method_cases']}/{summary['total_input_cases']}")
    print(f"Results: {args.output_dir.resolve()}")
    return 0 if summary["successful_judgments"] == summary["expected_judgments"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
