"""Download the public, revision-pinned RQ3 LoRA adapters from Hugging Face.

These repositories contain adapters, not full base-model weights. The
downloaded files are local artifacts and are excluded from Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODELS = {
    "Qwen2.5-0.5B-Solidity": "97bb627637181426fbb0f9ae496e8dbb92819c05",
    "Qwen3-0.6B-Solidity": "61695e85cc4209328381b559dff54a52b26754c6",
    "Phi-3.5-mini-instruct-Solidity": "6b882d632c14af2f5256d4852bfd6144ef3be807",
    "Qwen-Coder-2.5-3B-Solidity": "b0f073f30f1c50d7f730d7009546ea9614c3ad58",
    "DeepSeek-R1-Distill-Llama-8B-SOLIDITY-EVAL": "5d7339387d53462684e550b77e5341d0a1543a67",
    "Qwen-Coder-2.5-14B-Solidity": "428bfb5be6be6edf4a8a6b4a9fb04d4ba9d42d00",
}
WEIGHT_SHA256 = {
    "Qwen2.5-0.5B-Solidity": "acf5e482d8529502560c078022f2c49d9c83b6406e4e58f28be060b1fcf7d23a",
    "Qwen3-0.6B-Solidity": "bbe34a24ebedb8b78bdca9d1388015529173507d589d373269be85df55138228",
    "Phi-3.5-mini-instruct-Solidity": "fbf477a01e5ef2b8dce4c77acc76245877df22a1d2d22cfb8153a7c88c42c550",
    "Qwen-Coder-2.5-3B-Solidity": "f415f3228e06cfab1db7e8cb20fb819f50c2c5627e4ae963e65793d878e68ed7",
    "DeepSeek-R1-Distill-Llama-8B-SOLIDITY-EVAL": "f966bae4702ad7b4ff05f8eec591f6a014ab36e0745e26e1d4323efda24be652",
    "Qwen-Coder-2.5-14B-Solidity": "e8078c4a4ea745894bd1ceba0157a83c9ef75ff1b844fc3fc529fed7624749f9",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", choices=sorted(MODELS),
                        help="Select one or more adapters; default: all six")
    parser.add_argument("--output-dir", type=Path, default=HERE / "models")
    parser.add_argument("--dry-run", action="store_true", help="List revisions without downloading")
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify local adapter checksums without network requests")
    parser.add_argument("--retries", type=int, default=4,
                        help="Retry transient Hub/network failures this many times")
    args = parser.parse_args()

    selected = args.model or list(MODELS)
    if args.dry_run:
        for name in selected:
            print(f"Nhudang/{name} @ {MODELS[name]}")
        return

    if not args.verify_only:
        from huggingface_hub import snapshot_download

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in selected:
        repo_id = f"Nhudang/{name}"
        target = output_dir / name
        action = "Verifying" if args.verify_only else "Downloading"
        print(f"{action} {repo_id} @ {MODELS[name]} -> {target}", flush=True)
        if not args.verify_only:
            for attempt in range(1, args.retries + 1):
                try:
                    snapshot_download(
                        repo_id=repo_id,
                        revision=MODELS[name],
                        local_dir=target,
                        token=False,  # Public repositories; never use a pasted secret.
                        max_workers=1,
                    )
                    break
                except Exception as exc:
                    if attempt == args.retries:
                        raise RuntimeError(f"Could not download {repo_id} after {attempt} attempts") from exc
                    delay = min(2 ** attempt, 30)
                    print(f"  Hub request failed ({type(exc).__name__}); retrying in {delay}s", flush=True)
                    time.sleep(delay)
        adapter_config = target / "adapter_config.json"
        weights = target / "adapter_model.safetensors"
        if not adapter_config.is_file() or not weights.is_file():
            raise RuntimeError(f"Incomplete adapter snapshot: {target}")
        actual_hash = sha256_file(weights)
        if actual_hash != WEIGHT_SHA256[name]:
            raise RuntimeError(f"Checksum mismatch for {weights}; expected revision {MODELS[name]}")
        config = json.loads(adapter_config.read_text(encoding="utf-8"))
        entry = {
            "repo_id": repo_id,
            "revision": MODELS[name],
            "local_dir": str(target),
            "base_model_name_or_path": config.get("base_model_name_or_path"),
            "adapter_model_sha256": actual_hash,
        }
        manifest.append(entry)
        print(f"  adapter OK; base={entry['base_model_name_or_path']}", flush=True)
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    print(f"Downloaded {len(manifest)} adapter(s). Base weights are not included.")


if __name__ == "__main__":
    main()
