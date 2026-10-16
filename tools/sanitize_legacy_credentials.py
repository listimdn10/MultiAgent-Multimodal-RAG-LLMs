"""Remove hard-coded credential fallbacks from legacy prototype entry points.

This one-time migration is deliberately kept in the repository so a release
review can verify that the public source contains no embedded provider keys or
Neo4j passwords.  It does not read, print, or modify `.env`.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAG_FILES = (ROOT / "rag_agent.py", ROOT / "bi-modal" / "rag_agent.py")
SINGLE_AGENT_FILE = ROOT / "single_agent_baseline.py"
GEMINI_UTILITY_FILE = ROOT / "gemini.py"
OPENAI_KEY_CHECK_FILE = ROOT / "test_key.py"


def replace_exactly_once(text: str, pattern: str, replacement: str, path: Path) -> str:
    updated, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Expected one credential fallback in {path}, found {count}.")
    return updated


def sanitize_rag_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="surrogateescape")
    text = replace_exactly_once(
        text,
        r"^\s*GEMINI_API_KEY\s*=.*$",
        "    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')",
        path,
    )
    for variable in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_VECTOR_INDEX"):
        text = re.sub(
            rf'^({variable}\s*=\s*os\.environ\.get\("{variable}"),\s*"[^"]*"\)$',
            r"\1)",
            text,
            flags=re.MULTILINE,
        )
    text = re.sub(
        r'^(NEO4J_DATABASE\s*=\s*os\.environ\.get\("NEO4J_DATABASE"),\s*"[^"]*"\)$',
        r'\1, "neo4j")',
        text,
        flags=re.MULTILINE,
    )
    path.write_text(text, encoding="utf-8", errors="surrogateescape")


def sanitize_single_agent(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="surrogateescape")
    text = replace_exactly_once(
        text,
        r"^GEMINI_API_KEY\s*=.*$",
        "GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')",
        path,
    )
    for variable in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_VECTOR_INDEX"):
        text = re.sub(
            rf'^({variable}\s*=\s*os\.environ\.get\("{variable}"),\s*"[^"]*"\)$',
            r"\1)",
            text,
            flags=re.MULTILINE,
        )
    text = re.sub(
        r'^(NEO4J_DATABASE\s*=\s*os\.environ\.get\("NEO4J_DATABASE"),\s*"[^"]*"\)$',
        r'\1, "neo4j")',
        text,
        flags=re.MULTILINE,
    )
    path.write_text(text, encoding="utf-8", errors="surrogateescape")


def sanitize_gemini_utility(path: Path) -> None:
    """Replace the old literal key in the standalone Gemini model lister."""
    text = path.read_text(encoding="utf-8", errors="surrogateescape")
    if 'api_key = os.environ.get("GEMINI_API_KEY")' in text:
        return
    text, count = re.subn(
        r'genai\.configure\(api_key\s*=\s*["\'].*?["\']\)',
        'api_key = os.environ.get("GEMINI_API_KEY")\n'
        'if not api_key:\n'
        '    raise RuntimeError("GEMINI_API_KEY is required; configure it in .env")\n\n'
        'genai.configure(api_key=api_key)',
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"Expected one literal Gemini configuration in {path}")
    path.write_text(text, encoding="utf-8", errors="surrogateescape")


def sanitize_openai_key_check(path: Path) -> None:
    """Move the old literal key in the diagnostic script into the environment."""
    text = path.read_text(encoding="utf-8", errors="surrogateescape")
    if 'API_KEY = os.environ.get("OPENAI_API_KEY")' in text:
        return
    if "from dotenv import load_dotenv" not in text:
        text = text.replace("import sys", "import os\nimport sys\nfrom dotenv import load_dotenv", 1)
    text = replace_exactly_once(
        text,
        r"^API_KEY\s*=.*$",
        "load_dotenv()\nAPI_KEY = os.environ.get(\"OPENAI_API_KEY\")\n"
        "if not API_KEY:\n"
        "    raise RuntimeError(\"Set OPENAI_API_KEY in the environment or .env before running this check.\")",
        path,
    )
    path.write_text(text, encoding="utf-8", errors="surrogateescape")


def main() -> None:
    for path in RAG_FILES:
        sanitize_rag_file(path)
    sanitize_single_agent(SINGLE_AGENT_FILE)
    sanitize_gemini_utility(GEMINI_UTILITY_FILE)
    sanitize_openai_key_check(OPENAI_KEY_CHECK_FILE)
    print("Sanitized legacy credential fallbacks. Configure credentials through .env instead.")


if __name__ == "__main__":
    main()
