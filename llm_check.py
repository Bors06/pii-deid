import json
import re

import requests

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2:3b"        # swap for a bigger model (e.g. qwen2.5:7b) on a GPU machine
MAX_CHUNK_CHARS = 4000       # long code files are split into pieces this size

SYSTEM_PROMPT = """You find personal information that is STILL visible in student messages and code.

Flag ONLY text that could help identify a specific real person:
- names or nicknames of people, also inside code identifiers (e.g. chenSolver, maria_hw2)
- usernames, email addresses, phone numbers, ID numbers
- personal details that single someone out (hometown, being "the only" X in a class, specific job or family details)

Never flag:
- placeholders in angle brackets like <STUDENT_1> or <PERSON_2>
- normal code: keywords, variables, attributes, functions (user.name, user.email, average, graph)
- famous names used as technical terms (dijkstra, Bayes, Python)

Examples:
Input: "for row in rows: print(row.first_name, row.phone)"
Output: {"findings": []}
Input: "chenSolver() passes all tests"
Output: {"findings": [{"text": "chenSolver", "type": "name in code", "is_personal_info": true, "reason": "contains the surname Chen"}]}
Input: "I'm the only exchange student from Lyon in lab 3"
Output: {"findings": [{"text": "the only exchange student from Lyon in lab 3", "type": "identifying detail", "is_personal_info": true, "reason": "singles out one student"}]}

Copy "text" EXACTLY from the input. If nothing is personal, return {"findings": []}."""

SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "type": {"type": "string"},
                    "is_personal_info": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["text", "type", "is_personal_info", "reason"],
            },
        }
    },
    "required": ["findings"],
}

PLACEHOLDER = re.compile(r"<[A-Z0-9_]+>")


def check_ollama():
    """Fail early with a clear message if Ollama isn't running or the model isn't pulled."""
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
    except requests.exceptions.ConnectionError:
        raise SystemExit("Ollama is not running. Start the Ollama app and try again.")
    names = [m["name"] for m in tags.get("models", [])]
    if not any(n == MODEL or n.startswith(MODEL + ":") for n in names):
        raise SystemExit(f"Model {MODEL} not found. Run: ollama pull {MODEL}")


def _chunks(text):
    chunk, size = [], 0
    for line in text.splitlines(keepends=True):
        if chunk and size + len(line) > MAX_CHUNK_CHARS:
            yield "".join(chunk)
            chunk, size = [], 0
        chunk.append(line)
        size += len(line)
    if chunk:
        yield "".join(chunk)


def _ask(text):
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "format": SCHEMA,          # forces valid JSON in our shape
            "stream": False,
            "options": {"temperature": 0, "seed": 42, "num_ctx": 8192},
        },
        timeout=300,
    )
    resp.raise_for_status()
    try:
        return json.loads(resp.json()["message"]["content"]).get("findings", [])
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


def llm_find_pii(text):
    """Return validated findings: exact substrings of text, not placeholders, de-duplicated."""
    visible = PLACEHOLDER.sub(" ", text)   # text with placeholders blanked out
    findings, seen = [], set()
    for chunk in _chunks(text):
        for f in _ask(chunk):
            if f.get("is_personal_info") is not True:   # model said it's not PII
                continue
            span = str(f.get("text", "")).strip()
            if (
                len(span) < 2
                or span not in visible         # invented, altered, or only inside a placeholder
                or PLACEHOLDER.search(span)    # already masked
                or span.lower() in seen
            ):
                continue
            seen.add(span.lower())
            findings.append({"text": span, "type": str(f.get("type", "")), "reason": str(f.get("reason", ""))})
    return findings


def mask_findings(text, findings, label_fn):
    """Replace each finding (whole-word, never inside an existing <PLACEHOLDER>).
    Returns (new_text, findings actually applied)."""
    applied = []
    for f in sorted(findings, key=lambda f: len(f["text"]), reverse=True):  # longest first
        pattern = re.compile(r"(?<!\w)" + re.escape(f["text"]) + r"(?!\w)")
        parts = re.split(r"(<[A-Z0-9_]+>)", text)   # keeps placeholders as separate pieces
        if not any(pattern.search(p) for p in parts if not PLACEHOLDER.fullmatch(p)):
            continue
        label = label_fn(f["text"])
        text = "".join(p if PLACEHOLDER.fullmatch(p) else pattern.sub(lambda m: label, p)
                       for p in parts)
        applied.append(f)
    return text, applied