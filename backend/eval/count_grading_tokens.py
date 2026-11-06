"""Grading-call input tokens: 700-character excerpts vs full excerpts.

Rebuilds the full relevance-grading request for every question in an eval
results.jsonl, exactly as grade_node builds it (graph/nodes.py): the grader
system prompt, the redacted question, and the numbered excerpts with
filename and page number, as structured output with the grader's JSON
schema. Two variants per question, over the SAME retrieved chunks:

  - truncated_700: each excerpt cut to its first 700 characters (the grader
                   before commit 8cbc2e4);
  - full:          each excerpt whole (the grader after it).

Each request is counted with the Gemini API's countTokens endpoint
(models/{model}:countTokens with a full generateContentRequest), on the
configured grading model. The google-genai SDK refuses system_instruction
in countTokens outside Vertex, so this calls the REST endpoint directly.
Adding the JSON schema doesn't change the count (verified 2026-10-23).
No generation calls; countTokens doesn't consume generation quota.

Page numbers aren't in results.jsonl, only chunk text, so the excerpt
headers are rebuilt by chunking the dataset's document with the app's own
extraction + chunking code and matching each retrieved text exactly
(a mismatch stops the script rather than guessing).

If the results file has live usage (llm_usage.calls, recorded by the
gateway since the token-usage commit), each question's live grading
prompt_tokens is reported next to the "full" count, with the gap.

Usage (from backend/, GEMINI_API_KEY in .env):
    python eval/count_grading_tokens.py RESULTS_JSONL [--out FILE]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import httpx

EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(EVAL_DIR.parent))  # backend modules

from config import get_settings  # noqa: E402
from graph.nodes import _GRADE_SCHEMA, _GRADE_SYSTEM, _numbered_context  # noqa: E402
from graph.redaction import redact  # noqa: E402
from ingestion.chunking import chunk_pages  # noqa: E402
from ingestion.extract import validate_and_extract  # noqa: E402

OLD_MAX_CHARS = 700
COUNT_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:countTokens"


def grading_user_content(question: str, chunks: list[dict]) -> str:
    """The user content grade_node sends (same f-string, same helper)."""
    return f"Question: {question}\n\nExcerpts:\n{_numbered_context(chunks)}"


def truncated(chunks: list[dict], max_chars: int) -> list[dict]:
    """The pre-fix excerpts: the old helper did chunk["text"][:max_chars]."""
    return [{**chunk, "text": chunk["text"][:max_chars]} for chunk in chunks]


def page_index(document: str, settings) -> dict[str, dict]:
    """Retrieved (redacted) chunk text -> {filename, page_number}."""
    path = EVAL_DIR / document
    pages, _ = validate_and_extract(path.read_bytes(), path.name, settings)
    index = {}
    for chunk in chunk_pages(pages, path.name, settings):
        text, _ = redact(chunk.text)
        index[text] = {"filename": chunk.filename, "page_number": chunk.page_number}
    return index


def last_records(results_path: Path) -> list[dict]:
    """Last record per question, in first-seen order (as run_eval does)."""
    records: dict[str, dict] = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["user_input"]] = record
    return list(records.values())


def count_tokens(client: httpx.Client, model: str, api_key: str, user: str) -> int:
    request = {
        "model": f"models/{model}",
        "systemInstruction": {"parts": [{"text": _GRADE_SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"responseMimeType": "application/json",
                             "responseJsonSchema": _GRADE_SCHEMA},
    }
    response = client.post(COUNT_URL.format(model=model), headers={"x-goog-api-key": api_key},
                           json={"generateContentRequest": request}, timeout=60)
    response.raise_for_status()
    return response.json()["totalTokens"]


def live_grading_prompt_tokens(record: dict) -> int | None:
    calls = [c for c in (record.get("llm_usage") or {}).get("calls", []) if c["role"] == "grade"]
    return calls[0]["prompt_tokens"] if len(calls) == 1 else None


def stats(values: list[int]) -> dict:
    return {"n": len(values), "mean": round(statistics.mean(values), 1), "total": sum(values),
            "min": min(values), "max": max(values)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Count grading-call input tokens, 700-char vs full.")
    parser.add_argument("results", help="An eval results.jsonl (question + retrieved_contexts)")
    parser.add_argument("--dataset", default=str(EVAL_DIR / "dataset.json"))
    parser.add_argument("--out", default=None, help="Write the JSON report here")
    args = parser.parse_args()

    settings = get_settings()
    if settings.llm_provider != "gemini" or not settings.gemini_api_key:
        print("Needs LLM_PROVIDER=gemini and GEMINI_API_KEY (countTokens is a Gemini API call).")
        return 2
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    pages = page_index(dataset["document"], settings)
    model = settings.grading_model

    rows = []
    with httpx.Client() as client:
        for i, record in enumerate(last_records(Path(args.results)), start=1):
            contexts = record.get("retrieved_contexts") or []
            missing = [text[:60] for text in contexts if text not in pages]
            if missing:
                print(f"Q{i}: retrieved text not found in the re-chunked document: {missing}")
                return 1
            chunks = [{"text": text, **pages[text]} for text in contexts]
            question, _ = redact(record["user_input"].strip())
            before = count_tokens(client, model, settings.gemini_api_key,
                                  grading_user_content(question, truncated(chunks, OLD_MAX_CHARS)))
            after = count_tokens(client, model, settings.gemini_api_key,
                                 grading_user_content(question, chunks))
            row = {"q": i, "question": record["user_input"], "outcome": record["outcome"],
                   "excerpts": len(chunks), "truncated_700": before, "full": after,
                   "increase": after - before}
            live = live_grading_prompt_tokens(record)
            if live is not None:
                row |= {"live_prompt_tokens": live, "live_minus_full": live - after}
            rows.append(row)
            print(f"Q{i:>2}  700-char {before:>5}  full {after:>5}  +{after - before:>4}"
                  + (f"  live {live:>5} (gap {live - after:+d})" if live is not None else ""))

    report = {
        "what": "grading-call input tokens (system prompt + user content), counted with "
                "countTokens on reconstructed prompts",
        "model": model,
        "results": str(Path(args.results)),
        "old_max_chars": OLD_MAX_CHARS,
        "truncated_700": stats([r["truncated_700"] for r in rows]),
        "full": stats([r["full"] for r in rows]),
        "increase": stats([r["increase"] for r in rows]),
        "per_question": rows,
    }
    live_rows = [r for r in rows if "live_prompt_tokens" in r]
    if live_rows:
        report["live_minus_full"] = stats([r["live_minus_full"] for r in live_rows])
    print(json.dumps({k: report[k] for k in ("truncated_700", "full", "increase")}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
