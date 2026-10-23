"""Retrieval-only latency benchmark — the Phase 0 latency baseline.

Calls the store's hybrid search directly for every question in the eval
set: no API, no LangGraph, no LLM calls, so it costs nothing and isn't
distorted by free-tier queueing. Each question gets 1 warm-up call
(discarded) and then N timed calls; the report gives n, p50/p95/p99 over
all timed calls, plus per-question p50/p95.

The index is built from eval/sample.pdf with the app's own extraction,
chunking and local embeddings into a fresh directory (or reused with
--chroma-dir). The same script reruns unchanged after the pgvector
migration: only build_store() changes.

What's timed is HybridStore.search() scoped to the sample document — the
dense query embedding, the Chroma query, BM25 and RRF fusion. The pipeline's
retrieval_ms also includes the default-scope lookup and PII redaction, so
it runs slightly higher.

Usage (from backend/):
    python eval/bench_retrieval.py [--reps 20] [--warmup 1] [--out FILE.json]
"""
import argparse
import asyncio
import json
import platform
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(EVAL_DIR.parent))  # backend modules
sys.path.insert(0, str(EVAL_DIR))

from run_eval import percentile  # noqa: E402  same interpolation as the eval report


def build_store(settings):
    from ingestion.store import HybridStore
    from sentence_transformers import SentenceTransformer

    return HybridStore(settings, SentenceTransformer(settings.embedding_model))


async def ensure_indexed(store, settings, pdf_path: Path) -> str:
    """doc_id of the sample PDF, ingesting it first if the index lacks it."""
    from ingestion.chunking import chunk_pages
    from ingestion.extract import validate_and_extract

    for doc in store.list_documents():
        if doc["filename"] == pdf_path.name:
            return doc["doc_id"]
    pages, _ = validate_and_extract(pdf_path.read_bytes(), pdf_path.name, settings)
    chunks = chunk_pages(pages, pdf_path.name, settings)
    await store.add_chunks(chunks)
    return chunks[0].doc_id


async def bench(store, questions: list[str], doc_ids: list[str], reps: int, warmup: int) -> dict:
    """Timed search calls per question (ms); warm-up calls are discarded."""
    timings: dict[str, list[float]] = {}
    for question in questions:
        for _ in range(warmup):
            await store.search(question, doc_ids=doc_ids)
        runs = []
        for _ in range(reps):
            start = time.perf_counter()
            await store.search(question, doc_ids=doc_ids)
            runs.append((time.perf_counter() - start) * 1000)
        timings[question] = runs
    return timings


def summarize(timings: dict[str, list[float]]) -> dict:
    every = [ms for runs in timings.values() for ms in runs]
    return {
        "n": len(every),
        "p50": percentile(every, 50),
        "p95": percentile(every, 95),
        "p99": percentile(every, 99),
        "mean": round(statistics.mean(every), 1),
        "min": round(min(every), 1),
        "max": round(max(every), 1),
        "per_question": [
            {"question": q, "n": len(runs), "p50": percentile(runs, 50), "p95": percentile(runs, 95)}
            for q, runs in timings.items()
        ],
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval-only latency benchmark (no LLM calls).")
    parser.add_argument("--dataset", default=str(EVAL_DIR / "dataset.json"))
    parser.add_argument("--reps", type=int, default=20, help="Timed calls per question")
    parser.add_argument("--warmup", type=int, default=1, help="Discarded calls per question")
    parser.add_argument("--chroma-dir", default=None,
                        help="Index directory (default: a fresh temporary one)")
    parser.add_argument("--out", default=None, help="Write the results as JSON here")
    args = parser.parse_args()

    from config import Settings

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    chroma_dir = args.chroma_dir or tempfile.mkdtemp(prefix="documind-bench-")
    settings = Settings(chroma_host=None, chroma_dir=chroma_dir)
    store = build_store(settings)
    doc_id = await ensure_indexed(store, settings, EVAL_DIR / dataset["document"])

    questions = [pair["question"] for pair in dataset["pairs"]]
    timings = await bench(store, questions, [doc_id], args.reps, args.warmup)
    result = {
        "what": "HybridStore.search() scoped to the sample document; no LLM calls",
        "reps_per_question": args.reps,
        "warmup_per_question": args.warmup,
        "questions": len(questions),
        "index_chunks": store.count(),
        "embedding_model": settings.embedding_model,
        "rrf_candidates": settings.rrf_candidates,
        "top_k": settings.top_k,
        "platform": f"{platform.system()} {platform.release()}, Python {platform.python_version()}, "
                    f"{platform.processor() or platform.machine()}",
        "latency_ms": summarize(timings),
    }

    latency = result["latency_ms"]
    print(f"Retrieval-only, {result['questions']} questions x {args.reps} reps "
          f"(+{args.warmup} warm-up each), {result['index_chunks']} chunks indexed")
    print(f"n={latency['n']}  p50={latency['p50']} ms  p95={latency['p95']} ms  "
          f"p99={latency['p99']} ms  (min {latency['min']}, max {latency['max']})")
    for i, q in enumerate(latency["per_question"], start=1):
        print(f"  {i:2d}. p50 {q['p50']:6.1f}  p95 {q['p95']:6.1f}  {q['question'][:60]}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")
    if not args.chroma_dir:
        del store  # release Chroma's file handles before removing the index
        shutil.rmtree(chroma_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
