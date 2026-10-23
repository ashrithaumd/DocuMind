"""Retrieval-only benchmark: latency and ranking quality, no LLM calls.

Calls the store directly for every question in the eval set: no API, no
LangGraph, no LLM calls, so it costs nothing and isn't distorted by
free-tier queueing. Each question gets 1 warm-up call (discarded) and then
N timed calls. Two parts of a search are timed separately:

  - embed_ms:  embedding the query (local MiniLM here; a network call once
               a hosted embedding provider is used)
  - search_ms: everything after that: the vector query, the sparse leg and
               RRF fusion (HybridStore.search_with_vector)

The report gives n, p50/p95/p99 for embed, search and their total, plus
per-question p50/p95. search_ms is the store's latency; embed_ms depends on
the embedding provider.

Ranking quality uses each question's hand-verified `gold_page` from the
dataset (questions without one are skipped). For the fused ranking and for
each leg (dense, sparse) separately, it reports the rank of the first chunk
on the gold page, MRR, and recall@1/@3. The fused ranking is scored before
the pipeline's top_k cut. sample.pdf is chunked one chunk per page, so a
page rank is a chunk rank.

The index is built from eval/sample.pdf with the app's own extraction and
chunking into a fresh directory (or reused with --chroma-dir). --store and
--embeddings pick the store and embedding provider being measured.

Usage (from backend/):
    python eval/bench_retrieval.py [--store chroma] [--embeddings local]
        [--reps 20] [--warmup 1] [--out FILE.json]
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

STORES = ("chroma",)
EMBEDDINGS = ("local",)
RECALL_AT = (1, 3)


def build_store(settings, store: str, embeddings: str):
    from ingestion.store import HybridStore
    from sentence_transformers import SentenceTransformer

    assert store == "chroma" and embeddings == "local", (store, embeddings)
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


# --- latency -------------------------------------------------------------------

async def bench(store, questions: list[str], doc_ids: list[str], reps: int, warmup: int) -> dict:
    """Timed calls per question (ms), split into embed, search and their
    total; warm-up calls are discarded."""
    timings: dict[str, dict[str, list[float]]] = {"embed": {}, "search": {}, "total": {}}
    for question in questions:
        for _ in range(warmup):
            vector = await store.embed_query(question)
            await store.search_with_vector(question, vector, doc_ids=doc_ids)
        embed_runs, search_runs = [], []
        for _ in range(reps):
            start = time.perf_counter()
            vector = await store.embed_query(question)
            embedded = time.perf_counter()
            await store.search_with_vector(question, vector, doc_ids=doc_ids)
            done = time.perf_counter()
            embed_runs.append((embedded - start) * 1000)
            search_runs.append((done - embedded) * 1000)
        timings["embed"][question] = embed_runs
        timings["search"][question] = search_runs
        timings["total"][question] = [e + s for e, s in zip(embed_runs, search_runs)]
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


# --- ranking quality ------------------------------------------------------------

def gold_ranks(ranking, gold_page: int) -> dict[str, int | None]:
    """Rank of the first chunk on the gold page in the fused ranking and in
    each leg; None where it didn't place."""
    on_page = [c for c in ranking if c.page_number == gold_page]
    fused = next((i for i, c in enumerate(ranking, start=1) if c.page_number == gold_page), None)
    dense = min((c.dense_rank for c in on_page if c.dense_rank is not None), default=None)
    sparse = min((c.sparse_rank for c in on_page if c.sparse_rank is not None), default=None)
    return {"fused": fused, "dense": dense, "sparse": sparse}


def quality(ranks: list[int | None]) -> dict:
    """MRR and recall@k over gold ranks (None = not retrieved, counts as 0)."""
    n = len(ranks)
    return {
        "n": n,
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
        **{f"recall@{k}": round(sum(1 for r in ranks if r and r <= k) / n, 3) for k in RECALL_AT},
    }


async def ranking_quality(store, pairs: list[dict], doc_ids: list[str]) -> dict | None:
    labelled = [p for p in pairs if p.get("gold_page") is not None]
    if not labelled:
        return None
    per_question = []
    for pair in labelled:
        vector = await store.embed_query(pair["question"])
        ranking = await store.fused_ranking(pair["question"], vector, doc_ids=doc_ids)
        per_question.append({"question": pair["question"], "gold_page": pair["gold_page"],
                             **gold_ranks(ranking, pair["gold_page"])})
    return {
        **{leg: quality([q[leg] for q in per_question]) for leg in ("fused", "dense", "sparse")},
        "per_question": per_question,
    }


# --- main ---------------------------------------------------------------------------

async def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval-only benchmark (no LLM calls).")
    parser.add_argument("--dataset", default=str(EVAL_DIR / "dataset.json"))
    parser.add_argument("--store", choices=STORES, default="chroma")
    parser.add_argument("--embeddings", choices=EMBEDDINGS, default="local")
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
    store = build_store(settings, args.store, args.embeddings)
    doc_id = await ensure_indexed(store, settings, EVAL_DIR / dataset["document"])

    questions = [pair["question"] for pair in dataset["pairs"]]
    timings = await bench(store, questions, [doc_id], args.reps, args.warmup)
    result = {
        "what": "store.embed_query() + store.search_with_vector() scoped to the sample "
                "document; no LLM calls",
        "store": args.store,
        "embeddings": args.embeddings,
        "reps_per_question": args.reps,
        "warmup_per_question": args.warmup,
        "questions": len(questions),
        "index_chunks": store.count(),
        "embedding_model": settings.embedding_model,
        "rrf_candidates": settings.rrf_candidates,
        "top_k": settings.top_k,
        "platform": f"{platform.system()} {platform.release()}, Python {platform.python_version()}, "
                    f"{platform.processor() or platform.machine()}",
        "latency_ms": {part: summarize(runs) for part, runs in timings.items()},
        "ranking_quality": await ranking_quality(store, dataset["pairs"], [doc_id]),
    }

    print(f"Retrieval-only, store={args.store} embeddings={args.embeddings}: "
          f"{result['questions']} questions x {args.reps} reps "
          f"(+{args.warmup} warm-up each), {result['index_chunks']} chunks indexed")
    for part in ("embed", "search", "total"):
        latency = result["latency_ms"][part]
        print(f"  {part + '_ms':9s} n={latency['n']}  p50={latency['p50']}  p95={latency['p95']}  "
              f"p99={latency['p99']}  (min {latency['min']}, max {latency['max']})")
    rq = result["ranking_quality"]
    if rq is None:
        print("  ranking quality: skipped (no gold_page labels in the dataset)")
    else:
        for leg in ("fused", "dense", "sparse"):
            q = rq[leg]
            print(f"  {leg:6s} n={q['n']}  MRR={q['mrr']}  "
                  + "  ".join(f"recall@{k}={q[f'recall@{k}']}" for k in RECALL_AT))
        for i, q in enumerate(rq["per_question"], start=1):
            print(f"  {i:2d}. gold p{q['gold_page']}  fused {q['fused']}  dense {q['dense']}  "
                  f"sparse {q['sparse']}  {q['question'][:50]}")
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
