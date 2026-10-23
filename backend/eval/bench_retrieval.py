"""Retrieval-only benchmark: latency and ranking quality, no LLM calls.

Calls the store directly for every question in the eval set: no API, no
LangGraph, no LLM calls. Two parts of a search are timed separately:

  - search_ms: the store's work for an already-embedded query: the vector
               query, the full-text leg and RRF fusion
               (store.search_with_vector). This is the latency claim.
  - embed_ms:  embedding the question with the configured provider
               (store.embed_query, the same call and task prefix the app
               uses). Local MiniLM runs on the CPU; Gemini is a network call
               on the free tier, so it reflects that service, not the store.

Method, per question:
  1. embed the question once (untimed; also the embed warm-up); this
     vector is used for every search and for the ranking-quality scores;
  2. time --embed-reps further embed_query calls (default 3);
  3. run --warmup untimed searches, then time --reps searches (default 20)
     on the stored vector.
Searching a stored vector is the same work as searching a fresh one, and
it keeps a hosted provider's calls low: Gemini's free tier allows 100
embedded texts per minute, so this run makes 14 x (1 + 3) = 56 embedding
calls instead of 14 x 21. There is no per-rep "total": embed and search
are no longer paired, and their sum would mix a network service into the
store's number.

The report gives n, p50/p95/p99 and per-question p50/p95 for each part.

Ranking quality uses each question's hand-verified `gold_page` from the
dataset (questions without one are skipped). For the fused ranking and for
each leg (dense, sparse) separately, it reports the rank of the first chunk
on the gold page, MRR, and recall@1/@3. The fused ranking is scored before
the pipeline's top_k cut. sample.pdf is chunked one chunk per page, so a
page rank is a chunk rank.

Ties: RRF gives two chunks exactly equal scores whenever their leg ranks
mirror each other (dense 1 + sparse 2 vs dense 2 + sparse 1). The store
breaks ties by a fixed rule (ingestion/store.py, rrf_fuse), and the report
counts, per run, the questions whose fused rank 1 is tied and the
questions whose gold page's fused rank was decided by a tie. On a 5-chunk
corpus these ties drive differences in fused MRR, so MRR and recall here
are not evidence about retrieval quality.

The index is built from eval/sample.pdf with the app's own extraction and
chunking, in Postgres (pgvector + full-text search) at --database-url
(default DATABASE_URL), migrated first. Use an empty database per
embedding provider: the vector column's size follows the provider.
--embeddings picks the provider being measured. (Phase 0's Chroma + BM25
run is kept in docs/baseline/phase0/retrieval_bench_split.json.)

Usage (from backend/):
    python eval/bench_retrieval.py [--embeddings local|gemini]
        [--database-url URL] [--reps 20] [--warmup 1] [--embed-reps 3]
        [--out FILE.json]
"""
import argparse
import asyncio
import json
import platform
import statistics
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(EVAL_DIR.parent))  # backend modules
sys.path.insert(0, str(EVAL_DIR))

from run_eval import percentile  # noqa: E402  same interpolation as the eval report

EMBEDDINGS = ("local", "gemini")
RECALL_AT = (1, 3)


def build_store(settings):
    from db.session import init_db
    from embeddings import build_embeddings
    from ingestion.store import HybridStore

    if not init_db(settings):  # runs the migrations
        raise SystemExit("Could not connect to or migrate the database (--database-url)")
    return HybridStore(settings, build_embeddings(settings))


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

async def bench(store, questions: list[str], doc_ids: list[str], reps: int, warmup: int,
                embed_reps: int) -> tuple[dict, dict[str, list[float]]]:
    """Per question: one untimed embed (the vector every search uses),
    `embed_reps` timed embeds, `warmup` untimed searches, then `reps` timed
    searches on the stored vector. Returns ({"embed": {q: ms}, "search":
    {q: ms}}, {q: vector})."""
    timings: dict[str, dict[str, list[float]]] = {"embed": {}, "search": {}}
    vectors: dict[str, list[float]] = {}
    for question in questions:
        vectors[question] = vector = await store.embed_query(question)
        embed_runs = []
        for _ in range(embed_reps):
            start = time.perf_counter()
            await store.embed_query(question)
            embed_runs.append((time.perf_counter() - start) * 1000)
        for _ in range(warmup):
            await store.search_with_vector(question, vector, doc_ids=doc_ids)
        search_runs = []
        for _ in range(reps):
            start = time.perf_counter()
            await store.search_with_vector(question, vector, doc_ids=doc_ids)
            search_runs.append((time.perf_counter() - start) * 1000)
        timings["embed"][question] = embed_runs
        timings["search"][question] = search_runs
    return timings, vectors


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

def gold_ranks(ranking, gold_page: int) -> dict:
    """Rank of the first chunk on the gold page in the fused ranking and in
    each leg (None where it didn't place), plus whether the fused rank 1 is
    tied and whether the gold chunk's fused rank was decided by a tie."""
    on_page = [c for c in ranking if c.page_number == gold_page]
    fused = next((i for i, c in enumerate(ranking, start=1) if c.page_number == gold_page), None)
    dense = min((c.dense_rank for c in on_page if c.dense_rank is not None), default=None)
    sparse = min((c.sparse_rank for c in on_page if c.sparse_rank is not None), default=None)
    gold = ranking[fused - 1] if fused else None
    return {
        "fused": fused, "dense": dense, "sparse": sparse,
        "top_tied": len(ranking) > 1 and ranking[0].score == ranking[1].score,
        "gold_tied": gold is not None and any(
            c.score == gold.score for c in ranking if c.chunk_id != gold.chunk_id),
    }


def quality(ranks: list[int | None]) -> dict:
    """MRR and recall@k over gold ranks (None = not retrieved, counts as 0)."""
    n = len(ranks)
    return {
        "n": n,
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
        **{f"recall@{k}": round(sum(1 for r in ranks if r and r <= k) / n, 3) for k in RECALL_AT},
    }


async def ranking_quality(store, pairs: list[dict], doc_ids: list[str],
                          vectors: dict[str, list[float]]) -> dict | None:
    """Gold-page ranks for every labelled question, scored on the vectors the
    latency run embedded (no extra embedding calls)."""
    labelled = [p for p in pairs if p.get("gold_page") is not None]
    if not labelled:
        return None
    per_question = []
    for pair in labelled:
        vector = vectors.get(pair["question"]) or await store.embed_query(pair["question"])
        ranking = await store.fused_ranking(pair["question"], vector, doc_ids=doc_ids)
        per_question.append({"question": pair["question"], "gold_page": pair["gold_page"],
                             **gold_ranks(ranking, pair["gold_page"])})
    return {
        **{leg: quality([q[leg] for q in per_question]) for leg in ("fused", "dense", "sparse")},
        "ties": {
            "questions_with_top_tie": sum(q["top_tied"] for q in per_question),
            "questions_gold_rank_decided_by_tie": sum(q["gold_tied"] for q in per_question),
        },
        "per_question": per_question,
    }


# --- main ---------------------------------------------------------------------------

async def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval-only benchmark (no LLM calls).")
    parser.add_argument("--dataset", default=str(EVAL_DIR / "dataset.json"))
    parser.add_argument("--embeddings", choices=EMBEDDINGS, default="local")
    parser.add_argument("--reps", type=int, default=20, help="Timed calls per question")
    parser.add_argument("--warmup", type=int, default=1, help="Discarded searches per question")
    parser.add_argument("--embed-reps", type=int, default=3, help="Timed embed calls per question")
    parser.add_argument("--database-url", default=None,
                        help="Postgres URL (default: DATABASE_URL)")
    parser.add_argument("--out", default=None, help="Write the results as JSON here")
    args = parser.parse_args()

    from config import Settings

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    overrides = {"database_url": args.database_url} if args.database_url else {}
    settings = Settings(embedding_provider=args.embeddings, embedding_model="",
                        embedding_dimensions=0, **overrides)
    store = build_store(settings)
    doc_id = await ensure_indexed(store, settings, EVAL_DIR / dataset["document"])

    questions = [pair["question"] for pair in dataset["pairs"]]
    timings, vectors = await bench(store, questions, [doc_id], args.reps, args.warmup,
                                   args.embed_reps)
    result = {
        "what": "store.search_with_vector() on a stored query vector (search_ms) and "
                "store.embed_query() (embed_ms), scoped to the sample document; no LLM calls",
        "method": "per question: 1 untimed embed (the vector used for every search and for "
                  "ranking quality), embed_reps timed embeds, warmup untimed searches, reps "
                  "timed searches on the stored vector",
        "store": "pgvector",
        "embeddings": args.embeddings,
        "reps_per_question": args.reps,
        "warmup_per_question": args.warmup,
        "embed_reps_per_question": args.embed_reps,
        "questions": len(questions),
        "index_chunks": store.count(),
        "embedding_model": settings.embedding_model,
        "rrf_candidates": settings.rrf_candidates,
        "top_k": settings.top_k,
        "platform": f"{platform.system()} {platform.release()}, Python {platform.python_version()}, "
                    f"{platform.processor() or platform.machine()}",
        "latency_ms": {part: summarize(runs) for part, runs in timings.items()},
        "ranking_quality": await ranking_quality(store, dataset["pairs"], [doc_id], vectors),
    }

    print(f"Retrieval-only, store=pgvector embeddings={args.embeddings}: "
          f"{result['questions']} questions, {args.reps} search reps (+{args.warmup} warm-up) "
          f"and {args.embed_reps} embed reps each, {result['index_chunks']} chunks indexed")
    for part in ("search", "embed"):
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
        ties = rq["ties"]
        print(f"  ties: rank 1 tied in {ties['questions_with_top_tie']} questions; gold rank "
              f"decided by a tie in {ties['questions_gold_rank_decided_by_tie']}")
        for i, q in enumerate(rq["per_question"], start=1):
            print(f"  {i:2d}. gold p{q['gold_page']}  fused {q['fused']}  dense {q['dense']}  "
                  f"sparse {q['sparse']}{'  (tie)' if q['gold_tied'] else ''}  {q['question'][:50]}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
