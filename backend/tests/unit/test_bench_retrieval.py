"""The retrieval benchmark's bookkeeping, against a stub store: each
question is embedded once and every search reuses that vector, embed and
search reps are counted separately, warm-up searches are discarded, scope
is passed, and gold-page ranks turn into MRR, recall@k and tie counts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
import bench_retrieval  # noqa: E402
from ingestion.store import RetrievedChunk  # noqa: E402


def _chunk(page, dense_rank=None, sparse_rank=None, score=0.0):
    return RetrievedChunk(chunk_id=f"c{page}", text="", filename="f.pdf", page_number=page,
                          score=score, dense_rank=dense_rank, sparse_rank=sparse_rank,
                          similarity=None)


class _StubStore:
    def __init__(self, ranking=()):
        self.embedded = []
        self.searches = []
        self.ranking = list(ranking)

    async def embed_query(self, query):
        self.embedded.append(query)
        return [float(len(self.embedded))]  # a distinct vector per call

    async def search_with_vector(self, query, query_vector, doc_ids=None):
        self.searches.append((query, query_vector, doc_ids))
        return []

    async def fused_ranking(self, query, query_vector, doc_ids=None):
        return self.ranking


async def test_one_vector_per_question_reused_by_every_search():
    store = _StubStore()

    timings, vectors = await bench_retrieval.bench(store, ["q1", "q2"], ["doc"], reps=3,
                                                   warmup=1, embed_reps=2)

    # Per question: 1 untimed embed + 2 timed embeds; 1 warm-up + 3 timed searches.
    assert store.embedded == ["q1"] * 3 + ["q2"] * 3
    assert len(store.searches) == 8
    assert {q: len(runs) for q, runs in timings["embed"].items()} == {"q1": 2, "q2": 2}
    assert {q: len(runs) for q, runs in timings["search"].items()} == {"q1": 3, "q2": 3}
    assert set(timings) == {"embed", "search"}
    # Every search for a question used the first (untimed) embedding.
    assert {tuple(v) for q, v, _ in store.searches if q == "q1"} == {tuple(vectors["q1"])}
    assert all(doc_ids == ["doc"] for _, _, doc_ids in store.searches)


def test_summary_reports_n_and_percentiles():
    timings = {"q1": [10.0, 20.0, 30.0], "q2": [40.0, 50.0, 60.0]}

    summary = bench_retrieval.summarize(timings)

    assert summary["n"] == 6
    assert summary["p50"] == 35.0
    assert summary["min"] == 10.0 and summary["max"] == 60.0
    assert summary["p50"] <= summary["p95"] <= summary["p99"]
    assert [q["n"] for q in summary["per_question"]] == [3, 3]


def test_gold_ranks_per_leg():
    # Fused order: page 2, page 3, page 1. Page 3 is 1st in the sparse leg
    # but missing from the dense leg.
    ranking = [_chunk(2, dense_rank=1, sparse_rank=2, score=3.0), _chunk(3, sparse_rank=1, score=2.0),
               _chunk(1, dense_rank=2, score=1.0)]

    ranks = bench_retrieval.gold_ranks(ranking, 3)
    assert (ranks["fused"], ranks["dense"], ranks["sparse"]) == (2, None, 1)
    assert not ranks["top_tied"] and not ranks["gold_tied"]
    missing = bench_retrieval.gold_ranks(ranking, 5)
    assert (missing["fused"], missing["gold_tied"]) == (None, False)


def test_ties_are_flagged():
    # Run A's Q7 shape: gold (page 2) tied with page 4 at the top, after it.
    ranking = [_chunk(4, dense_rank=1, sparse_rank=2, score=0.5),
               _chunk(2, dense_rank=2, sparse_rank=1, score=0.5),
               _chunk(5, dense_rank=3, sparse_rank=3, score=0.1)]

    ranks = bench_retrieval.gold_ranks(ranking, 2)

    assert ranks["fused"] == 2
    assert ranks["top_tied"] and ranks["gold_tied"]
    assert not bench_retrieval.gold_ranks(ranking, 5)["gold_tied"]


def test_quality_mrr_and_recall_count_misses_as_zero():
    q = bench_retrieval.quality([1, 2, 4, None])

    assert q["n"] == 4
    assert q["mrr"] == round((1 + 1 / 2 + 1 / 4) / 4, 3)
    assert q["recall@1"] == 0.25
    assert q["recall@3"] == 0.5


async def test_ranking_quality_skips_unlabelled_questions():
    store = _StubStore(ranking=[_chunk(1, dense_rank=1, sparse_rank=1)])
    pairs = [{"question": "a", "gold_page": 1}, {"question": "b"}]

    result = await bench_retrieval.ranking_quality(store, pairs, ["doc"], {"a": [1.0]})

    assert [q["question"] for q in result["per_question"]] == ["a"]
    assert result["fused"]["mrr"] == 1.0
    assert result["ties"] == {"questions_with_top_tie": 0, "questions_gold_rank_decided_by_tie": 0}
    assert store.embedded == []  # scored on the stored vector, no new embedding call
    assert await bench_retrieval.ranking_quality(store, [{"question": "b"}], ["doc"], {}) is None
