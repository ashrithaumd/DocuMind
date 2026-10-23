"""The retrieval benchmark's bookkeeping, against a stub store: warm-up
calls are discarded, every timed call is counted, scope is passed, embed
and search are timed separately, and gold-page ranks turn into MRR and
recall@k for the fused ranking and each leg."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
import bench_retrieval  # noqa: E402
from ingestion.store import RetrievedChunk  # noqa: E402


def _chunk(page, dense_rank=None, sparse_rank=None):
    return RetrievedChunk(chunk_id=f"c{page}", text="", filename="f.pdf", page_number=page,
                          score=0.0, dense_rank=dense_rank, sparse_rank=sparse_rank,
                          similarity=None)


class _StubStore:
    def __init__(self, ranking=()):
        self.embedded = []
        self.searches = []
        self.ranking = list(ranking)

    async def embed_query(self, query):
        self.embedded.append(query)
        return [0.0]

    async def search_with_vector(self, query, query_vector, doc_ids=None):
        self.searches.append((query, doc_ids))
        return []

    async def fused_ranking(self, query, query_vector, doc_ids=None):
        return self.ranking


async def test_warmup_is_discarded_and_every_timed_call_counted():
    store = _StubStore()

    timings = await bench_retrieval.bench(store, ["q1", "q2"], ["doc"], reps=3, warmup=1)

    assert len(store.embedded) == len(store.searches) == 8  # (1 warm-up + 3 timed) x 2
    assert all(doc_ids == ["doc"] for _, doc_ids in store.searches)
    for part in ("embed", "search", "total"):
        assert {q: len(runs) for q, runs in timings[part].items()} == {"q1": 3, "q2": 3}


async def test_total_is_embed_plus_search():
    timings = await bench_retrieval.bench(_StubStore(), ["q1"], ["doc"], reps=2, warmup=0)

    for embed, search, total in zip(timings["embed"]["q1"], timings["search"]["q1"],
                                    timings["total"]["q1"]):
        assert total == embed + search


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
    ranking = [_chunk(2, dense_rank=1, sparse_rank=2), _chunk(3, sparse_rank=1),
               _chunk(1, dense_rank=2)]

    assert bench_retrieval.gold_ranks(ranking, 3) == {"fused": 2, "dense": None, "sparse": 1}
    assert bench_retrieval.gold_ranks(ranking, 5) == {"fused": None, "dense": None, "sparse": None}


def test_quality_mrr_and_recall_count_misses_as_zero():
    q = bench_retrieval.quality([1, 2, 4, None])

    assert q["n"] == 4
    assert q["mrr"] == round((1 + 1 / 2 + 1 / 4) / 4, 3)
    assert q["recall@1"] == 0.25
    assert q["recall@3"] == 0.5


async def test_ranking_quality_skips_unlabelled_questions():
    store = _StubStore(ranking=[_chunk(1, dense_rank=1, sparse_rank=1)])
    pairs = [{"question": "a", "gold_page": 1}, {"question": "b"}]

    result = await bench_retrieval.ranking_quality(store, pairs, ["doc"])

    assert [q["question"] for q in result["per_question"]] == ["a"]
    assert result["fused"]["mrr"] == 1.0
    assert await bench_retrieval.ranking_quality(store, [{"question": "b"}], ["doc"]) is None
