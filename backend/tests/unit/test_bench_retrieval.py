"""The retrieval benchmark's bookkeeping, against a stub store: warm-up
calls are discarded, every timed call is counted, and scope is passed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
import bench_retrieval  # noqa: E402


class _StubStore:
    def __init__(self):
        self.calls = []

    async def search(self, query, doc_ids=None):
        self.calls.append((query, doc_ids))
        return []


async def test_warmup_is_discarded_and_every_timed_call_counted():
    store = _StubStore()

    timings = await bench_retrieval.bench(store, ["q1", "q2"], ["doc"], reps=3, warmup=1)

    assert len(store.calls) == 8  # (1 warm-up + 3 timed) x 2
    assert all(doc_ids == ["doc"] for _, doc_ids in store.calls)
    assert {q: len(runs) for q, runs in timings.items()} == {"q1": 3, "q2": 3}


def test_summary_reports_n_and_percentiles():
    timings = {"q1": [10.0, 20.0, 30.0], "q2": [40.0, 50.0, 60.0]}

    summary = bench_retrieval.summarize(timings)

    assert summary["n"] == 6
    assert summary["p50"] == 35.0
    assert summary["min"] == 10.0 and summary["max"] == 60.0
    assert summary["p50"] <= summary["p95"] <= summary["p99"]
    assert [q["n"] for q in summary["per_question"]] == [3, 3]
