"""The corpus loader's pacing (scripts/load_corpus.py): never more than the
limit in any rolling minute, slices that fit, every text counted. A fake
clock, no network."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import load_corpus  # noqa: E402


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def test_limiter_lets_a_full_minute_through_then_waits_for_the_window():
    clock = FakeClock()
    limiter = load_corpus.RateLimiter(90, clock=clock, sleep=clock.sleep)
    limiter.acquire(60)
    clock.now = 10
    limiter.acquire(30)  # 90 in the window: still allowed
    assert clock.slept == []
    clock.now = 20
    limiter.acquire(20)  # would be 110: waits until the first 60 leave the window (t=60)
    assert clock.slept == [40]
    assert clock.now == 60


def test_no_rolling_minute_ever_exceeds_the_limit():
    clock = FakeClock()
    limiter = load_corpus.RateLimiter(90, clock=clock, sleep=clock.sleep)
    sent: list[tuple[float, int]] = []
    for n in [29, 29, 24, 20, 29, 28, 11, 29, 9, 25, 25, 29, 1, 3, 29, 29]:  # corpus-like sizes
        limiter.acquire(n)
        sent.append((clock.now, n))
        clock.now += 0.5  # embedding time
    for t, _ in sent:
        in_window = sum(n for s, n in sent if t - 60 < s <= t)
        assert in_window <= 90, (t, in_window)


def test_a_batch_bigger_than_a_minute_is_refused_by_the_limiter():
    with pytest.raises(ValueError, match="split them"):
        load_corpus.RateLimiter(10).acquire(11)


class CountingEmbedder:
    model, dimension = "fake", 3

    def __init__(self):
        self.calls: list[int] = []

    async def embed_documents(self, texts):
        self.calls.append(len(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_paced_embedder_slices_to_the_limit_and_counts_every_text():
    clock = FakeClock()
    inner = CountingEmbedder()
    paced = load_corpus.PacedEmbedder(inner, load_corpus.RateLimiter(10, clock=clock, sleep=clock.sleep))

    vectors = paced.embed([f"t{i}" for i in range(25)])

    assert len(vectors) == 25
    assert inner.calls == [10, 10, 5]
    assert paced.requests == 25
    assert clock.slept == [60, 60]  # each new slice waited for the window
