"""Embedding pacing (ingestion/pacing.py): never more than the limit in any
rolling minute, slices that fit, every text counted, and a daily quota stops
cleanly. A fake clock, no network."""
import pytest

from ingestion import pacing


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
    limiter = pacing.RateLimiter(90, clock=clock, sleep=clock.sleep)
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
    limiter = pacing.RateLimiter(90, clock=clock, sleep=clock.sleep)
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
        pacing.RateLimiter(10).acquire(11)


class CountingEmbedder:
    model, dimension = "fake", 3

    def __init__(self):
        self.calls: list[int] = []

    async def embed_documents(self, texts):
        self.calls.append(len(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]


def _paced(inner, clock, texts_per_minute, tokens_per_minute, tokens=lambda t: 1):
    return pacing.PacedEmbedder(
        inner, pacing.RateLimiter(texts_per_minute, clock=clock, sleep=clock.sleep),
        pacing.RateLimiter(tokens_per_minute, clock=clock, sleep=clock.sleep), tokens=tokens)


def test_paced_embedder_slices_to_the_text_limit():
    clock = FakeClock()
    inner = CountingEmbedder()
    paced = _paced(inner, clock, texts_per_minute=10, tokens_per_minute=10_000)

    vectors = paced.embed([f"t{i}" for i in range(25)])

    assert len(vectors) == 25
    assert inner.calls == [10, 10, 5]
    assert paced.tokens_paced == 25
    assert clock.slept == [60, 60]  # each new slice waited for the window


def test_paced_embedder_slices_to_the_token_limit():
    # 40-token texts under 100 tokens a minute: two texts per slice, and a
    # wait whenever the next slice would push the rolling minute over 100.
    clock = FakeClock()
    inner = CountingEmbedder()
    paced = _paced(inner, clock, texts_per_minute=90, tokens_per_minute=100, tokens=lambda t: 40)

    paced.embed([f"t{i}" for i in range(5)])

    assert inner.calls == [2, 2, 1]
    assert paced.tokens_paced == 200
    assert clock.slept == [60, 60]


def test_token_count_uses_the_chunkers_tokenizer_with_the_gemini_prefix():
    import tiktoken

    from embeddings import GEMINI_DOCUMENT_PREFIX

    text = "Pump 3 runs at 402 RPM in module 3."
    expected = len(tiktoken.get_encoding("cl100k_base").encode(GEMINI_DOCUMENT_PREFIX + text))
    assert pacing.count_tokens(text) == expected > len(text.split())


class FailingEmbedder(CountingEmbedder):
    def __init__(self, errors):
        super().__init__()
        self.errors = list(errors)

    async def embed_documents(self, texts):
        if self.errors:
            raise self.errors.pop(0)
        return await super().embed_documents(texts)


def test_a_per_minute_limit_waits_then_retries(monkeypatch):
    monkeypatch.setattr("llm.classify_error", lambda exc: str(exc))
    clock = FakeClock()
    paced = _paced(FailingEmbedder([RuntimeError("rate_limited")]), clock, 90, 10_000)

    assert len(pacing.embed_with_retries(paced, ["a", "b"], sleep=clock.sleep)) == 2
    assert clock.slept == [60]


def test_a_daily_quota_stops_without_retrying(monkeypatch):
    monkeypatch.setattr("llm.classify_error", lambda exc: str(exc))
    clock = FakeClock()
    inner = FailingEmbedder([RuntimeError("daily_quota")])

    with pytest.raises(pacing.QuotaStop):
        pacing.embed_with_retries(_paced(inner, clock, 90, 10_000), ["a"], sleep=clock.sleep)
    assert inner.calls == [] and clock.slept == []
