"""Pacing for bulk embedding under the Gemini free tier (from the Phase 3
corpus loader on phase3-corpus; FinPilot's document loader uses it).

The free tier limits embedding requests (about 100 texts a minute) and,
tighter in practice, embedding tokens (30,000 a minute, measured):
  - RateLimiter keeps texts, or tokens, under a per-minute cap in any rolling
    minute, before they are sent;
  - PacedEmbedder sends texts in slices that fit both caps;
  - embed_with_retries waits out a per-minute 429 that outlasts the client's
    own retries, and turns a daily-quota error into QuotaStop, so a loader
    stops cleanly and resumes on the next run (embed before the write
    transaction opens, and write each document in one transaction).
"""
import asyncio
import time
from collections import deque

TOKENS_PER_MINUTE = 27_000  # under the measured free-tier limit of 30,000
TEXTS_PER_MINUTE = 90  # under the free tier's ~100
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_WAIT_S = 60


class QuotaStop(Exception):
    """The provider's daily quota is used up; the run should stop cleanly."""


class RateLimiter:
    """At most `per_minute` units (texts, or tokens) in any rolling 60-second window."""

    def __init__(self, per_minute: int, clock=time.monotonic, sleep=time.sleep):
        if per_minute < 1:
            raise ValueError("per_minute must be at least 1")
        self.per_minute = per_minute
        self._clock, self._sleep = clock, sleep
        self._sent: deque[tuple[float, int]] = deque()

    def acquire(self, n: int) -> None:
        if n > self.per_minute:
            raise ValueError(f"{n} can never fit in {self.per_minute} a minute; split them")
        while True:
            now = self._clock()
            while self._sent and self._sent[0][0] <= now - 60:
                self._sent.popleft()
            if sum(count for _, count in self._sent) + n <= self.per_minute:
                self._sent.append((now, n))
                return
            self._sleep(self._sent[0][0] + 60 - now)


def count_tokens(text: str) -> int:
    """The chunker's tokenizer (cl100k_base), with the prefix the Gemini
    embedder adds to every document text: an estimate of Gemini's own count,
    hence the margin in TOKENS_PER_MINUTE."""
    import tiktoken

    from embeddings import GEMINI_DOCUMENT_PREFIX

    return len(tiktoken.get_encoding("cl100k_base").encode(GEMINI_DOCUMENT_PREFIX + text))


class PacedEmbedder:
    """Sends texts in slices that fit both limits, waiting for each.
    tokens_paced counts the tokens it let through (first sends; retries
    resend them)."""

    def __init__(self, embeddings, limiter: RateLimiter | None = None,
                 token_limiter: RateLimiter | None = None, tokens=count_tokens):
        self._embeddings = embeddings
        self._limiter = limiter or RateLimiter(TEXTS_PER_MINUTE)
        self._token_limiter = token_limiter or RateLimiter(TOKENS_PER_MINUTE)
        self._tokens = tokens
        self.model, self.dimension = embeddings.model, embeddings.dimension
        self.tokens_paced = 0

    def _slices(self, texts: list[str]):
        part, part_tokens = [], 0
        for text in texts:
            n = self._tokens(text)
            if part and (len(part) == self._limiter.per_minute
                         or part_tokens + n > self._token_limiter.per_minute):
                yield part, part_tokens
                part, part_tokens = [], 0
            part.append(text)
            part_tokens += n
        if part:
            yield part, part_tokens

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for part, part_tokens in self._slices(texts):
            self._limiter.acquire(len(part))
            self._token_limiter.acquire(part_tokens)
            self.tokens_paced += part_tokens
            vectors += asyncio.run(self._embeddings.embed_documents(part))
        return vectors


def embed_with_retries(embedder: PacedEmbedder, texts: list[str],
                       sleep=time.sleep) -> list[list[float]]:
    """Raises QuotaStop on a daily quota; waits and retries a per-minute one."""
    import llm

    for attempt in range(1, RATE_LIMIT_RETRIES + 2):
        try:
            return embedder.embed(texts)
        except Exception as exc:
            kind = llm.classify_error(exc)
            if kind == "daily_quota":
                raise QuotaStop(type(exc).__name__) from exc
            if kind in ("rate_limited", "unavailable") and attempt <= RATE_LIMIT_RETRIES:
                print(f"  {kind}: waiting {RATE_LIMIT_WAIT_S} s, then retry {attempt} of "
                      f"{RATE_LIMIT_RETRIES}", flush=True)
                sleep(RATE_LIMIT_WAIT_S)
                continue
            raise


def usage_counts(usage: dict, model: str | None) -> dict:
    """Texts, requests and retries for `model`, from the embedding client's
    own tracking (llm.track_usage), so its internal retries are included."""
    if model is None:
        return {"embedding_texts_sent": 0, "embedding_requests": 0, "embedding_retries": 0}
    return {
        "embedding_texts_sent": usage.get("embedding_texts_per_model", {}).get(model, 0),
        "embedding_requests": usage["attempts_per_model"].get(model, 0),
        "embedding_retries": sum(1 for f in usage["failures"] if f["model"] == model),
    }
