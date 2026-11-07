"""scripts/load_corpus.py on real Postgres, with the real corpus (kb/) and
fake embeddings (no network): a full load, an idempotent re-run, a
daily-quota stop part-way followed by a clean resume, a per-minute 429
retried, and an access change applied in place."""
import sys
from pathlib import Path

import pytest
import sqlalchemy as sa

from config import Settings
from db import session as db_session
from tests.helpers import FakeEmbeddings

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "kb" / "tools"))
import load_corpus  # noqa: E402
from manifest import load_manifest  # noqa: E402

pytestmark = pytest.mark.skipif(not (REPO / "kb" / "manifest.yaml").exists(),
                                reason="needs the Phase 3 corpus (kb/)")


def _quota_error(daily: bool):
    from google.genai import errors as genai_errors

    quota_id = "EmbedRequestsPerDayPerProject" if daily else "EmbedRequestsPerMinutePerProject"
    return genai_errors.ClientError(429, {"error": {
        "code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED",
        "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                     "violations": [{"quotaId": quota_id}]}]}})


class GatewayEmbeddings(FakeEmbeddings):
    """FakeEmbeddings that records into llm.track_usage the way the real
    Gemini embedder's gateway does: an attempt and its texts on every call,
    and a failure when it raises. Raises `error` once `fail_after` texts
    have gone through (only `times` times, if set)."""

    def __init__(self, fail_after: int | None = None, error=None, times: int | None = None):
        super().__init__()
        self.sent, self.fail_after, self.error, self.times = 0, fail_after, error, times

    async def embed_documents(self, texts):
        from llm import _usage

        usage = _usage.get()
        usage["attempts_per_model"][self.model] = usage["attempts_per_model"].get(self.model, 0) + 1
        sent = usage.setdefault("embedding_texts_per_model", {})
        sent[self.model] = sent.get(self.model, 0) + len(texts)
        failing = self.fail_after is not None and self.sent + len(texts) > self.fail_after
        if failing and (self.times is None or self.times > 0):
            if self.times is not None:
                self.times -= 1
            usage["failures"].append({"model": self.model, "attempt": 1, "error": repr(self.error)})
            raise self.error
        self.sent += len(texts)
        return await super().embed_documents(texts)


@pytest.fixture
def owner(pg_url):
    # FakeEmbeddings produces 32-dim vectors.
    db_session.migrate_database(Settings(_env_file=None, database_url=pg_url, embedding_dimensions=32))
    engine = sa.create_engine(pg_url)
    yield engine
    engine.dispose()


def _fast():
    return load_corpus.RateLimiter(10**12)  # never waits: the corpus is ~120k tokens


def _db_state(engine):
    with engine.connect() as conn:
        docs = conn.execute(sa.text("SELECT count(*) FROM documents WHERE doc_id LIKE 'kb-%'")).scalar()
        chunks = conn.execute(sa.text("SELECT count(*) FROM chunks WHERE doc_id LIKE 'kb-%'")).scalar()
        # Every loaded document is complete: no document without chunks.
        empty = conn.execute(sa.text(
            "SELECT count(*) FROM documents d WHERE NOT EXISTS "
            "(SELECT 1 FROM chunks c WHERE c.doc_id = d.doc_id)")).scalar()
    return docs, chunks, empty


def test_full_load_then_an_idempotent_rerun(owner, pg_url):
    manifest = load_manifest()
    first = load_corpus.load(pg_url, manifest, embeddings=GatewayEmbeddings(), limiter=_fast(), token_limiter=_fast())

    docs, chunks, empty = _db_state(owner)
    assert (first["insert"], docs, empty) == (50, 50, 0)
    assert first["chunks_by_org"] == {"aurora": 151, "bluefin": 133}  # as in the Phase 3 load
    assert first["embedding_texts_sent"] == chunks == 284
    # Every document fits one slice (under 90 texts and 27,000 tokens).
    assert (first["embedding_requests"], first["embedding_retries"]) == (50, 0)
    assert first["embedding_tokens_paced"] > chunks * 100

    again = load_corpus.load(pg_url, manifest, embeddings=GatewayEmbeddings(), limiter=_fast(), token_limiter=_fast())
    assert (again["skip"], again["insert"], again["embedding_texts_sent"]) == (50, 0, 0)
    assert _db_state(owner) == (docs, chunks, 0)


def test_a_daily_quota_stop_leaves_whole_documents_and_a_rerun_resumes(owner, pg_url, capsys):
    manifest = load_manifest()
    with pytest.raises(load_corpus.QuotaStop) as stop:
        load_corpus.load(pg_url, manifest, embeddings=GatewayEmbeddings(100, _quota_error(daily=True)),
                         limiter=_fast(), token_limiter=_fast())
    assert stop.value.counts["embedding_retries"] == 1  # the failed attempt is counted
    docs_partial, chunks_partial, empty = _db_state(owner)
    assert 0 < docs_partial < 50 and empty == 0  # whole documents only
    assert "Re-run the same command to resume" in capsys.readouterr().out

    resumed = load_corpus.load(pg_url, manifest, embeddings=GatewayEmbeddings(), limiter=_fast(), token_limiter=_fast())

    assert resumed["skip"] == docs_partial and resumed["insert"] == 50 - docs_partial
    docs, chunks, empty = _db_state(owner)
    assert (docs, empty) == (50, 0)
    full = load_corpus.load(pg_url, manifest, embeddings=GatewayEmbeddings(), limiter=_fast(), token_limiter=_fast())
    assert sum(full["chunks_by_org"].values()) == chunks  # same totals as a clean load would give
    with owner.connect() as conn:  # one documents row per file, no leftovers
        assert conn.execute(sa.text(
            "SELECT count(*) FROM (SELECT org_id, filename FROM documents GROUP BY 1, 2 "
            "HAVING count(*) > 1) d")).scalar() == 0


def test_a_per_minute_429_waits_and_retries(owner, pg_url):
    waits: list[float] = []
    embeddings = GatewayEmbeddings(0, _quota_error(daily=False), times=1)  # fails once, first call
    counts = load_corpus.load(pg_url, load_manifest(), embeddings=embeddings, limiter=_fast(), token_limiter=_fast(),
                              sleep=waits.append)
    assert waits == [load_corpus.RATE_LIMIT_WAIT_S]
    assert counts["insert"] == 50
    # The failed attempt and its texts were sent too, so they count.
    assert counts["embedding_retries"] == 1
    assert counts["embedding_requests"] == 51
    assert counts["embedding_texts_sent"] > sum(counts["chunks_by_org"].values())


def test_an_access_change_in_the_manifest_is_applied_in_place(owner, pg_url, tmp_path):
    load_corpus.load(pg_url, load_manifest(), embeddings=GatewayEmbeddings(), limiter=_fast(), token_limiter=_fast())
    original = (REPO / "kb" / "manifest.yaml").read_text(encoding="utf-8")
    changed = original.replace(
        "  - file: aurora/atlas-battery-spec.txt\n    org: aurora\n    team: null\n    clearance: 1",
        "  - file: aurora/atlas-battery-spec.txt\n    org: aurora\n    team: firmware\n    clearance: 2")
    assert changed != original
    path = tmp_path / "manifest.yaml"
    path.write_text(changed, encoding="utf-8")

    counts = load_corpus.load(pg_url, load_manifest(path, REPO / "kb" / "docs"),
                              embeddings=GatewayEmbeddings(), limiter=_fast(), token_limiter=_fast())

    assert (counts["update_access"], counts["skip"], counts["embedding_texts_sent"]) == (1, 49, 0)
    with owner.connect() as conn:
        rows = conn.execute(sa.text(
            "SELECT DISTINCT d.team_id, d.clearance, c.team_id, c.clearance FROM documents d "
            "JOIN chunks c USING (doc_id) WHERE d.filename = 'atlas-battery-spec.txt'")).all()
    assert rows == [("firmware", 2, "firmware", 2)]
