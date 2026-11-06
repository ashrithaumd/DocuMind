"""Fixtures shared by unit and integration tests.

The app uses lru_cache singletons (settings, LLM gateway, embedding
provider, store) plus a module-global DB session factory. The `client`
fixture resets all of them around a temp environment: a fresh Postgres +
pgvector database per test (pg_url, from TEST_POSTGRES_URL; those tests skip
without it), FakeLLM in place of the LLM gateway, and FakeEmbeddings in place
of the embedding provider, so no test ever calls a model, loads torch, or
spends a token. Unit tests that only need the log tables use SQLite.
"""
import pytest

import config
import deps
from db import session as db_session
from graph import nodes
from tests.helpers import FakeEmbeddings, FakeLLM


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def app_env(pg_url, monkeypatch, fake_llm):
    """Fresh settings + stubbed LLM/embeddings + a fresh Postgres database."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.delenv("GENERATION_MODEL", raising=False)
    monkeypatch.delenv("GRADING_MODEL", raising=False)
    monkeypatch.setenv("DATABASE_URL", pg_url)
    monkeypatch.delenv("API_KEY", raising=False)
    # FakeEmbeddings stands in for the provider; the chunks table must be
    # sized for its 32-dim vectors.
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "32")

    config.get_settings.cache_clear()
    deps.get_llm.cache_clear()
    deps.get_embeddings.cache_clear()
    deps.get_store.cache_clear()

    # Both the graph nodes (get_llm) and the store factory (get_embeddings)
    # must see the fakes. Embedder activity is mirrored onto the FakeLLM so
    # tests can assert "zero model traffic" via one object.
    embeddings = FakeEmbeddings(record_into=fake_llm.embed_calls)
    monkeypatch.setattr(deps, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(nodes, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(deps, "get_embeddings", lambda: embeddings)

    yield fake_llm

    # Next test rebuilds everything against its own database.
    config.get_settings.cache_clear()
    if deps.get_store.cache_info().currsize:
        deps.get_store()._engine.dispose()  # release pooled connections to the test database
    deps.get_store.cache_clear()
    db_session._session_factory = None


@pytest.fixture
def client(app_env):
    """TestClient with lifespan (DB init + store build) executed."""
    from fastapi.testclient import TestClient

    from main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def pg_url():
    """URL of a fresh, empty Postgres database (with pgvector available),
    dropped afterwards. Needs TEST_POSTGRES_URL, an admin URL to a pgvector
    server (CI runs one as a service container); skipped without it."""
    import os
    import uuid

    import sqlalchemy as sa

    admin_url = os.environ.get("TEST_POSTGRES_URL")
    if not admin_url:
        if os.environ.get("REQUIRE_POSTGRES_TESTS"):  # CI: a skip would hide the tests
            pytest.fail("REQUIRE_POSTGRES_TESTS is set but TEST_POSTGRES_URL is not")
        pytest.skip("TEST_POSTGRES_URL not set (needs a pgvector Postgres)")
    name = f"documind_test_{uuid.uuid4().hex[:12]}"
    admin = sa.create_engine(admin_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(sa.text(f'CREATE DATABASE "{name}"'))
    try:
        yield sa.engine.make_url(admin_url).set(database=name).render_as_string(hide_password=False)
    finally:
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()
