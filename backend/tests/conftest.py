"""Fixtures shared by unit and integration tests.

The app uses three lru_cache singletons (settings, LLM gateway, store) plus
a module-global DB session factory. The `client` fixture resets all of them
around a temp environment: embedded Chroma in tmp_path, SQLite standing in
for Postgres (same SQLAlchemy models and queries — dialect-neutral SQL),
FakeLLM in place of the LLM gateway, and FakeEmbedder in place of the local
sentence-transformers model — so no test ever makes a network call, loads
torch, or spends a token.
"""
import pytest

import config
import deps
from db import session as db_session
from graph import nodes
from tests.helpers import FakeEmbedder, FakeLLM


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def app_env(tmp_path, monkeypatch, fake_llm):
    """Fresh settings + stubbed LLM/embedder + temp Chroma/SQLite for one test."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.delenv("GENERATION_MODEL", raising=False)
    monkeypatch.delenv("GRADING_MODEL", raising=False)
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'log.db'}")
    monkeypatch.setenv("COLLECTION_NAME", "test_documents")
    monkeypatch.delenv("CHROMA_HOST", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)

    config.get_settings.cache_clear()
    deps.get_llm.cache_clear()
    deps.get_embedder.cache_clear()
    deps.get_store.cache_clear()

    # Both the graph nodes (get_llm) and the store factory (get_embedder)
    # must see the fakes. Embedder activity is mirrored onto the FakeLLM so
    # tests can assert "zero model traffic" via one object.
    embedder = FakeEmbedder(record_into=fake_llm.embed_calls)
    monkeypatch.setattr(deps, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(nodes, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(deps, "get_embedder", lambda: embedder)

    yield fake_llm

    # Next test rebuilds everything from its own tmp_path.
    config.get_settings.cache_clear()
    deps.get_store.cache_clear()
    db_session._session_factory = None


@pytest.fixture
def client(app_env):
    """TestClient with lifespan (DB init + store build) executed."""
    from fastapi.testclient import TestClient

    from main import app

    with TestClient(app) as test_client:
        yield test_client
