"""Scoping in retrieve_node: by default every document the caller may see,
in the version in force today; explicit doc_ids and filters pass through."""
from datetime import date

import pytest

from graph import nodes
from ingestion.store import Filters, RetrievedChunk


class FakeStore:
    def __init__(self, results=()):
        self.results = list(results)
        self.search_calls = []

    def scoped(self, access):
        return self

    async def search(self, query, doc_ids=None, filters=None):
        self.search_calls.append((doc_ids, filters))
        return self.results


@pytest.fixture
def store(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(nodes, "get_store", lambda: store)
    return store


async def test_default_scope_is_every_visible_document_in_force_today(store):
    update = await nodes.retrieve_node({"question": "q"})
    assert update["scoped_doc_ids"] == []
    [(doc_ids, filters)] = store.search_calls
    assert doc_ids is None
    assert filters == Filters() and filters.params()["as_of"] == date.today()


async def test_explicit_doc_ids_used_verbatim(store):
    update = await nodes.retrieve_node({"question": "q", "doc_ids": ["older", "newer"]})
    assert update["scoped_doc_ids"] == ["older", "newer"]
    assert store.search_calls[0][0] == ["older", "newer"]


async def test_filters_pass_through(store):
    asked = Filters(department="card_services", doc_type="policy", as_of=date(2025, 6, 1))
    await nodes.retrieve_node({"question": "q", "filters": asked})
    assert store.search_calls[0][1] == asked


async def test_retrieved_chunk_text_is_redacted(monkeypatch):
    chunk = RetrievedChunk(
        chunk_id="doc1-0000",
        text="HR contact is hr@corp.com, phone 555-123-4567.",
        filename="handbook.txt",
        page_number=1,
        score=0.5,
        dense_rank=1,
        sparse_rank=None,
        similarity=0.9,
    )
    monkeypatch.setattr(nodes, "get_store", lambda: FakeStore([chunk]))
    update = await nodes.retrieve_node({"question": "q", "redaction_categories": []})
    text = update["chunks"][0]["text"]
    assert "hr@corp.com" not in text and "555-123-4567" not in text
    assert update["redaction_categories"] == ["email", "phone"]
