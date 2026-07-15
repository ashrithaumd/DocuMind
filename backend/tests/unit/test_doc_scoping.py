"""Doc scoping logic in retrieve_node: explicit doc_ids honored, default is
the single most recent upload, empty corpus scopes to nothing."""
import pytest

from graph import nodes


class FakeStore:
    def __init__(self, documents):
        self.documents = documents
        self.search_calls = []

    def list_documents(self):
        return self.documents

    async def search(self, query, doc_ids=None):
        self.search_calls.append(doc_ids)
        return []


@pytest.fixture
def store_with_two_docs(monkeypatch):
    store = FakeStore(
        [
            {"doc_id": "newer", "filename": "b.pdf", "uploaded_at": "2026-07-14", "chunks": 3},
            {"doc_id": "older", "filename": "a.pdf", "uploaded_at": "2026-07-01", "chunks": 5},
        ]
    )
    monkeypatch.setattr(nodes, "get_store", lambda: store)
    return store


async def test_default_scope_is_most_recent_upload(store_with_two_docs):
    update = await nodes.retrieve_node({"question": "q"})
    assert update["scoped_doc_ids"] == ["newer"]
    assert store_with_two_docs.search_calls == [["newer"]]


async def test_explicit_doc_ids_used_verbatim(store_with_two_docs):
    update = await nodes.retrieve_node({"question": "q", "doc_ids": ["older"]})
    assert update["scoped_doc_ids"] == ["older"]
    assert store_with_two_docs.search_calls == [["older"]]


async def test_multiple_doc_ids_passed_through(store_with_two_docs):
    update = await nodes.retrieve_node({"question": "q", "doc_ids": ["older", "newer"]})
    assert update["scoped_doc_ids"] == ["older", "newer"]


async def test_empty_corpus_scopes_to_nothing(monkeypatch):
    store = FakeStore([])
    monkeypatch.setattr(nodes, "get_store", lambda: store)
    update = await nodes.retrieve_node({"question": "q"})
    assert update["scoped_doc_ids"] == []
    assert store.search_calls == [None]  # no scope filter when there is nothing
    assert update["chunks"] == []


async def test_retrieved_chunk_text_is_redacted(monkeypatch):
    from ingestion.store import RetrievedChunk

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

    class Store(FakeStore):
        async def search(self, query, doc_ids=None):
            return [chunk]

    monkeypatch.setattr(
        nodes, "get_store",
        lambda: Store([{"doc_id": "doc1", "filename": "handbook.txt",
                        "uploaded_at": "2026-07-14", "chunks": 1}]),
    )
    update = await nodes.retrieve_node({"question": "q", "redaction_categories": []})
    text = update["chunks"][0]["text"]
    assert "hr@corp.com" not in text and "555-123-4567" not in text
    assert update["redaction_categories"] == ["email", "phone"]
