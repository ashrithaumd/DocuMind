"""Chunking edge cases: empty, tiny, and boundary-sized documents."""
import tiktoken

from config import Settings
from ingestion.chunking import chunk_pages
from ingestion.extract import PageText

SETTINGS = Settings(anthropic_api_key="x")
ENCODING = tiktoken.get_encoding("cl100k_base")


def _tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def test_empty_document_yields_no_chunks():
    assert chunk_pages([], "empty.pdf", SETTINGS) == []


def test_page_with_empty_text_yields_no_chunks():
    # Extraction never emits blank pages, but chunking must not crash on them.
    assert chunk_pages([PageText(1, "")], "blank.pdf", SETTINGS) == []


def test_single_sentence_document():
    chunks = chunk_pages([PageText(1, "The pump runs at 400 RPM.")], "one.txt", SETTINGS)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text == "The pump runs at 400 RPM."
    assert chunk.page_number == 1
    assert chunk.chunk_index == 0
    assert chunk.filename == "one.txt"
    assert chunk.chunk_id == f"{chunk.doc_id}-0000"


def test_exactly_boundary_sized_page_stays_one_chunk():
    # Build a text of exactly chunk_size_tokens tokens.
    word = "reactor "
    text = word * 400
    while _tokens(text) > SETTINGS.chunk_size_tokens:
        text = text[: text.rfind(" reactor")]
    assert _tokens(text) <= SETTINGS.chunk_size_tokens

    chunks = chunk_pages([PageText(1, text)], "boundary.txt", SETTINGS)
    assert len(chunks) == 1


def test_oversized_page_splits_with_overlap():
    sentences = [f"Fact number {i} lives in this sentence." for i in range(200)]
    text = " ".join(sentences)
    assert _tokens(text) > SETTINGS.chunk_size_tokens

    chunks = chunk_pages([PageText(1, text)], "big.txt", SETTINGS)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert _tokens(chunk.text) <= SETTINGS.chunk_size_tokens
        assert chunk.page_number == 1  # per-page splitting: page never changes
    # 15% overlap: consecutive chunks share text.
    assert chunks[0].text[-30:] in chunks[0].text  # sanity
    tail = chunks[0].text.split(". ")[-2]
    assert tail in chunks[1].text


def test_multi_page_indexing_is_global():
    pages = [PageText(1, "Page one text, long enough to matter."),
             PageText(2, "Page two text, also long enough.")]
    chunks = chunk_pages(pages, "two.pdf", SETTINGS)
    assert [c.chunk_index for c in chunks] == [0, 1]
    assert [c.page_number for c in chunks] == [1, 2]
    assert len({c.doc_id for c in chunks}) == 1
