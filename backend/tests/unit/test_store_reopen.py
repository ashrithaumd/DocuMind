"""Reopening a store whose index already holds vectors: the startup
dimension check (the only path that queries the embedder's dimension) must
pass for the same model and fail loudly for a different one."""
import pytest

from config import Settings
from ingestion.chunking import Chunk
from ingestion.store import HybridStore
from tests.helpers import FakeEmbedder


class SixteenDimEmbedder(FakeEmbedder):
    """A different embedding model: same interface, other dimension."""

    def get_embedding_dimension(self) -> int:
        return 16


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        chroma_host=None,
        chroma_dir=str(tmp_path / "chroma"),
        collection_name="reopen_test",
    )


async def _populated(settings) -> None:
    store = HybridStore(settings, FakeEmbedder())
    await store.add_chunks(
        [Chunk(chunk_id="d1-0", doc_id="d1", filename="a.pdf", page_number=1,
               chunk_index=0, text="The pump runs at 400 RPM.")]
    )


async def test_reopening_populated_index_with_same_model(settings):
    await _populated(settings)

    reopened = HybridStore(settings, FakeEmbedder())  # runs the dimension check

    assert reopened.count() == 1


async def test_reopening_populated_index_with_other_model_fails_loudly(settings):
    await _populated(settings)

    with pytest.raises(RuntimeError, match="holds 32-dim vectors.*produces 16-dim"):
        HybridStore(settings, SixteenDimEmbedder())
