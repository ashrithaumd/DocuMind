"""Phase 4 retrieval options without a database: the question leg in RRF,
the metadata filters' defaults, parent-section expansion and reranking."""
from datetime import date

from config import Settings
from ingestion.chunking import tokens
from ingestion.store import Filters, HybridStore, RetrievedChunk, expand_to_parents, rrf_fuse


def _meta(cid, section="Doc > 1. A"):
    return {"filename": "doc.md", "page_number": 1, "doc_id": "d", "section": section,
            "chunk_index": int(cid[-1])}


def test_the_question_leg_adds_to_the_fused_score():
    dense = {"d-1": (1, 0.9, "one", _meta("d-1")), "d-2": (2, 0.8, "two", _meta("d-2"))}
    questions = {"d-2": (1, "two", _meta("d-2")), "d-3": (2, "three", _meta("d-3"))}
    fused = rrf_fuse(dense, {}, 60, questions)
    assert [c.chunk_id for c in fused] == ["d-2", "d-1", "d-3"]  # d-2: two legs beat one
    assert (fused[0].dense_rank, fused[0].question_rank) == (2, 1)
    assert fused[2].dense_rank is None and fused[2].question_rank == 2


def test_filters_default_to_the_versions_in_force_today():
    assert Filters().params() == {"department": None, "doc_type": None, "as_of": date.today()}
    assert Filters(as_of=date(2025, 6, 1)).params()["as_of"] == date(2025, 6, 1)
    assert Filters(all_versions=True).params()["as_of"] is None


def _hit(cid, section="Doc > 1. A"):
    return RetrievedChunk(chunk_id=cid, text=f"text {cid}", score=1.0, dense_rank=1, sparse_rank=None,
                          similarity=0.9, **_meta(cid, section))


def _row(cid, section, words):
    return {"chunk_id": cid, "doc_id": "d", "section": section,
            "text": f"{section}\n\n" + " ".join(["word"] * words)}


def test_a_hit_grows_into_its_parent_section_within_the_cap():
    rows = [_row("d-0", "Doc > 1. A", 100), _row("d-1", "Doc > 1. A > 1.1 B", 100),
            _row("d-2", "Doc > 1. A > 1.2 C", 100), _row("d-3", "Doc > 1. A > 1.3 D", 100),
            _row("d-4", "Doc > 2. E", 100)]
    cap = tokens("\n\n".join(r["text"] for r in rows[1:4]))  # room for exactly three
    [hit] = expand_to_parents([_hit("d-2", "Doc > 1. A > 1.2 C")], rows, max_tokens=cap)
    assert hit.context_chunk_ids == ["d-1", "d-2", "d-3"]  # after, then before, contiguous; never 2. E
    assert tokens(hit.text) <= cap and hit.chunk_id == "d-2"
    [whole] = expand_to_parents([_hit("d-2", "Doc > 1. A > 1.2 C")], rows, max_tokens=10_000)
    assert whole.context_chunk_ids == ["d-0", "d-1", "d-2", "d-3"]


def test_a_second_hit_in_the_same_parent_is_dropped_and_sectionless_hits_stay():
    rows = [_row("d-0", "Doc > 1. A", 50), _row("d-1", "Doc > 1. A", 50)]
    first, upload = expand_to_parents([_hit("d-0"), _hit("d-1"), _hit("u-9", section="")], rows, 1500)
    assert first.context_chunk_ids == ["d-0", "d-1"]
    assert first.text.count("Doc > 1. A") == 1  # the path line once per section
    assert (upload.chunk_id, upload.text, upload.context_chunk_ids) == ("u-9", "text u-9", ["u-9"])


class FakeReranker:
    def scores(self, query, texts):
        return [float(len(t)) for t in texts]  # longest text wins


def test_rerank_reorders_the_fused_candidates_then_cuts_to_top_k():
    store = object.__new__(HybridStore)  # no database: _fused_ranking is replaced
    store._settings = Settings(_env_file=None, rerank=True, top_k=2, rerank_candidates=3)
    store._reranker = FakeReranker()
    fused = [_hit("d-1"), _hit("d-2"), _hit("d-3"), _hit("d-4")]
    fused[1].text, fused[2].text, fused[3].text = "longer text", "the longest text", "longest of all, never scored"
    store._fused_ranking = lambda *args: fused
    hits = store._search("q", [0.0], None, None)
    assert [h.chunk_id for h in hits] == ["d-3", "d-2"]  # d-4 was outside rerank_candidates
    assert hits[0].rerank_score == len("the longest text")
    assert hits[0].context_chunk_ids == ["d-3"]
