"""RRF fusion and its explicit tie-break rule: equal fused scores order a
dense-leg chunk first (by dense rank), then sparse-only chunks (by sparse
rank). Run A's Q7/Q11 case: mirrored ranks (dense 1 + sparse 2 vs dense 2 +
sparse 1) give exactly equal scores."""
from ingestion.store import rrf_fuse

META = {"filename": "f.pdf", "page_number": 1}


def _dense(*ids):
    return {cid: (rank, 0.5, "", META) for rank, cid in enumerate(ids, start=1)}


def _sparse(*ids):
    return {cid: (rank, "", META) for rank, cid in enumerate(ids, start=1)}


def test_scores_add_across_legs():
    ranking = rrf_fuse(_dense("a", "b"), _sparse("b"), k=60)

    assert [c.chunk_id for c in ranking] == ["b", "a"]
    assert ranking[0].score == 1 / 62 + 1 / 61
    assert (ranking[0].dense_rank, ranking[0].sparse_rank) == (2, 1)


def test_mirrored_ranks_tie_exactly_and_dense_rank_breaks_it():
    # x: dense 1, sparse 2. y: dense 2, sparse 1. Same fused score.
    ranking = rrf_fuse(_dense("x", "y"), _sparse("y", "x"), k=60)

    assert ranking[0].score == ranking[1].score
    assert [c.chunk_id for c in ranking] == ["x", "y"]


def test_dense_found_chunk_beats_sparse_only_chunk_on_a_tie():
    # d: dense-only at rank 1; s: sparse-only at rank 1. Same score 1/61.
    ranking = rrf_fuse(_dense("d"), _sparse("s"), k=60)

    assert ranking[0].score == ranking[1].score
    assert [c.chunk_id for c in ranking] == ["d", "s"]


def test_sparse_only_ties_follow_sparse_rank():
    ranking = rrf_fuse(_dense("d1", "d2"), _sparse("s1", "s2", "d1", "d2"), k=60)

    sparse_only = [c.chunk_id for c in ranking if c.dense_rank is None]
    assert sparse_only == ["s1", "s2"]


def test_order_does_not_depend_on_dict_insertion_order():
    forward = rrf_fuse(_dense("x", "y"), _sparse("y", "x"), k=60)
    reversed_dense = rrf_fuse(dict(reversed(list(_dense("x", "y").items()))),
                              _sparse("y", "x"), k=60)

    assert [c.chunk_id for c in forward] == [c.chunk_id for c in reversed_dense]
