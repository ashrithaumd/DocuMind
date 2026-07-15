"""Shared state flowing through the LangGraph pipeline.

Each node returns a partial update; LangGraph merges it into the full state.
`timings` uses dict-merge (operator.or_) as its reducer so every node can
contribute its own stage timing without clobbering the others.
"""
import operator
from typing import Annotated, TypedDict


class GraphState(TypedDict, total=False):
    question: str
    doc_ids: list[str] | None  # documents the caller asked to query (None = default)

    # input_validation
    rejection: str | None      # user-facing message when the query is rejected
    # PII categories masked so far (question + retrieved context); values are
    # names like "email", never the matched text
    redaction_categories: list[str]

    # route_query — "routed_direct" (small talk, pipeline skipped) or
    # "full_pipeline" (real document question)
    route: str

    # retrieve — the scope actually used: the requested doc_ids, or the most
    # recent upload when the caller didn't specify any
    scoped_doc_ids: list[str]

    # retrieve — each chunk is a plain dict (chunk_id, text, filename,
    # page_number, score, dense_rank, sparse_rank, similarity)
    chunks: list[dict]

    # grade
    grade_relevant: bool

    # generate
    answer: str
    citations: list[dict]      # [{n, chunk_id, filename, page_number, snippet}]
    regenerated: bool          # True once generation has run a second time

    # groundedness_check
    grounded: bool
    unsupported_claims: list[str]

    # terminal bookkeeping
    # one of: answered | refused_no_context | refused_ungrounded | rejected_input
    outcome: str
    timings: Annotated[dict[str, float], operator.or_]
