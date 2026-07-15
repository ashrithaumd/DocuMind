"""Audit / observability schema.

One row per query answers "how would you detect a retrieval failure in
production?": it records what was retrieved (with scores), what the checker
models decided, which outcome the user got, and where the time went.
The same table doubles as chat history (question + answer + timestamp).
"""
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class QueryLog(Base):
    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)

    # answered | small_talk | refused_no_context | refused_ungrounded |
    # rejected_input | error
    outcome: Mapped[str] = mapped_column(String(32), index=True)

    # Which path the query took: full_pipeline | routed_direct.
    # Null for queries rejected before routing (and pre-routing rows).
    route: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # [{chunk_id, page_number, score, similarity, dense_rank, sparse_rank}]
    retrieved_chunks: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # doc_ids retrieval was actually scoped to (requested, or the default
    # most-recent upload). Null for queries that never reached retrieval.
    doc_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # PII masking: whether any redaction happened, and which categories
    # (e.g. ["email", "ssn"]). Never the matched values. The `question`
    # column itself stores the REDACTED text.
    redacted: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    redaction_categories: Mapped[list | None] = mapped_column(JSON, nullable=True)

    grade_relevant: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    grounded: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    regenerated: Mapped[bool] = mapped_column(Boolean, default=False)

    retrieval_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    grading_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    generation_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    groundedness_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "question": self.question,
            "answer": self.answer,
            "outcome": self.outcome,
            "route": self.route,
            "retrieved_chunks": self.retrieved_chunks,
            "doc_ids": self.doc_ids,
            "redacted": self.redacted,
            "redaction_categories": self.redaction_categories,
            "grade_relevant": self.grade_relevant,
            "grounded": self.grounded,
            "regenerated": self.regenerated,
            "timings": {
                "retrieval_ms": self.retrieval_ms,
                "grading_ms": self.grading_ms,
                "generation_ms": self.generation_ms,
                "groundedness_ms": self.groundedness_ms,
                "total_ms": self.total_ms,
            },
        }


class Feedback(Base):
    """User verdict on one answer. Exactly one row per query (upserted), so a
    user adding a comment or changing their mind updates rather than
    double-counts — that keeps the /admin/stats aggregates honest."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    query_log_id: Mapped[int] = mapped_column(
        ForeignKey("query_logs.id"), unique=True, index=True
    )
    thumbs_up: Mapped[bool] = mapped_column(Boolean)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "query_log_id": self.query_log_id,
            "thumbs_up": self.thumbs_up,
            "comment": self.comment,
        }
