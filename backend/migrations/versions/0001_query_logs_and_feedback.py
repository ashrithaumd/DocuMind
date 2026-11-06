"""query_logs and feedback: the schema as of Phase 0.

Matches what Base.metadata.create_all built before Alembic, index names
included, so databases created that way are adopted by stamping this
revision (db.session.init_db) instead of being rebuilt.

Revision ID: 0001
Revises:
"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "query_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                  nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("outcome", sa.String(32), nullable=False),
        sa.Column("route", sa.String(16), nullable=True),
        sa.Column("retrieved_chunks", sa.JSON(), nullable=True),
        sa.Column("doc_ids", sa.JSON(), nullable=True),
        sa.Column("redacted", sa.Boolean(), nullable=True),
        sa.Column("redaction_categories", sa.JSON(), nullable=True),
        sa.Column("grade_relevant", sa.Boolean(), nullable=True),
        sa.Column("grounded", sa.Boolean(), nullable=True),
        sa.Column("regenerated", sa.Boolean(), nullable=False),
        sa.Column("retrieval_ms", sa.Float(), nullable=True),
        sa.Column("grading_ms", sa.Float(), nullable=True),
        sa.Column("generation_ms", sa.Float(), nullable=True),
        sa.Column("groundedness_ms", sa.Float(), nullable=True),
        sa.Column("total_ms", sa.Float(), nullable=True),
    )
    op.create_index("ix_query_logs_created_at", "query_logs", ["created_at"])
    op.create_index("ix_query_logs_outcome", "query_logs", ["outcome"])

    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                  nullable=False),
        sa.Column("query_log_id", sa.Integer(), sa.ForeignKey("query_logs.id"), nullable=False),
        sa.Column("thumbs_up", sa.Boolean(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
    )
    op.create_index("ix_feedback_query_log_id", "feedback", ["query_log_id"], unique=True)


def downgrade() -> None:
    raise NotImplementedError("forward-only")
