"""Migration 0002 on real Postgres + pgvector: the chunks table gets a
vector column sized from EMBEDDING_DIMENSIONS, HNSW/GIN/doc_id indexes, and
an 'english' tsvector; the log tables still match the models."""
import pytest
import sqlalchemy as sa
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext

from config import Settings
from db import session as db_session
from db.models import Base


def _migrate(url, **overrides) -> sa.Engine:
    assert db_session.init_db(Settings(_env_file=None, database_url=url, **overrides))
    return sa.create_engine(url)


def test_chunks_table_vector_size_follows_the_provider(pg_url):
    engine = _migrate(pg_url, embedding_provider="gemini")

    with engine.connect() as conn:
        column_type = conn.execute(sa.text(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
        )).scalar()
        indexes = {row[0] for row in conn.execute(sa.text(
            "SELECT indexdef FROM pg_indexes WHERE tablename = 'chunks'"
        ))}
    assert column_type == "vector(1536)"
    assert any("USING hnsw (embedding vector_cosine_ops)" in i for i in indexes)
    assert any("USING gin (tsv)" in i for i in indexes)
    assert any("(doc_id)" in i for i in indexes)


def test_local_provider_gets_a_384_dim_column(pg_url):
    engine = _migrate(pg_url, embedding_provider="local")

    with engine.connect() as conn:
        column_type = conn.execute(sa.text(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
        )).scalar()
    assert column_type == "vector(384)"


def test_tsvector_uses_english_stemming_and_stop_words(pg_url):
    engine = _migrate(pg_url, embedding_provider="local")

    with engine.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO chunks (chunk_id, doc_id, filename, page_number, chunk_index, text, "
            "embedding_model, embedding) VALUES ('d-0', 'd', 'f.pdf', 1, 0, "
            "'Cooling pumps were running at 400 RPM', 'm', :v)"
        ), {"v": "[" + ",".join(["0.1"] * 384) + "]"})
        tsv = conn.execute(sa.text("SELECT tsv::text FROM chunks")).scalar()
    assert tsv == "'400':6 'cool':1 'pump':2 'rpm':7 'run':4"


@pytest.mark.filterwarnings("ignore:Did not recognize type 'vector':sqlalchemy.exc.SAWarning")
def test_log_tables_still_match_the_models(pg_url):
    engine = _migrate(pg_url, embedding_provider="local")

    with engine.connect() as conn:
        diffs = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    # The chunks table (and its indexes) is raw SQL in the store, not an ORM
    # model, so autogenerate sees it as "not in the models".
    def on_chunks(diff):
        obj = diff[1]
        return getattr(obj, "name", None) == "chunks" or getattr(getattr(obj, "table", None), "name", None) == "chunks"

    assert [d for d in diffs if not on_chunks(d)] == []
