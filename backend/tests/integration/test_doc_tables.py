"""Document tables as rows (migration 0010) on real Postgres: written with
their document, linked to the chunk holding each row, under the documents'
access rules, and deleted with the document. Skipped without
TEST_POSTGRES_URL."""
import sqlalchemy as sa

from config import Settings
from db import session as db_session
from ingestion.chunking import chunk_blocks
from ingestion.parse import parse_markdown
from ingestion.store import Access, HybridStore
from ingestion.tables import extract_tables
from tests.helpers import FakeEmbeddings

ADMIN = Access(org_id="o1", clearance=2, role="admin")
CUSTOMER = Access(org_id="o1", clearance=0)

PUBLIC = """# Fees

| Service | Fee |
|---|---|
| Overdraft | $15.00 |
| Stop payment | $30.00 |
"""
STAFF = """# Rules

| Rule | Action |
|---|---|
| R1 velocity | Hold the card |
"""


def _store(pg_url, app_url) -> HybridStore:
    settings = Settings(_env_file=None, database_url=app_url, migration_database_url=pg_url,
                        embedding_dimensions=32)
    db_session.migrate_database(settings)
    engine = sa.create_engine(pg_url)
    with engine.begin() as conn:
        conn.execute(sa.text("INSERT INTO orgs VALUES ('o1', 'Org 1') ON CONFLICT DO NOTHING"))
    engine.dispose()
    return HybridStore(settings, FakeEmbeddings())


async def _add(store, text, doc_id, clearance):
    chunks = chunk_blocks(parse_markdown(text), f"{doc_id}.md", doc_id)
    await store.scoped(ADMIN).add_document(chunks, team_id=None, clearance=clearance,
                                           uploaded_by=None, tables=extract_tables(chunks))


def _rows(store, access):
    with store.scoped(access)._begin() as conn:
        return conn.execute(sa.text("""
            SELECT t.doc_id, t.columns, r.row_index, r.cells, r.chunk_id, c.chunk_id AS found
            FROM doc_table_rows r JOIN doc_tables t USING (table_id)
            LEFT JOIN chunks c ON c.chunk_id = r.chunk_id
            ORDER BY t.doc_id, r.row_index
        """)).mappings().all()


async def test_rows_follow_their_documents_access_and_link_to_chunks(pg_url, app_url):
    store = _store(pg_url, app_url)
    await _add(store, PUBLIC, "pub", clearance=0)
    await _add(store, STAFF, "staff", clearance=2)

    seen = _rows(store, CUSTOMER)
    assert [(r["doc_id"], r["row_index"], r["cells"]) for r in seen] == [
        ("pub", 0, ["Overdraft", "$15.00"]), ("pub", 1, ["Stop payment", "$30.00"])]
    assert all(r["columns"] == ["Service", "Fee"] for r in seen)
    assert all(r["chunk_id"] == r["found"] == "pub-0000" for r in seen)

    assert {r["doc_id"] for r in _rows(store, ADMIN)} == {"pub", "staff"}
    assert _rows(store, Access(org_id="o2", clearance=2)) == []


async def test_deleting_a_document_deletes_its_tables(pg_url, app_url):
    store = _store(pg_url, app_url)
    await _add(store, PUBLIC, "pub", clearance=0)
    with store.scoped(ADMIN)._begin() as conn:
        conn.execute(sa.text("DELETE FROM chunks WHERE doc_id = 'pub'"))
        conn.execute(sa.text("DELETE FROM documents WHERE doc_id = 'pub'"))
        assert conn.execute(sa.text("SELECT count(*) FROM doc_tables")).scalar() == 0
        assert conn.execute(sa.text("SELECT count(*) FROM doc_table_rows")).scalar() == 0
