"""The whole bank corpus loaded into real Postgres (bank/corpus.py load):
manifest metadata on documents, chunk metadata, tables, and the staff memo
invisible to customers. Re-loading replaces, never duplicates. Skipped
without TEST_POSTGRES_URL."""
import sqlalchemy as sa

from bank.corpus import load, load_corpus
from bank.synthetic import BANK_ORG
from config import Settings
from db import session as db_session
from ingestion.store import Access, HybridStore
from ingestion.tables import extract_tables
from tests.helpers import FakeEmbeddings
from tests.unit.test_metadata import MetadataLLM

ADMIN = Access(org_id=BANK_ORG, clearance=2, role="admin")
CUSTOMER = Access(org_id=BANK_ORG, clearance=0, customer_id="cust-0001")


def _store(pg_url, app_url) -> HybridStore:
    settings = Settings(_env_file=None, database_url=app_url, migration_database_url=pg_url,
                        embedding_dimensions=32)
    db_session.migrate_database(settings)
    engine = sa.create_engine(pg_url)
    with engine.begin() as conn:
        conn.execute(sa.text("INSERT INTO orgs VALUES (:o, 'FinPilot Bank') ON CONFLICT DO NOTHING"),
                     {"o": BANK_ORG})
    engine.dispose()
    return HybridStore(settings, FakeEmbeddings())


def _one(store, access, sql):
    with store.scoped(access)._begin() as conn:
        return conn.execute(sa.text(sql)).all()


async def test_the_corpus_loads_with_its_metadata_and_reloads_in_place(pg_url, app_url):
    store = _store(pg_url, app_url)
    docs = load_corpus(Settings(anthropic_api_key="x"))
    counts = await load(store.scoped(ADMIN), docs, llm=MetadataLLM(), model="m", cache={})
    assert counts["documents"] == len(docs) == 13
    chunks = counts["chunks"]

    [(n_docs, n_chunks)] = _one(store, ADMIN, "SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM chunks)")
    assert (n_docs, n_chunks) == (13, chunks)
    rows = _one(store, ADMIN, "SELECT doc_type, version, effective_date::text, superseded_on::text "
                              "FROM documents WHERE filename = 'fee-schedule-2025.md'")
    assert rows == [("fee_schedule", "2025", "2025-01-01", "2026-02-01")]
    assert _one(store, ADMIN, "SELECT count(*) FROM chunks WHERE summary = '' OR keywords = '{}' "
                              "OR questions = '{}' OR section = ''") == [(0,)]

    await load(store.scoped(ADMIN), docs)  # again, no metadata calls: replaced, not duplicated
    assert _one(store, ADMIN, "SELECT count(*) FROM chunks") == [(chunks,)]
    tables = sum(len(extract_tables(d.chunks)) for d in docs)
    assert _one(store, ADMIN, "SELECT count(*) FROM doc_tables") == [(tables,)]


async def test_customers_never_see_the_staff_memo_its_tables_or_its_canary(pg_url, app_url):
    store = _store(pg_url, app_url)
    await load(store.scoped(ADMIN), load_corpus(Settings(anthropic_api_key="x")))
    assert _one(store, ADMIN, "SELECT count(*) FROM chunks WHERE text LIKE '%FINPILOT-CANARY-%'") == [(1,)]
    assert _one(store, CUSTOMER, "SELECT count(*) FROM chunks WHERE text LIKE '%FINPILOT-CANARY-%'") == [(0,)]
    assert _one(store, CUSTOMER, "SELECT count(*) FROM documents WHERE filename LIKE 'internal/%'") == [(0,)]
    assert _one(store, CUSTOMER, "SELECT count(*) FROM doc_tables WHERE section LIKE '%Fraud%'") == [(0,)]
    assert _one(store, CUSTOMER, "SELECT count(*) FROM documents") == [(12,)]
    assert _one(store, ADMIN, "SELECT count(*) FROM documents WHERE clearance = 2") == [(1,)]
