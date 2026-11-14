"""scripts/leakage_check.py on real Postgres: clean on a correctly loaded
database, and it does catch a leak (row-level security switched off on one
table) and an owner connection. Skipped without TEST_POSTGRES_URL."""
import sys
from pathlib import Path

import sqlalchemy as sa

from bank import synthetic
from bank.corpus import load, load_corpus
from config import Settings
from tests.integration.test_corpus_load import ADMIN, _store

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from leakage_check import check  # noqa: E402


async def test_clean_database_passes_and_leaks_are_caught(pg_url, app_url):
    store = _store(pg_url, app_url)  # migrates, as the owner
    data = synthetic.generate(customers=3)
    synthetic.load(pg_url, data)
    await load(store.scoped(ADMIN), load_corpus(Settings(anthropic_api_key="x")))
    ids = [c["customer_id"] for c in data["customers"]]

    assert check(app_url, ids) == []

    assert any("would bypass" in p for p in check(pg_url, ids))  # the owner sees everything

    engine = sa.create_engine(pg_url)
    with engine.begin() as conn:
        conn.execute(sa.text("ALTER TABLE accounts DISABLE ROW LEVEL SECURITY"))
    engine.dispose()
    problems = check(app_url, ids)
    assert "accounts: row-level security is off" in problems  # still checked, not dropped
    assert any(p.startswith("no context: accounts shows") for p in problems)
    assert any("rows of other customers in accounts" in p for p in problems)
