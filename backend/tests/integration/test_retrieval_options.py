"""Phase 4 retrieval on real Postgres: metadata filters (current version by
default), the hypothetical-question leg, dense-only mode and parent-section
expansion. Skipped without TEST_POSTGRES_URL."""
from datetime import date

import sqlalchemy as sa

from ingestion.chunking import chunk_blocks
from ingestion.parse import parse_markdown
from ingestion.store import Filters
from tests.integration.test_corpus_load import ADMIN, _one, _store

FEES = "# Fees\n\n## 1. Overdraft\n\nThe overdraft fee is {fee} per item, at most three a day.\n"
CARD = """# Card Agreement

## 1. Rewards

Cash back is 1.5% on every purchase, paid monthly.

## 2. Fees

### 2.1 Late payment

A late payment costs $29 if the minimum payment is not received.

### 2.2 Foreign transactions

Purchases in another currency carry a 3% foreign transaction fee.
"""
ABROAD = "Do I pay extra when I shop abroad?"


async def _load(store):
    docs = [("f25", FEES.format(fee="$35"), {"doc_type": "fee_schedule", "department": "retail_banking",
                                             "effective_date": date(2025, 1, 1), "superseded_on": date(2026, 2, 1)}),
            ("f26", FEES.format(fee="$15"), {"doc_type": "fee_schedule", "department": "retail_banking",
                                             "effective_date": date(2026, 2, 1)}),
            ("card", CARD, {"doc_type": "agreement", "department": "card_services"})]
    for doc_id, text, meta in docs:
        chunks = chunk_blocks(parse_markdown(text), f"{doc_id}.md", doc_id)
        for c in chunks:
            c.questions = [ABROAD] if "Foreign" in c.section else [f"What does {c.section} say?"]
        await store.scoped(ADMIN).add_document(chunks, team_id=None, clearance=0, uploaded_by=None, meta=meta)


async def _docs(store, query, **filters):
    hits = await store.scoped(ADMIN).search(query, filters=Filters(**filters))
    return {h.doc_id for h in hits}


async def test_filters_default_to_the_version_in_force_today(pg_url, app_url):
    store = _store(pg_url, app_url)
    await _load(store)
    assert {"f26"} <= await _docs(store, "overdraft fee") and "f25" not in await _docs(store, "overdraft fee")
    assert "f26" not in await _docs(store, "overdraft fee", as_of=date(2025, 6, 1))
    assert {"f25", "f26"} <= await _docs(store, "overdraft fee", all_versions=True)
    assert await _docs(store, "fee", department="card_services") == {"card"}
    assert await _docs(store, "fee", doc_type="fee_schedule", all_versions=True) == {"f25", "f26"}


async def test_the_question_leg_finds_a_chunk_by_its_hypothetical_question(pg_url, app_url):
    store = _store(pg_url, app_url)
    await _load(store)
    assert _one(store, ADMIN, "SELECT count(*) FROM chunks WHERE question_embedding IS NULL") == [(0,)]
    hits = await store.scoped(ADMIN).search(ABROAD)
    by_question = min((h for h in hits if h.question_rank), key=lambda h: h.question_rank)
    assert by_question.question_rank == 1 and "Foreign" in by_question.section

    dense = await store.configured(retrieval_mode="dense").scoped(ADMIN).search(ABROAD)
    assert dense and all(h.sparse_rank is None and h.question_rank is None for h in dense)


async def test_parent_expansion_gives_the_whole_top_level_section(pg_url, app_url):
    store = _store(pg_url, app_url)
    await _load(store)
    expanded = store.configured(parent_expansion=True, top_k=10).scoped(ADMIN)
    hits = await expanded.search("foreign transaction fee currency", filters=Filters(department="card_services"))
    [fees] = [h for h in hits if "2. Fees" in h.section]  # its sibling's hit isn't repeated
    with store.scoped(ADMIN)._begin() as conn:
        sections = dict(conn.execute(sa.text(
            "SELECT chunk_id, section FROM chunks WHERE chunk_id = ANY(:ids)"),
            {"ids": fees.context_chunk_ids}).all())
    assert set(sections.values()) == {"Card Agreement > 2. Fees > 2.1 Late payment",
                                      "Card Agreement > 2. Fees > 2.2 Foreign transactions"}
    assert "late payment costs $29" in fees.text and "3% foreign" in fees.text
    assert "Cash back" not in fees.text
