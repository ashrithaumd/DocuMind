"""Receipt extraction stored on real Postgres (receipt_images, migration
0010): written inside each receipt's customer context, so it works for an
owner that can't bypass RLS; a second run makes no vision calls; the app
role reads a receipt only as its customer. Skipped without
TEST_POSTGRES_URL."""
import sqlalchemy as sa

from bank.receipts import extract_all, pick, render
from bank.synthetic import BANK_ORG, generate, load
from db.access import Access, set_access
from tests.helpers import FakeLLM


def _receipts(owner_url, tmp_path):
    data = generate(seed=11, customers=3)
    load(owner_url, data)
    truths = pick(data)
    for truth in truths:
        (tmp_path / f"{truth['txn_id']}.png").write_bytes(render(truth))
    return truths


async def test_an_owner_without_bypassrls_stores_and_reuses_extractions(neon_like_owner_url, at, tmp_path):
    owner = sa.create_engine(neon_like_owner_url)
    at(owner, "head")
    owner.dispose()
    truths = _receipts(neon_like_owner_url, tmp_path)
    assert len(truths) == 3

    llm = FakeLLM()
    first = truths[0]
    llm.receipt = {"merchant": first["merchant"], "date": first["date"],
                   "total": f"{first['amount_cents'] / 100:.2f}", "card_last4": first["card_last4"]}
    results = await extract_all(neon_like_owner_url, llm, "vision-x", truths, folder=tmp_path)
    assert [r["cached"] for r in results] == [False] * 3
    assert all(results[0]["matches"].values())
    assert len(llm.chat_calls) == 3 and all(c["image"] for c in llm.chat_calls)

    again = await extract_all(neon_like_owner_url, llm, "vision-x", truths, folder=tmp_path)
    assert [r["cached"] for r in again] == [True] * 3
    assert again[0]["fields"] == llm.receipt
    assert len(llm.chat_calls) == 3  # nothing sent the second time


async def test_the_app_role_reads_a_receipt_only_as_its_customer(client, pg_url, app_url, tmp_path):
    truths = _receipts(pg_url, tmp_path)
    await extract_all(pg_url, FakeLLM(), "vision-x", truths, folder=tmp_path)
    app = sa.create_engine(app_url)
    try:
        for truth in truths:
            with app.begin() as conn:
                set_access(conn, Access(org_id=BANK_ORG, clearance=0, customer_id=truth["customer_id"]))
                rows = conn.execute(sa.text("SELECT customer_id, txn_id FROM receipt_images")).all()
            assert rows == [(truth["customer_id"], truth["txn_id"])]
        with app.begin() as conn:
            set_access(conn, Access(org_id=BANK_ORG, clearance=2))  # staff: no customer
            assert conn.execute(sa.text("SELECT count(*) FROM receipt_images")).scalar() == 0
    finally:
        app.dispose()
