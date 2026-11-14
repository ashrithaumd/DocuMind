"""Loading the synthetic customer data (bank/synthetic.py load) on real
Postgres: as the server's superuser (BYPASSRLS, like a Neon owner on a new
project) and as an owner that can't bypass RLS (the strictest Neon case).
Re-runs replace a customer's rows without touching anyone else's, and the
app role then sees each customer's own rows only."""
import pytest
import sqlalchemy as sa

from bank.synthetic import BANK_ORG, canary, generate, load
from db.access import Access, set_access

OWNED = ("customers", "accounts", "cards", "transactions", "fees", "holdings")


@pytest.fixture(scope="module")
def data():
    return generate(seed=11, customers=4)


def _counts(url, data) -> dict[str, int]:
    """Rows per table; owned tables counted customer by customer, inside each
    customer's context, so this works whether or not the role bypasses RLS."""
    engine = sa.create_engine(url)
    try:
        counts = {}
        for table in OWNED + ("securities", "prices"):
            with engine.begin() as conn:
                if table in OWNED:
                    total = 0
                    for cid in (c["customer_id"] for c in data["customers"]):
                        conn.execute(sa.text("SELECT set_config('app.customer_id', :c, true)"),
                                     {"c": cid})
                        total += conn.execute(sa.text(
                            f"SELECT count(*) FROM {table} WHERE customer_id = :c"), {"c": cid}).scalar()
                    counts[table] = total
                else:
                    counts[table] = conn.execute(sa.text(f"SELECT count(*) FROM {table}")).scalar()
        return counts
    finally:
        engine.dispose()


def _expected(data) -> dict[str, int]:
    return {t: len(data[t]) for t in OWNED + ("securities", "prices")}


def test_load_as_the_superuser_owner_and_reload(client, pg_url, data):
    load(pg_url, data)
    assert _counts(pg_url, data) == _expected(data)
    load(pg_url, data)  # idempotent
    assert _counts(pg_url, data) == _expected(data)


def test_reloading_one_customer_leaves_the_others_alone(client, pg_url, data):
    load(pg_url, data)
    first = data["customers"][0]["customer_id"]
    only_first = {t: [r for r in rows if r.get("customer_id", first) == first]
                  for t, rows in data.items()}
    load(pg_url, only_first)
    assert _counts(pg_url, data) == _expected(data)


def test_load_as_an_owner_that_cannot_bypass_rls(neon_like_owner_url, at, data):
    owner = sa.create_engine(neon_like_owner_url)
    at(owner, "head")
    owner.dispose()
    load(neon_like_owner_url, data)
    assert _counts(neon_like_owner_url, data) == _expected(data)
    load(neon_like_owner_url, data)
    assert _counts(neon_like_owner_url, data) == _expected(data)


def test_the_app_role_sees_each_customers_own_rows_only(client, pg_url, app_url, data):
    load(pg_url, data)
    app = sa.create_engine(app_url)
    try:
        for customer in data["customers"]:
            cid = customer["customer_id"]
            with app.begin() as conn:
                set_access(conn, Access(org_id=BANK_ORG, clearance=0, customer_id=cid))
                owners = {r[0] for r in conn.execute(sa.text("SELECT customer_id FROM transactions"))}
                text = " ".join(r[0] for r in conn.execute(
                    sa.text("SELECT description FROM transactions")))
            assert owners == {cid}
            others = [c["customer_id"] for c in data["customers"] if c["customer_id"] != cid]
            assert canary(cid) in text and not any(canary(o) in text for o in others)
    finally:
        app.dispose()
