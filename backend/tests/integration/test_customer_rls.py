"""Customer data under per-customer row-level security (migration 0009), on
real Postgres, with the app connected as finpilot_app.

Seeded by the owner: three customers of one bank. c1 and c2 each have an
account, a card, transactions whose merchant and description carry a canary
naming the customer, a fee and a holding; c3 has a customer row only.
A customer must see exactly their own rows, in every owned table, through
every query shape, and never another customer's canary. No context, or a
staff context (no customer_id), sees nothing. The app role can't write.
"""
import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError, ProgrammingError

from db.access import Access, set_access

BANK = "bank"
CUSTOMERS = ("c1", "c2", "c3")
WITH_DATA = ("c1", "c2")
OWNED = ("customers", "accounts", "cards", "transactions", "fees", "holdings")
REFERENCE = ("securities", "prices")


def canary(cid: str) -> str:
    return f"CANARY-{cid}-7f3a"


def as_customer(cid: str | None) -> Access:
    return Access(org_id=BANK, clearance=0, customer_id=cid)


STAFF = Access(org_id=BANK, clearance=2, role="admin")  # no customer_id


@pytest.fixture
def owner(pg_url):
    engine = sa.create_engine(pg_url)
    yield engine
    engine.dispose()


@pytest.fixture
def seeded(client, owner):
    """Needs `client` first: its startup runs the migrations."""
    with owner.begin() as conn:
        conn.execute(sa.text("INSERT INTO orgs VALUES (:b, 'FinPilot Bank')"), {"b": BANK})
        conn.execute(sa.text("INSERT INTO securities VALUES ('VTI', 'Total Market ETF', 'etf', 0.0003)"))
        conn.execute(sa.text("INSERT INTO prices VALUES ('VTI', '2026-11-02', 301.25, 'synthetic')"))
        for cid in CUSTOMERS:
            conn.execute(sa.text(
                "INSERT INTO customers VALUES (:c, :b, :name, 'retail', '2024-01-15')"),
                {"c": cid, "b": BANK, "name": f"Customer {cid}"})
        for cid in WITH_DATA:
            p = {"c": cid, "a": f"{cid}-chk", "b": f"{cid}-brk", "card": f"{cid}-card",
                 "canary": canary(cid)}
            conn.execute(sa.text("""
                INSERT INTO accounts (account_id, customer_id, type, product, opened_on) VALUES
                    (:a, :c, 'checking', 'Everyday Checking', '2024-01-15'),
                    (:b, :c, 'brokerage', 'Self-Directed', '2024-02-01')
            """), p)
            conn.execute(sa.text("""
                INSERT INTO cards (card_id, account_id, customer_id, last4, network, expires_on)
                VALUES (:card, :a, :c, '4242', 'visa', '2029-01-31')
            """), p)
            for i, amount in enumerate((-1250, -4999, 250000)):
                conn.execute(sa.text("""
                    INSERT INTO transactions (txn_id, account_id, customer_id, card_id, posted_on,
                                              amount_cents, merchant, category, description)
                    VALUES (:t, :a, :c, :card, CAST(:d AS date), :amount, :canary, 'shopping',
                            'Purchase at ' || :canary)
                """), p | {"t": f"{cid}-t{i}", "d": f"2026-10-0{i + 1}", "amount": amount})
            conn.execute(sa.text("""
                INSERT INTO fees (fee_id, account_id, customer_id, txn_id, fee_type, amount_cents,
                                  assessed_on)
                VALUES (:c || '-f0', :a, :c, :c || '-t1', 'overdraft', 3400, '2026-10-02')
            """), p)
            conn.execute(sa.text("""
                INSERT INTO holdings VALUES (:b, :c, 'VTI', 12.5, 310000)
            """), p)


@pytest.fixture
def app_engine(seeded, app_url):
    engine = sa.create_engine(app_url, pool_size=1, max_overflow=0)
    yield engine
    engine.dispose()


def _run(engine, access, sql, **params):
    with engine.begin() as conn:
        if access is not None:
            set_access(conn, access)
        return conn.execute(sa.text(sql), params).all()


# --- each customer sees exactly their own rows -----------------------------------

@pytest.mark.parametrize("cid", WITH_DATA)
@pytest.mark.parametrize("table", OWNED)
def test_a_customer_sees_only_their_own_rows(app_engine, table, cid):
    owners = {r[0] for r in _run(app_engine, as_customer(cid), f"SELECT customer_id FROM {table}")}
    assert owners == {cid}


def test_a_customer_with_no_data_sees_only_their_customer_row(app_engine):
    for table in OWNED:
        rows = _run(app_engine, as_customer("c3"), f"SELECT customer_id FROM {table}")
        assert rows == ([("c3",)] if table == "customers" else []), table


@pytest.mark.parametrize("table", OWNED)
def test_no_customer_canary_ever_crosses(app_engine, table):
    """Every column of every visible row, as text: c1 never sees c2's canary."""
    text = " ".join(r[0] for r in _run(app_engine, as_customer("c1"),
                                       f"SELECT CAST(t AS text) FROM {table} t"))
    assert canary("c2") not in text


@pytest.mark.parametrize("sql", [
    "SELECT txn_id FROM transactions WHERE customer_id = 'c2'",
    "SELECT txn_id FROM transactions WHERE merchant LIKE 'CANARY-c2%'",
    "SELECT txn_id FROM transactions WHERE txn_id = 'c2-t0'",
    "SELECT a.account_id FROM accounts a JOIN transactions t USING (account_id, customer_id) "
    "WHERE t.customer_id = 'c2'",
    "SELECT f.fee_id FROM fees f JOIN transactions t USING (txn_id, customer_id) WHERE f.customer_id = 'c2'",
    "SELECT symbol FROM holdings WHERE account_id = 'c2-brk'",
    "SELECT card_id FROM cards WHERE last4 = '4242' AND customer_id <> 'c1'",
])
def test_asking_for_another_customers_rows_returns_nothing(app_engine, sql):
    assert _run(app_engine, as_customer("c1"), sql) == []


def test_aggregates_count_only_the_callers_rows(app_engine):
    rows = _run(app_engine, as_customer("c1"),
                "SELECT count(*), sum(amount_cents) FROM transactions")
    assert rows == [(3, -1250 - 4999 + 250000)]


# --- no context, staff, and the context's lifetime ------------------------------

@pytest.mark.parametrize("access", [None, STAFF, as_customer(None)], ids=["none", "staff", "empty"])
def test_no_customer_context_sees_no_customer_data(app_engine, access):
    for table in OWNED:
        assert _run(app_engine, access, f"SELECT 1 FROM {table}") == [], table


@pytest.mark.parametrize("access", [None, as_customer("c1")], ids=["none", "customer"])
def test_reference_data_is_shared(app_engine, access):
    for table in REFERENCE:
        assert len(_run(app_engine, access, f"SELECT 1 FROM {table}")) == 1, table


def test_the_context_ends_with_its_transaction(app_engine):
    # pool_size=1: every transaction below reuses the same connection.
    assert len(_run(app_engine, as_customer("c1"), "SELECT 1 FROM transactions")) == 3
    assert _run(app_engine, None, "SELECT 1 FROM transactions") == []
    assert {r[0] for r in _run(app_engine, as_customer("c2"),
                               "SELECT customer_id FROM transactions")} == {"c2"}


# --- writes ------------------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "INSERT INTO transactions (txn_id, account_id, customer_id, posted_on, amount_cents, "
    "category, description) VALUES ('x', 'c1-chk', 'c1', '2026-11-01', -1, 'x', 'x')",
    "UPDATE transactions SET amount_cents = 1 WHERE customer_id = 'c1'",
    "DELETE FROM fees",
    "UPDATE customers SET full_name = 'x'",
])
def test_the_app_role_cannot_write_customer_data(app_engine, sql):
    with pytest.raises(ProgrammingError, match="permission denied"):
        _run(app_engine, as_customer("c1"), sql)


@pytest.mark.parametrize("sql", [
    # c1's account, c2 as the owner
    "INSERT INTO transactions (txn_id, account_id, customer_id, posted_on, amount_cents, "
    "category, description) VALUES ('x', 'c1-chk', 'c2', '2026-11-01', -1, 'x', 'x')",
    # c2's transaction under c1's fee
    "INSERT INTO fees (fee_id, account_id, customer_id, txn_id, fee_type, amount_cents, "
    "assessed_on) VALUES ('x', 'c1-chk', 'c1', 'c2-t1', 'overdraft', 3400, '2026-11-01')",
    # c2's card on c1's transaction
    "INSERT INTO transactions (txn_id, account_id, customer_id, card_id, posted_on, amount_cents, "
    "category, description) VALUES ('x', 'c1-chk', 'c1', 'c2-card', '2026-11-01', -1, 'x', 'x')",
    "INSERT INTO holdings VALUES ('c1-brk', 'c2', 'VTI', 1, 100)",
])
def test_a_row_cannot_claim_a_different_owner_than_its_parent(seeded, owner, sql):
    with pytest.raises(IntegrityError), owner.begin() as conn:
        conn.execute(sa.text(sql))


# --- the signed-in user ------------------------------------------------------------

def test_me_reports_the_users_customer(client, seeded, owner):
    with owner.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO users (firebase_uid, org_id, role, clearance, customer_id) "
            "VALUES ('uid-c1', :b, 'viewer', 0, 'c1')"), {"b": BANK})
    body = client.get("/me", headers={"Authorization": "Bearer token:uid-c1"}).json()
    assert (body["set_up"], body["customer_id"]) == (True, "c1")


def test_two_users_cannot_be_the_same_customer(seeded, owner):
    with owner.begin() as conn:
        conn.execute(sa.text("INSERT INTO users (firebase_uid, org_id, role, clearance, customer_id) "
                             "VALUES ('u1', :b, 'viewer', 0, 'c1')"), {"b": BANK})
    with pytest.raises(IntegrityError), owner.begin() as conn:
        conn.execute(sa.text("INSERT INTO users (firebase_uid, org_id, role, clearance, customer_id) "
                             "VALUES ('u2', :b, 'viewer', 0, 'c1')"), {"b": BANK})
