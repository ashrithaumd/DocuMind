"""The synthetic customer data (bank/synthetic.py): deterministic, internally
consistent, and charging exactly what the fee schedule in force says."""
from collections import Counter

import pytest

from bank.synthetic import FEE_SCHEDULES, SCHEDULE_CHANGE, canary, generate, schedule_on


@pytest.fixture(scope="module")
def data():
    return generate(seed=7, customers=20)


def test_the_same_seed_gives_the_same_rows():
    assert generate(seed=3, customers=4) == generate(seed=3, customers=4)
    assert generate(seed=3, customers=4) != generate(seed=4, customers=4)


def test_every_child_row_belongs_to_its_accounts_customer(data):
    owner = {a["account_id"]: a["customer_id"] for a in data["accounts"]}
    card_owner = {c["card_id"]: c["customer_id"] for c in data["cards"]}
    txn_owner = {t["txn_id"]: t["customer_id"] for t in data["transactions"]}
    for table in ("cards", "transactions", "fees", "holdings"):
        for row in data[table]:
            assert owner[row["account_id"]] == row["customer_id"], (table, row)
    for t in data["transactions"]:
        assert t["card_id"] is None or card_owner[t["card_id"]] == t["customer_id"]
    for f in data["fees"]:
        assert f["txn_id"] is None or txn_owner[f["txn_id"]] == f["customer_id"]


def test_ids_are_unique(data):
    for table, key in (("customers", "customer_id"), ("accounts", "account_id"),
                       ("cards", "card_id"), ("transactions", "txn_id"), ("fees", "fee_id")):
        ids = [r[key] for r in data[table]]
        assert len(ids) == len(set(ids)), table


def test_holdings_are_only_in_brokerage_accounts(data):
    kind = {a["account_id"]: a["type"] for a in data["accounts"]}
    assert data["holdings"]
    assert {kind[h["account_id"]] for h in data["holdings"]} == {"brokerage"}


def test_money_is_whole_cents_and_never_zero(data):
    assert all(isinstance(t["amount_cents"], int) and t["amount_cents"] != 0
               for t in data["transactions"])
    assert all(isinstance(f["amount_cents"], int) and f["amount_cents"] > 0 for f in data["fees"])


def test_fees_follow_the_schedule_in_force_on_their_date(data):
    by_type = {"overdraft": "overdraft", "atm": "atm_out_of_network",
               "monthly_maintenance": "maintenance", "late_payment": "late_payment"}
    for f in data["fees"]:
        if f["fee_type"] in by_type:
            expected = getattr(schedule_on(f["assessed_on"]), by_type[f["fee_type"]])
            assert f["amount_cents"] == expected, f
    assert {schedule_on(f["assessed_on"]) for f in data["fees"]} == set(FEE_SCHEDULES.values())


def test_overdraft_fees_respect_the_daily_cap(data):
    per_day = Counter((f["customer_id"], f["assessed_on"]) for f in data["fees"]
                      if f["fee_type"] == "overdraft")
    for (_, day), n in per_day.items():
        assert n <= schedule_on(day).overdraft_max_per_day


def test_the_2026_schedule_charges_less_for_an_overdraft():
    before, after = schedule_on(SCHEDULE_CHANGE.replace(day=1, month=1)), schedule_on(SCHEDULE_CHANGE)
    assert after.overdraft < before.overdraft and after.nsf == 0


def test_every_customer_has_exactly_one_canary(data):
    for c in data["customers"]:
        mine = [t for t in data["transactions"] if canary(c["customer_id"]) in t["description"]]
        assert len(mine) == 1 and mine[0]["customer_id"] == c["customer_id"]
