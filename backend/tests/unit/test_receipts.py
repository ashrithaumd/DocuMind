"""Synthetic receipts (bank/receipts.py): ground truth comes from the
generator's own rows, the committed images match it, and extracted fields
are checked exactly."""
import io

import pytest
from PIL import Image

from bank.receipts import RECEIPTS, check, extract, pick, render
from bank.synthetic import generate
from tests.helpers import FakeLLM


@pytest.fixture(scope="module")
def truths():
    return pick(generate())


def test_each_receipt_is_a_real_card_purchase(truths):
    data = generate()
    txns = {t["txn_id"]: t for t in data["transactions"]}
    last4 = {c["card_id"]: c["last4"] for c in data["cards"]}
    assert len(truths) == 10 and len({t["customer_id"] for t in truths}) == 10
    for truth in truths:
        t = txns[truth["txn_id"]]
        assert (t["merchant"], t["posted_on"].isoformat(), -t["amount_cents"], last4[t["card_id"]]) == \
               (truth["merchant"], truth["date"], truth["amount_cents"], truth["card_last4"])


def test_the_committed_images_are_the_picked_receipts(truths):
    assert sorted(p.name for p in RECEIPTS.glob("*.png")) == sorted(f"{t['txn_id']}.png" for t in truths)
    for truth in truths:
        committed = Image.open(RECEIPTS / f"{truth['txn_id']}.png")
        fresh = Image.open(io.BytesIO(render(truth)))
        assert (committed.format, committed.size) == ("PNG", fresh.size)


def test_rendering_is_deterministic(truths):
    assert render(truths[0]) == render(truths[0])


def test_check_accepts_the_same_values_written_differently():
    truth = {"merchant": "Café Lumière Paris", "date": "2026-03-04", "amount_cents": 3976, "card_last4": "0612"}
    fields = {"merchant": "CAFE LUMIERE PARIS", "date": "2026-03-04", "total": "$39.76",
              "card_last4": "**** 0612"}
    assert check(truth, fields) == dict.fromkeys(["merchant", "date", "amount_cents", "card_last4"], True)


def test_check_flags_each_wrong_field():
    truth = {"merchant": "Corner Grocer", "date": "2026-03-04", "amount_cents": 3976, "card_last4": "0612"}
    fields = {"merchant": "Corner Grocery", "date": "2026-04-03", "total": "39.67", "card_last4": "0621"}
    assert check(truth, fields) == dict.fromkeys(["merchant", "date", "amount_cents", "card_last4"], False)
    assert not any(check(truth, {}).values())  # a refused or malformed reply matches nothing


async def test_one_vision_call_per_receipt_with_the_image():
    llm = FakeLLM()
    llm.receipt = {"merchant": "X", "date": "", "total": "", "card_last4": ""}
    png = b"\x89PNG fake"
    assert await extract(llm, "vision-x", png) == llm.receipt
    assert len(llm.chat_calls) == 1
    assert llm.chat_calls[0]["image"] == png and llm.chat_calls[0]["role"] == "receipt"
