"""Synthetic receipt images, and the fields one vision call reads from each.

    cd backend
    python -m bank.receipts render                        # writes data/bank/receipts/*.png
    python -m bank.receipts extract --database-url OWNER  # one vision call per new image

Each receipt is drawn from a real row of the synthetic data (seed 7, 50
customers): one card purchase for each of the first RECEIPTS customers.
The transaction is the ground truth, so every extracted field (merchant,
date, total, card last 4) is checked exactly, not judged.

Extraction is cached in receipt_images (migration 0010): an image whose
bytes and model already have fields is not sent again. Rows are written as
the owner inside the receipt's customer context, like the customer data,
and are visible only to that customer.
"""
import argparse
import asyncio
import hashlib
import io
import os
import random
import re
import sys
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path

from bank.synthetic import generate

RECEIPTS = Path(__file__).resolve().parents[2] / "data" / "bank" / "receipts"
COUNT = 10

ITEMS = {
    "groceries": ["Bananas", "Whole milk 1 gal", "Sourdough loaf", "Eggs, dozen", "Coffee beans",
                  "Spinach", "Pasta 1 lb", "Cheddar"],
    "dining": ["Latte", "Noodle bowl", "Green tea", "Dumplings", "Pastry", "Soup"],
    "transport": ["Fuel", "Car wash", "Transit pass"],
    "health": ["Pain relief", "Vitamins", "Bandages", "Prescription copay"],
    "home": ["Light bulbs", "Paint, 1 qt", "Screws", "Extension cord"],
    "shopping": ["Paperback", "Notebook", "Socks", "Rain jacket"],
    "travel": ["Breakfast", "Rail pass", "Espresso"],
}

SYSTEM = (
    "You are a receipt reader. From the receipt image, return the merchant name as printed, "
    "the purchase date as YYYY-MM-DD, the total charged as a decimal number (e.g. 23.40), and "
    "the last four digits of the card. Use an empty string for anything you can't read. Report "
    "only what the image shows."
)
SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "string"} for k in ("merchant", "date", "total", "card_last4")},
    "required": ["merchant", "date", "total", "card_last4"],
}


def pick(data: dict[str, list[dict]], count: int = COUNT) -> list[dict]:
    """Ground truth for `count` receipts: one card purchase per customer, in
    customer order, chosen by a generator seeded with the customer id."""
    last4 = {c["card_id"]: c["last4"] for c in data["cards"]}
    purchases: dict[str, list[dict]] = {}
    for t in data["transactions"]:
        if t["card_id"] and t["description"].startswith("Card purchase"):
            purchases.setdefault(t["customer_id"], []).append(t)
    out = []
    for customer in data["customers"]:
        rows = purchases.get(customer["customer_id"])
        if rows and len(out) < count:
            t = random.Random(customer["customer_id"]).choice(rows)
            out.append({"txn_id": t["txn_id"], "customer_id": t["customer_id"],
                        "merchant": t["merchant"], "date": t["posted_on"].isoformat(),
                        "amount_cents": -t["amount_cents"], "card_last4": last4[t["card_id"]],
                        "category": t["category"]})
    return out


def _money(cents: int) -> str:
    return f"${cents / 100:,.2f}"


def render(truth: dict) -> bytes:
    """A receipt as PNG bytes. Items are invented but add up to the total."""
    from PIL import Image, ImageDraw, ImageFont

    rng = random.Random(truth["txn_id"])
    pool, total = ITEMS.get(truth["category"], ["Item"]), truth["amount_cents"]
    names = rng.sample(pool, min(3, len(pool), total))  # every item costs at least a cent
    cuts = sorted(rng.sample(range(1, total), len(names) - 1))
    prices = [b - a for a, b in zip([0, *cuts], [*cuts, total])]
    year, month, day = truth["date"].split("-")

    font, big = ImageFont.load_default(size=18), ImageFont.load_default(size=26)
    rule = ("-" * 40, "", font)
    lines = [(truth["merchant"].upper(), "", big), (f"{rng.randint(10, 999)} Market Street", "", font),
             ("", "", font),
             (f"Date {month}/{day}/{year}", f"Time {rng.randint(8, 20):02d}:{rng.randint(0, 59):02d}", font),
             rule, *[(n, _money(p), font) for n, p in zip(names, prices)], rule,
             ("TOTAL", _money(total), big), ("", "", font),
             (f"CARD ************{truth['card_last4']}", "", font), ("APPROVED", "", font), ("", "", font),
             ("Synthetic receipt: FinPilot test data", "", font)]
    width = 460
    image = Image.new("L", (width, 40 + 34 * len(lines)), 255)
    draw = ImageDraw.Draw(image)
    for i, (left, right, face) in enumerate(lines):
        y = 20 + 34 * i
        draw.text((24, y), left, fill=0, font=face)
        draw.text((width - 24 - draw.textlength(right, font=face), y), right, fill=0, font=face)
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _plain(text) -> str:
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^\w&]+", " ", text).split()).casefold()


def _cents(total) -> int | None:
    try:
        return int((Decimal(re.sub(r"[^\d.\-]", "", str(total or ""))) * 100).to_integral_value())
    except InvalidOperation:
        return None


def check(truth: dict, fields: dict) -> dict[str, bool]:
    """Each extracted field against the transaction."""
    return {
        "merchant": _plain(fields.get("merchant")) == _plain(truth["merchant"]),
        "date": str(fields.get("date", "")).strip() == truth["date"],
        "amount_cents": _cents(fields.get("total")) == truth["amount_cents"],
        "card_last4": re.sub(r"\D", "", str(fields.get("card_last4", "")))[-4:] == truth["card_last4"],
    }


async def extract(llm, model: str, png: bytes) -> dict:
    """One vision call. A refusal or malformed reply comes back as {}."""
    return await llm.complete_json(model, SYSTEM, "Read this receipt.", SCHEMA,
                                   max_tokens=512, role="receipt", image=png)


async def extract_all(owner_url: str, llm, model: str, truths: list[dict],
                      folder: Path = RECEIPTS) -> list[dict]:
    """Extract and store every receipt not already done with `model`.
    Returns one result per receipt: truth, fields, matches, cached."""
    import json

    import sqlalchemy as sa

    engine = sa.create_engine(owner_url)
    results = []
    try:
        for truth in truths:
            path = folder / f"{truth['txn_id']}.png"
            png = path.read_bytes()
            image_id = hashlib.sha256(png).hexdigest()[:16]
            with engine.begin() as conn:
                conn.execute(sa.text("SELECT set_config('app.customer_id', :c, true)"),
                             {"c": truth["customer_id"]})
                done = conn.execute(sa.text(
                    "SELECT fields FROM receipt_images WHERE image_id = :i AND model = :m "
                    "AND fields IS NOT NULL"), {"i": image_id, "m": model}).scalar()
            if done is not None:
                results.append({"truth": truth, "fields": done, "matches": check(truth, done), "cached": True})
                continue
            fields = await extract(llm, model, png)
            matches = check(truth, fields)
            with engine.begin() as conn:
                conn.execute(sa.text("SELECT set_config('app.customer_id', :c, true)"),
                             {"c": truth["customer_id"]})
                conn.execute(sa.text("""
                    INSERT INTO receipt_images (image_id, customer_id, txn_id, path, model, fields,
                                                matches, extracted_at)
                    VALUES (:i, :c, :t, :p, :m, CAST(:f AS jsonb), CAST(:x AS jsonb), now())
                    ON CONFLICT (image_id) DO UPDATE SET model = EXCLUDED.model,
                        fields = EXCLUDED.fields, matches = EXCLUDED.matches,
                        extracted_at = EXCLUDED.extracted_at
                """), {"i": image_id, "c": truth["customer_id"], "t": truth["txn_id"],
                       "p": f"data/bank/receipts/{path.name}", "m": model,
                       "f": json.dumps(fields), "x": json.dumps(matches)})
            results.append({"truth": truth, "fields": fields, "matches": matches, "cached": False})
    finally:
        engine.dispose()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Render synthetic receipts or extract their fields.")
    parser.add_argument("command", choices=["render", "extract"])
    parser.add_argument("--database-url", default=os.environ.get("MIGRATION_DATABASE_URL"),
                        help="Owner Postgres URL, for extract (default: MIGRATION_DATABASE_URL)")
    parser.add_argument("--model", help="Vision model (default: the grading model)")
    args = parser.parse_args()
    truths = pick(generate())
    if args.command == "render":
        RECEIPTS.mkdir(exist_ok=True)
        for truth in truths:
            (RECEIPTS / f"{truth['txn_id']}.png").write_bytes(render(truth))
        print(f"Wrote {len(truths)} receipts to {RECEIPTS}")
        return 0
    if not args.database_url:
        print("Pass --database-url (the owner) or set MIGRATION_DATABASE_URL.", file=sys.stderr)
        return 2
    from config import get_settings
    from llm import build_llm

    settings = get_settings()
    model = args.model or settings.grading_model
    results = asyncio.run(extract_all(args.database_url, build_llm(settings), model, truths))
    for field in ("merchant", "date", "amount_cents", "card_last4"):
        right = sum(r["matches"][field] for r in results)
        print(f"{field:13} {right}/{len(results)}")
    print(f"vision calls: {sum(not r['cached'] for r in results)} ({model})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
