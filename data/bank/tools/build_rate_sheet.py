"""Build rates/rate-sheet.xlsx from the two rate CSVs (the source of truth).

    backend/.venv/Scripts/python data/bank/tools/build_rate_sheet.py

The workbook looks like a published rate sheet, not a data dump: a title
row, an "effective" note, the current rates on the first sheets and the
previous version on a sheet of its own, so a parser has to find the header
row instead of assuming row 1. Timestamps are fixed so a rebuild from the
same CSVs gives the same content.
"""
import csv
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font

RATES = Path(__file__).resolve().parents[1] / "rates"
CURRENT, PREVIOUS = "2026-02-01", "2025-07-01"


def _rows(name):
    with open(RATES / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _sheet(wb, title, heading, rows, columns):
    ws = wb.create_sheet(title)
    ws.append([f"FinPilot Bank: {heading}"])
    ws["A1"].font = Font(bold=True, size=14)
    ws.append(["DRAFT, pending review. Fictional bank; not a real institution."])
    ws.append([])
    ws.append([label for label, _ in columns])
    for cell in ws[4]:
        cell.font = Font(bold=True)
    for row in rows:
        ws.append([_value(row[key]) for _, key in columns])
    ws.append([])
    ws.append(["Rates are variable unless shown as a CD term. APY assumes interest stays on deposit."])


def _value(text):
    try:
        return float(text) if text else None
    except ValueError:
        return text


def main():
    deposits, cards = _rows("deposit-rates.csv"), _rows("card-rates.csv")
    deposit_columns = [("Product", "product"), ("Term (months)", "term_months"),
                       ("Balance from ($)", "balance_min"), ("Balance to ($)", "balance_max"),
                       ("Interest rate (%)", "interest_rate_pct"), ("APY (%)", "apy_pct"),
                       ("Minimum deposit ($)", "minimum_deposit")]
    card_columns = [("Product", "product"), ("Rate type", "rate_type"), ("APR from (%)", "apr_min_pct"),
                    ("APR to (%)", "apr_max_pct"), ("Variable", "variable"),
                    ("Prime rate (%)", "prime_rate_pct")]
    wb = Workbook()
    wb.remove(wb.active)
    _sheet(wb, "Deposit rates", f"Deposit rates effective {CURRENT}",
           [r for r in deposits if r["effective_date"] == CURRENT], deposit_columns)
    _sheet(wb, "Card rates", f"Credit card rates effective {CURRENT}",
           [r for r in cards if r["effective_date"] == CURRENT], card_columns)
    _sheet(wb, "Previous deposit rates", f"Deposit rates effective {PREVIOUS} (superseded {CURRENT})",
           [r for r in deposits if r["effective_date"] == PREVIOUS], deposit_columns)
    wb.properties.creator = "FinPilot Bank Treasury"
    wb.properties.created = wb.properties.modified = datetime(2026, 2, 1)
    wb.save(RATES / "rate-sheet.xlsx")


if __name__ == "__main__":
    main()
