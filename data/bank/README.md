# FinPilot Bank corpus

> **DRAFT, pending review.** Nothing here is final until reviewed and the
> `status` in `manifest.yaml` changes from `draft` to `approved`.

FinPilot Bank is a **fictional bank** made up for this software project. It
is not a real institution, offers no real products, and is not FDIC-insured.
These documents were written for FinPilot. Their structure follows the kinds
of disclosure the CFPB publishes model forms and guidance for (fee schedules,
account comparisons, error-resolution notices), but no text was copied from
any CFPB, bank or other source.

## What's here

| File | Type | Audience | Notes |
|---|---|---|---|
| `fee-schedule-2025.md` | fee schedule, v2025 | customers | In force 2025-01-01 to 2026-01-31 |
| `fee-schedule-2026.md` | fee schedule, v2026 | customers | In force from 2026-02-01; lower overdraft fee, no NSF fee |
| `overdraft-policy-2025.md` | policy, v2025 | customers | Pairs with the 2025 fee schedule |
| `overdraft-policy-2026.md` | policy, v2026 | customers | Adds the $50 cushion and the 2-per-day cap |
| `dispute-policy.md` | policy | customers | Card and electronic-transfer disputes, timelines |
| `account-comparison.md` | product comparison | customers | Checking, savings, card, investing side by side |
| `deposit-agreement.md` | agreement | customers | The terms behind every deposit account (about 5,400 words, 24 sections) |
| `cardmember-agreement.md` | agreement | customers | Cash Back Card terms: pricing table, how APRs follow prime, interest, payments, rewards |
| `rates/deposit-rates.csv`, `rates/card-rates.csv` | rate tables | customers | Two versions each (2025-07-01, 2026-02-01) |
| `rates/rate-sheet.xlsx` | spreadsheet | customers | The same tables as a published workbook, built from the CSVs |
| `api/openapi.yaml` | code (OpenAPI 3.1) | developers | The bank's customer-data API, which FinPilot's tools mirror |
| `internal/fraud-rules-memo.md` | memo | **staff only (clearance 2)** | Contains a canary string no customer may ever retrieve |

`manifest.yaml` carries each document's version, effective date, department,
document type, clearance and review status: the metadata that Phase 3 attaches
to every chunk.

## Consistency rules

- Fee amounts match `backend/bank/synthetic.py` (`FEE_SCHEDULES`), so the
  synthetic customers are charged exactly what these documents say, and an
  answer about a customer's fee can cite the schedule in force on that date.
- Every amount appears once per version. If a number changes, it changes in
  the code and the documents in the same commit.
- The canary string in the staff memo appears nowhere else in the repo.
