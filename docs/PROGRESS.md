# FinPilot progress log

The plan, the architecture checklist and the commit budget are in
[FINPILOT_PLAN.md](FINPILOT_PLAN.md). DocuMind's own log (Phases 0–2 of the
enterprise build, its eval baselines and decisions) is in git history: see
`docs/PROGRESS.md` at commit 51ba57a.

**Commits: 6 of 22** (budget pending your choice of 20 or 22; see the plan).

## Chores outside the phase budget

- 2026-11-13: GitHub repo renamed DocuMind → FinPilot (old URL redirects);
  `docker-compose.yml` pins `name: documind` so the existing volumes
  (`documind_postgres_data`, `documind_hf_cache`) survive renaming the
  local folder to FinPilot.

## Phase 0: cleanup and rename (done)

| # | Commit | What |
|---|---|---|
| 1 | `0fccfd7` | Remove upload and the access-change API; rename DocuMind to FinPilot; migration 0008 moves the app role to `finpilot_app` |
| spare 1 | `00fbfda` | Integration tests loaded documents through `/upload`; they now use the store (CI caught it) |
| 2 | `59f9659` | Plan, embedding pacing from `phase3-corpus`, `docs/baseline/` deleted, fresh progress log, README |

### Notes

- **Role rename without `ALTER ROLE ... RENAME`.** Roles are cluster-wide and
  every test database migrates on one server, so 0008 creates `finpilot_app`
  if missing, moves the grants to it and strips them from `documind_app` in
  that database. A test checks both directions.
- **`@documind.test` stays** in the eval-account guard: the real Firebase
  eval account and the Firebase project (`documind-29040`) use that name.
  New test accounts come with the Phase 1 users.
- **Local `.env` and dev volume untouched.** `.env` names the DB user and
  database `documind` explicitly, which still works; compose now builds
  `finpilot_app` URLs, and the migrate step runs 0008 on the existing volume.
- **From `phase3-corpus`**, only the generic parts came over: rate limiting
  by texts and tokens, paced embedding, a clean stop on the daily quota
  (`backend/ingestion/pacing.py`), and per-model embedding text counts. The
  `kb/` corpus, its manifest loader and test users stay on that branch.
- **Testing rule from here on** (your instruction, 2026-11-13): no Docker,
  no local Postgres, no Gemini calls on this machine. Light tests locally
  (backend tests without Postgres, frontend lint and build); GitHub CI runs
  the full Postgres suite on every push.

## Phase 1: customer data (done; PR #1 waits for you)

| # | Commit | What |
|---|---|---|
| 1 | `223f0ae` | Migration 0009 (customers, accounts, cards, transactions, fees, holdings, securities, prices), per-customer RLS via `app.customer_id`, `users.customer_id`, leakage tests, a Neon-style owner test |
| 2 | `acb6ae4` | Seeded synthetic generator and its loader (`backend/bank/synthetic.py`); the schema test learns the new tables (CI failure on commit 1) |

### Notes

- **Ownership can't diverge.** Child rows repeat `customer_id` and reference
  `(account_id, customer_id)` (and `(card_id, customer_id)`,
  `(txn_id, customer_id)`), so a transaction can't sit on another
  customer's account or card.
- **Staff see no customer data.** Their `customer_id` is empty, which no row
  matches. A support view, if ever needed, is a separate, audited path.
- **Neon owner.** CI runs every migration as a role with `CREATEROLE` and
  `CREATEDB` but neither superuser nor `BYPASSRLS` (stricter than Neon's
  owner), and checks that FORCE RLS applies to it: customer rows go in only
  inside that customer's context. The loader (next commit) writes that way.
- **The generator** (seed 7, 50 customers): 12 months from 2025-07-01,
  about 24,000 transactions, 790 fees, 19 holdings, in under a second. The
  fee schedule changes on 2026-02-01 (overdraft $34 → $15, NSF fee
  dropped, a $50 overdraft cushion); every fee matches the schedule in force
  on its date, and the same numbers go into the bank's fee-schedule
  documents. Each customer has one canary transaction.
- **The loader** writes each customer in one transaction bound to that
  customer, deleting with an explicit `customer_id` filter (an owner with
  BYPASSRLS isn't limited by the policy) and upserting the customer row.
  Tested as the superuser and as an owner without BYPASSRLS.
- **Not run locally** (no Postgres here); CI runs the Postgres tests.

## Phase 2: bank corpus and sources (drafts, pending your review)

| # | Commit | What |
|---|---|---|
| 1 | this commit | `data/bank/`: fee schedules 2025 and 2026, overdraft policies 2025 and 2026, dispute policy, account comparison, deposit agreement, rate tables (CSV and XLSX), the bank's OpenAPI spec, a staff-only fraud-rules memo with a canary; `manifest.yaml`; `data/SOURCES.md` |

### Notes

- **All drafts.** Every document says "DRAFT, pending review" and has
  `status: draft` in the manifest. Written for FinPilot; the structure
  follows the kinds of disclosure the CFPB publishes model forms for, with
  no text copied.
- **Numbers agree with the generator.** A unit test checks every fee row in
  both schedules against `FEE_SCHEDULES`; APYs and rates agree under daily
  compounding; card APRs equal prime plus the stated margins.
- **The canary** (`internal/fraud-rules-memo.md`, clearance 2) appears in no
  other file; a test checks the repo.
- **XLSX** is built from the CSVs by `data/bank/tools/build_rate_sheet.py`
  (openpyxl, installed in the local venv only; Phase 3 adds it to the
  requirements when it parses spreadsheets). The workbook has title rows
  above its headers on purpose.
- **Licences** (`data/SOURCES.md`): TabFormer (Apache-2.0 repo, no
  data-specific terms) and Berka (no licence found) are not used; Alpha
  Vantage's free key is personal and non-commercial only; FRED needs its
  notice and excludes copyrighted series; SEC needs a User-Agent with a
  contact, max 10 requests/second; CFPB content is public domain; FDIC's own
  pages make no public-domain statement (BankFind is listed as public
  domain); investor.gov's policies page couldn't be read (403).
