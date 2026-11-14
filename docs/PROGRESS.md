# FinPilot progress log

The plan, the architecture checklist and the commit budget are in
[FINPILOT_PLAN.md](FINPILOT_PLAN.md). DocuMind's own log (Phases 0–2 of the
enterprise build, its eval baselines and decisions) is in git history: see
`docs/PROGRESS.md` at commit 51ba57a.

**Commits: 8 of 22** (budget decided 2026-11-13; the Compose-name chore is outside it).

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

## Phase 1: customer data (done; PR #5 waits for you)

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

## Phase 2: bank corpus and sources (approved as drafts, 2026-11-13)

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
- **Your decisions (2026-11-13)** are recorded in the plan and in
  `data/SOURCES.md`: Investor.gov reused like other SEC content, with
  attribution; real prices only for signed-in demo accounts; no TabFormer.

## Phase 3: parsing, restructuring, chunking

PR #5 (`finpilot` → `main`, GitHub's number: #1–4 are DocuMind's) holds
Phases 0–2 for your review. Phase 3 is on `finpilot-phase3`, stacked on it,
so the PR doesn't change under review.

| # | Commit | What |
|---|---|---|
| 1 | `4137f21` | Parsers for Markdown, CSV, XLSX, OpenAPI and PDF/DOCX/TXT; noise removal; structure-aware chunker (256–512 tokens); the corpus loader; full-length deposit agreement and a new cardmember agreement |
| 2 | this commit | Migration 0010: document tables as rows linked to their chunks; synthetic receipt images and one vision call each, checked against the transaction and cached; image input in the LLM gateway; recurring subscriptions and occasional (not chronic) overdrafts in the generator |

### Notes

- **Corpus changes ride in this commit**, not a Phase 2 commit, to keep the
  spare: they exist so chunking can be tested. The deposit agreement is now
  5,391 words in 24 numbered sections with four tables (opening deposits,
  funds availability, daily limits, CD penalties). Its worked examples
  ($29.14 interest, $100.36 penalty) were recomputed. The fee schedule and
  account comparison both cited a "Cardmember Agreement" that didn't exist;
  it does now (2026, card services: pricing table, APR = prime + margin
  matching `card-rates.csv`, average daily balance interest, payments,
  rewards). Both fee schedules gained the balance-transfer fee row that
  agreement states.
- **Multi-chunk check.** 13 documents, 62 chunks. Six documents span 3 or
  more chunks (deposit agreement 19, OpenAPI 13, cardmember agreement 8,
  both fee schedules and the dispute policy 3); a test keeps it at six or
  more. The short policies stay at 2 chunks: they're the version-pair
  distractors (2025 vs 2026), not chunking tests.
- **Blocks, then chunks.** Parsers turn each file into blocks a chunk may
  not cut: a paragraph, a list item, a whole table, one API operation.
  The chunker packs blocks of the same section up to 512 tokens and lets a
  chunk under 256 take in the next section; every chunk starts with its
  section path. A table over the limit splits between rows, repeating its
  header; an operation is always its own chunk.
- **Spreadsheets.** CSVs split by `effective_date`, so the 2025 and 2026
  rates never share a chunk. In the workbook the header is the widest row,
  found under the title rows on purpose.
- **Noise.** Running headers and footers (on half the pages or more, at
  the page edges), page numbers at the edges only (a lone "12" mid-page can
  be a table cell), navigation lines, and paragraphs repeated in 3+
  documents. The draft banner is boilerplate; one copy stays (in the 2025
  fee schedule) so "not FDIC-insured" is still retrievable. Removed share
  per document: 0–6%.
- **Stable ids.** `doc_id` is a hash of the file path, so chunk ids are the
  same on every run and can key the Phase 3 metadata cache.
- **`chunk_pages` stays** for DocuMind's PDF eval tooling. FinPilot's
  corpus uses `chunk_blocks`.
- **Venv rebuilt** after the folder rename (pinned requirements, no torch,
  eval extras included): 234 passed, 163 Postgres tests skipped locally.

### Notes (commit 2)

- **Tables as rows.** `ingestion/tables.py` turns every table block into a
  `doc_tables` row (columns, section) and one `doc_table_rows` row per table
  row, with the `chunk_id` that holds it; a table split across chunks stays
  one table. Cells stay text (`"$15.00 per item"`); the calculator parses
  what it needs and cites the row. They are written in `add_document`'s
  transaction, copying org, team and clearance from the documents row in
  SQL, under the same `tenant_access` policy as chunks, so the staff memo's
  table is as hidden as its text. The corpus has 29 tables, 161 rows.
- **Receipts.** Ten PNGs in `data/bank/receipts/`, one card purchase each
  for customers 1–10, rendered from the transaction itself (merchant, date,
  total, card last 4; the items are invented but add up to the total). The
  vision call returns those four fields; `check` compares each exactly
  (merchant ignoring case and accents, total in cents, last 4 digits).
  Results go in `receipt_images` under the **customer** policy, written in
  the receipt's customer context (works for a Neon owner without
  BYPASSRLS), keyed by the image's hash and the model, so a re-run makes
  no calls. `txn_id` isn't a foreign key: reloading the synthetic data
  replaces transactions, and the cached extraction must survive that.
- **Not run yet: the real vision pass** (10 Gemini calls, about one minute
  under the 15-a-minute limit). It needs a database with the customer data
  and your go: `python -m bank.receipts extract --database-url OWNER`.
- **Generator realism** (your instruction: salary, rent, subscriptions,
  groceries, occasional overdrafts). Subscriptions were random one-off
  purchases; each customer now has 1–3 that charge the same amount on the
  same day every month. Overdrafts were chronic (31 of 50 customers, one at
  80 fees a year): customers without direct deposit paid rent sized to a
  paycheck they never received. Now everyone is paid every other Friday
  (direct deposit or mobile check), card spending is 75% of what pay
  leaves after rent (60% premier, 80% students), and a month-end sweep
  keeps checking near rent plus a per-customer cushion. Result, seed 7:
  23,115 transactions, 625 fees; 31 customers overdraw, **median 3 times a
  year** (one struggling customer at 38), 121 overdraft fees under the 2025
  schedule and 83 under 2026. Tests hold subscriptions to monthly fixed
  amounts and overdrafts to "occasional, under both schedules".
