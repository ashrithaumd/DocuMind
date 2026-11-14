# FinPilot plan

FinPilot is a banking assistant for the customers of a fictional bank. It
answers from three kinds of source: the customer's own data (accounts,
cards, transactions, fees, holdings), the bank's documents (fee schedules,
policies, agreements, rate tables, its API spec), and public market and
regulatory data (prices, FRED series, SEC filings, CFPB and Investor.gov
guidance). Every answer is planned, run through tools, and validated before
the user sees it, and every step is traced.

It is built on the DocuMind codebase (hybrid pgvector + full-text retrieval,
Firebase auth, row-level security, the LLM gateway and the eval harness).
DocuMind's own history is in git, up to commit 51ba57a on `main`.

## Decisions

| Topic | Decision |
|---|---|
| LLM budget | Specialist agents are tool steps (plain code), not separate LLM calls. The three validators are **one** structured LLM call returning the gatekeeper, auditor and strategist verdicts. Target: **3–4 LLM calls per question** (planner, answer, validators, plus one retry at most). |
| Eval size | About **40 questions**. Chunk metadata and fetched data are cached, keyed by content hash, so re-runs cost nothing for unchanged input. |
| SEC filings | Only Items 1A (risk factors) and 7 (MD&A), for 3–4 companies, plus 2–3 ETF prospectuses. |
| Customer data | Ownership RLS: every customer-owned row carries `customer_id`; one policy compares it with `app.customer_id`, set per transaction like `app.org_id`. New migrations only; 0001–0008 stay as they are. |
| Bank documents | The org/clearance machinery stays, reduced: one org (the bank), **clearance 0** for customer-facing documents, **clearance 2** for staff-only ones. A staff-only fraud-rules memo carries a canary string that no customer may ever retrieve. Teams are unused. |
| Upload | Removed. Documents come only from the loader. |
| Customer data source | A seeded synthetic generator with realistic patterns (salary, rent, subscriptions, groceries, occasional overdrafts). Berka and TabFormer are not used (see `data/SOURCES.md`). |
| Market prices | Real prices (Alpha Vantage), shown only to signed-in demo accounts, cached in Postgres, never committed, with attribution and an as-of date. FINRA content is linked, never ingested. |
| Hosting | Neon (serverless Postgres + pgvector) from Phase 4. Until then plain Postgres in CI. The code runs on both (see Neon notes). |

## Architecture checklist

Every box of the target architecture, with the phase that builds it. A box
is ticked only when its code is merged and tested.

### 1. Data sources (all four types)

- [ ] **Documents**: the bank's policies, agreements, fee schedules (Markdown/PDF), CFPB and Investor.gov guidance, SEC 10-K Items 1A and 7, ETF prospectuses. *Phase 2 (corpus), Phase 3 (ingestion).*
- [ ] **Spreadsheets**: the bank's rate tables (CSV and XLSX), fee tables. *Phase 2, Phase 3.*
- [ ] **Images**: synthetic receipts and statement snippets, rendered from the generator's own transactions so the extracted fields can be checked exactly. *Phase 3.*
- [ ] **Code**: the bank's OpenAPI spec (and its example requests), chunked by path and operation. *Phase 2, Phase 3.*

### 2. Restructuring

- [ ] **Noise removal**: headers, footers, page numbers, repeated legal boilerplate and navigation text stripped before chunking; the removed share is reported per document. *Phase 3.*
- [ ] **Unstructured → structured: tables**: tables in documents (fee schedules, rate tables) become relational rows (`doc_tables`, `doc_table_rows`), each linked to the chunk it came from, so a calculation can cite its source. *Phase 3.*
- [ ] **Unstructured → structured: images**: receipts and statement images become fields (merchant, date, amount, last-4) through **one Gemini vision call per image**, validated against the generator's ground truth and stored with a link to the image. *Phase 3.*

### 3. Chunking

- [ ] **Structure-aware chunking at 256–512 tokens**: split on headings, sections, list items, table boundaries and OpenAPI operations; never mid-table; small neighbours merged up to 256, large sections split below 512. *Phase 3.*

### 4. Chunk metadata

- [ ] **Summary, keywords, hypothetical questions**: one LLM call per section (several chunks per call), cached by content hash. *Phase 3.*
- [ ] **Version, effective date, department, document type**: from the corpus manifest, not the LLM. *Phase 3.*

### 5. Storage: Neon

- [ ] **Serverless Postgres + pgvector** on Neon. *Phase 4.*
- [ ] **Metadata filters** in SQL (document type, department, version, effective date "as of" a date). *Phase 4.*
- [ ] **Parent-document joins**: retrieve small chunks, return their parent section. *Phase 4.*
- [ ] **One branching experiment**: a Neon branch holds a re-chunked or re-embedded index; the eval runs on both branches and the result is written up. *Phase 4.*

### 6. Retrieval

- [ ] **Hybrid search** (dense + full-text, RRF): exists, extended with the hypothetical-question index and the filters. *Phase 4.*
- [ ] **Rerank**, behind a toggle, measured on and off: a local cross-encoder run through ONNX (fastembed's reranker), so torch isn't loaded. *Phase 4.*

### 7. Reasoning engine

- [ ] **Planner**: one LLM call turns the question into a typed plan (which tools, which arguments). *Phase 6.*
- [ ] **Tool execution**: plain code runs the plan's steps, under the caller's RLS context. *Phase 5 (tools), Phase 6 (executor).*
- [ ] **Conditional router**: code, not an LLM, routes small talk, refusals, single-tool and multi-tool plans. *Phase 6.*

### 8. Agents (real, each a tool-backed step)

- [ ] **Financial-data agent**: the customer's accounts, cards, transactions, fees, holdings (RLS-scoped), and market data (prices, FRED, SEC). *Phase 5.*
- [ ] **Documents agent**: retrieval over the bank's documents and public guidance, with citations. *Phase 4–5.*
- [ ] **Summarization agent**: condenses retrieved sections and tool results into the answer; this is the answer LLM call. *Phase 6.*
- [ ] **Calculation agent**: deterministic arithmetic (fees, interest, balances, returns) in code, never in the LLM. *Phase 5.*

### 9. Validation (one structured call)

- [ ] **Gatekeeper**: is the question in scope, and does the answer reveal anything the caller may not see? *Phase 7.*
- [ ] **Auditor**: is every claim supported by a cited chunk or tool result? *Phase 7.*
- [ ] **Strategist**: is the answer complete and useful; if not, one targeted retry. *Phase 7.*

### 10. Observability

- [ ] **Tracing**: every step (plan, tool, LLM call, validator verdict) with timings. *Phase 8.*
- [ ] **Cost**: tokens and calls per question, per model. *Phase 8.*

### 11. Evaluation

- [ ] **LLM judges**: faithfulness, relevance, depth (one judge call per question returning all three). *Phase 10.*
- [ ] **Retrieval precision and recall** on labelled chunks. *Phase 10.*
- [ ] **Latency, tokens, cost** per question and per run. *Phase 10.*
- [ ] **Feedback loop**: thumbs and comments feed a review queue that becomes new eval items. *Phase 10* (the feedback table exists).

### 12. Red-teaming

- [ ] **Reasoning engine**: prompt injection in questions and in documents, plan hijacking, scope escape, validator evasion. *Phase 11.*
- [ ] **Agents**: cross-customer access through tool arguments, staff-only canary extraction, calculation tampering, SQL-shaped inputs. *Phase 11.*

## Phases and commit budget

| Phase | Commits | Contents |
|---|---|---|
| 0 | 2 | Remove upload, rename; plan, pacing, cleanup |
| 1 | 2 | Banking schema + customer RLS + leakage tests; seeded generator + loader. **PR #1** |
| 2 | 1 | Bank corpus (documents, spreadsheets, OpenAPI), public sources, `data/SOURCES.md` |
| 3 | 3 | Parsers for four source types + noise removal + structure-aware chunker; unstructured → structured (tables, vision); chunk metadata with cache |
| 4 | 2 | Neon: filters, parent joins, question index, version filter; rerank toggle + branching experiment |
| 5 | 2 | Accounts tool + adversarial tests; market tools + calculator. **PR #2** |
| 6 | 1 | Planner, executor, router, agents |
| 7 | 1 | Validators (one call) |
| 8 | 1 | Tracing and cost |
| 9 | 2 | Chat UI; "how I got this" panel. **PR #3** |
| 10 | 1 | Evaluation, `RESULTS.md` |
| 11 | 1 | Red-team report and fixes |
| 12 | 1 | Deploy and README. **PR #4** |
| Spare | 2 | Fixes found late (one already used, Phase 0) |
| **Total** | **22** | |

The full architecture needs **two more commits** than the original 20
(Phase 3 grows from 2 to 3 for restructuring and vision; Phase 4 from 1 to 2
for Neon and the branching experiment). With 20, no spare is left, and one
spare was already spent on a Phase 0 test fix. **Budget: 22** (decided 2026-11-13).

## LLM call budget (Gemini free tier)

| Work | Calls | When |
|---|---|---|
| Chunk metadata | 1 per section, cached | Phase 3, once per corpus version |
| Image extraction | 1 per image (about 10 images) | Phase 3, cached |
| A question | 3–4 (planner, answer, validators, ≤1 retry) | Phases 6–9 |
| Eval run | about 40 × 4 + 40 judge calls ≈ 200 | Phase 10, per run |
| Embeddings | paced (`ingestion/pacing.py`): ≤90 texts and ≤27,000 tokens a minute | Phases 3–4 |

Measured free-tier limits so far: `gemini-3.6-flash` 20 requests/day,
`gemini-3.5-flash-lite` 15 requests/minute (no daily cap hit at ~32/day),
embeddings ~100 texts and 30,000 tokens a minute. Failed attempts count.

## Neon notes

Neon has no true superuser. Roles made in the console or API are members of
`neon_superuser`, which has `CREATEROLE`, `CREATEDB` and (projects created
after 2023-08-15) `BYPASSRLS`. What that means here:

- **Migrations** run as that owner role. `CREATE EXTENSION vector` is
  allowed for it. Creating `finpilot_app` and setting its password needs
  only `CREATEROLE`.
- **The app role** stays `NOSUPERUSER NOBYPASSRLS` and not the table owner;
  the backend refuses to start otherwise.
- **The migrate step's embedding-model check** needs to read every org's
  chunks, so it needs `BYPASSRLS` (the owner has it).
- **Loaders never rely on bypassing RLS.** The customer-data loader writes
  each customer's rows inside a transaction bound to that customer
  (`app.customer_id`), so the policy's `WITH CHECK` passes for any owner
  role, with or without `BYPASSRLS`. Shared reference data (securities,
  prices) has no RLS.
- **Pooling**: the access context is set with `set_config(..., true)`
  (transaction-local), so it is safe behind Neon's PgBouncer in transaction
  mode.
- CI simulates this owner: a role with `CREATEROLE CREATEDB BYPASSRLS`,
  not a superuser, runs the migrations and the loader.

## Phase 1 design: customer data

Tables (migration 0009), all `customer_id`-owned except the reference data:

| Table | Key columns |
|---|---|
| `customers` | `customer_id`, `org_id`, name, segment, joined_on |
| `accounts` | `account_id`, `customer_id`, type (checking, savings, credit, brokerage), product, opened_on, status |
| `cards` | `card_id`, `account_id`, `customer_id`, last4, network, status |
| `transactions` | `txn_id`, `account_id`, `customer_id`, `card_id`, posted_on, amount (cents), merchant, category, description |
| `fees` | `fee_id`, `account_id`, `customer_id`, `txn_id`, fee_type, amount (cents), assessed_on, waived |
| `holdings` | `account_id`, `customer_id`, `symbol`, quantity, cost basis (cents) |
| `securities` | `symbol`, name, type (ETF, stock), expense ratio (reference: no RLS) |
| `prices` | `symbol`, `price_date`, close (reference: no RLS; real prices never committed) |

- Every child row repeats `customer_id`, and a composite foreign key
  `(account_id, customer_id) → accounts` makes it impossible for a row to
  claim a different owner than its account.
- One policy per owned table, for every command:
  `customer_id = nullif(current_setting('app.customer_id', true), '')`.
  No context, no rows. Staff (no `customer_id`) see no customer data.
- `users.customer_id` (nullable, unique) links a Firebase user to a customer.
- The app role gets `SELECT` only on customer data: tools read; nothing in
  the app writes it.
- Leakage tests: two or more customers with canary strings in their
  merchants and descriptions; every table, every query shape, as each
  customer and with no context; the canary never crosses.

## Decisions made on 2026-11-13

| Topic | Decision |
|---|---|
| Commit budget | **22**. The Compose-name chore before the folder rename is outside it. |
| Neon | The project exists. `NEON_MIGRATION_DATABASE_URL` (direct, owner) and `NEON_POOLED_DATABASE_URL` (pooled) are in `.env`; never printed. FinPilot creates the app role. No connection before Phase 4 unless needed. |
| SEC User-Agent | `FinPilot research project <name> <email>`, from `SEC_USER_AGENT` in `.env`; under 10 requests a second. |
| Reranker | Local cross-encoder through ONNX (fastembed), no torch. If that isn't workable, ask before adding torch back. |
| Accounts tools | Several narrow tools (balances, transactions, fees, holdings). |
| Prices | Real prices, only for signed-in demo accounts, cached in the database, with attribution and an as-of date. |
| Transactions | Fully synthetic generator with realistic patterns. |
| Investor.gov | Treated like other SEC-created content: reused with attribution, recorded in `data/SOURCES.md`. |
| Bank documents | Approved as drafts. The deposit agreement is full length (about 5,400 words, 24 numbered sections, tables), and the corpus must have enough multi-chunk documents to test chunking and retrieval (a test checks it). |
| Staff-only documents | Clearance 0 for customers, 2 for staff. |
