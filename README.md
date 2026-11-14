# FinPilot

A banking assistant for the customers of a fictional bank. It answers from the customer's own data (accounts, cards, transactions, fees, holdings), the bank's documents (fee schedules, policies, agreements, rate tables, its API spec) and public market and regulatory data, with every answer planned, run through tools, validated and traced.

**Status: Phase 0 of 12.** What runs today is the retrieval core FinPilot is built on (inherited from DocuMind): hybrid pgvector + full-text search with reciprocal rank fusion, a LangGraph pipeline that refuses rather than guesses (relevance grading, generation, a groundedness check), Firebase sign-in, and row-level security in Postgres. The plan, the architecture checklist (data sources, restructuring, chunking, metadata, Neon, retrieval, planner and agents, validation, evaluation, red-teaming) and the phase-by-phase commit budget are in [docs/FINPILOT_PLAN.md](docs/FINPILOT_PLAN.md); progress is in [docs/PROGRESS.md](docs/PROGRESS.md).

## What exists today

- **Retrieval.** Every chunk is stored once in Postgres with a pgvector embedding (`gemini-embedding-2` by default, or local `all-MiniLM-L6-v2`) and a full-text index. A query runs both legs and fuses them with reciprocal rank fusion. Questions are scoped to chosen documents (the newest one by default).
- **Pipeline.** Input validation and PII masking, a small-talk shortcut, retrieval, a relevance grade, generation with inline citations, and a groundedness check with one retry; anything that fails becomes an explicit refusal. Answers stream only after they pass the check.
- **Access.** Firebase identifies the user; Postgres decides what they may see (below). The bank's documents carry a clearance: 0 for customer-facing, 2 for staff-only.
- **Documents** come only from a loader (the bank corpus loader arrives in Phase 3); there is no upload. Embedding is paced under the Gemini free tier's limits (`backend/ingestion/pacing.py`).
- **Observability.** Every query is logged with its route, retrieved chunks, verdicts and per-stage timings; `/admin/logs` and `/admin/stats` read them, and users give thumbs up/down feedback.

## Setup

You need Docker and a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey); the free tier is enough, no billing required. The key powers all LLM reasoning (generation and the grading/groundedness checks, all on `gemini-3.5-flash-lite` by default) and the default embeddings (`gemini-embedding-2`). With `EMBEDDING_PROVIDER=local`, embeddings need no key: the sentence-transformers model (~90 MB) is downloaded from HuggingFace on first startup and cached in a Docker volume after that. Changing the embedding provider, model or dimension needs a fresh database and a reload of the documents.

```bash
cp .env.example .env
# edit .env and set GEMINI_API_KEY - the LLM calls need it
docker compose up --build
```

LLM settings in `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` or `anthropic` |
| `GEMINI_API_KEY` | — | Required when the provider is `gemini` |
| `ANTHROPIC_API_KEY` | — | Required when the provider is `anthropic` |
| `GENERATION_MODEL` | provider default | `gemini-3.5-flash-lite` / `claude-opus-4-8`; `gemini-3.6-flash` for demos (free tier: 20 requests/day) |
| `GRADING_MODEL` | provider default | `gemini-3.5-flash-lite` / `claude-haiku-4-5` |
| `GRADING_THINKING_LEVEL` | `minimal` | Gemini thinking level for the checker calls; empty = model default |
| `LLM_MAX_ATTEMPTS` | `5` | Attempts per call on 429/overloaded before the error surfaces |
| `LLM_REQUEST_TIMEOUT_S` | `60` | Seconds before a single Gemini request is abandoned; counts as one retryable attempt |
| `EMBEDDING_PROVIDER` | `gemini` | `gemini` (`gemini-embedding-2`) or `local` (`all-MiniLM-L6-v2`) |
| `EMBEDDING_MODEL` | provider default | Override the embedding model |
| `EMBEDDING_DIMENSIONS` | model default | 1536 for `gemini-embedding-2`, 384 for MiniLM; must match the database's vector column |
| `EMBEDDING_USE_VERTEXAI` | `false` | Embed through Vertex AI instead of the API key (needs billing; untested here) |

- Web UI: http://localhost:3000 (set `FRONTEND_PORT` in `.env` if 3000 is taken)
- API docs: http://localhost:8000/docs (set `BACKEND_PORT` in `.env` if 8000 is taken)
- Health check: `GET http://localhost:8000/health`

Every endpoint except `/me` and `/health` requires a Firebase ID token (`Authorization: Bearer ...`) for a user set up in Postgres, in the project named by `FIREBASE_PROJECT_ID`. The web UI signs in with Firebase email/password (the `VITE_FIREBASE_*` web config in `.env`), sends the ID token on every call, and shows a "not set up yet" screen, with the reason from `/me`, to users who aren't registered, have no org, or were deactivated.

### Running without Docker

```bash
# backend - put GEMINI_API_KEY in backend/.env or the environment
cd backend
python -m venv .venv
.venv/Scripts/activate          # source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
uvicorn main:app --reload
```

```bash
# frontend
cd frontend
npm install
npm run dev                     # Vite proxies /api to localhost:8000
```

Outside Docker the backend still needs Postgres with pgvector, because the chunks live there: set `DATABASE_URL` to the `finpilot_app` role (e.g. `postgresql+psycopg2://finpilot_app:change-me-too@localhost:5432/finpilot`) run `docker compose up postgres`, then run the migrate step once with the owner's URL given only to it: `MIGRATION_DATABASE_URL=postgresql+psycopg2://finpilot:change-me@localhost:5432/finpilot python migrate.py` (from `backend/`). The backend connects only as `finpilot_app`, which row-level security applies to; it refuses to start on a superuser or table-owner connection, or with `MIGRATION_DATABASE_URL` set. In Docker, the one-shot `migrate` service does this before the backend starts.

## Authentication and access control

**Firebase identifies the user; Postgres decides what they may see.** The web UI and the eval harness sign in with Firebase email/password and send the ID token as `Authorization: Bearer ...` on every call. The backend checks the token with the Firebase Admin SDK (allowing 5 seconds of clock skew, `FIREBASE_CLOCK_SKEW_SECONDS`), then loads the user's row from Postgres on every request: org, teams, role (`viewer`, `editor`, `admin`), integer clearance (higher sees more) and an `active` flag. Nothing about access lives in Firebase custom claims, so a change applies on the user's next request, not at their next token refresh.

- **Deny by default.** No valid token: 401. A valid token whose user has no row, no org, or `active=false`: 403 on everything except `GET /me`, which always answers and says why (`unknown_user`, `no_org`, `inactive`). `/health` is open and reports nothing about the data.
- **Roles.** Viewers ask questions and give feedback. There is no upload or access-change API: documents come only from the loader, which writes as an editor or admin of the bank's org. `/admin/*` is admin-only, also checks with Firebase that the token wasn't revoked, and shows only the admin's own org. LLM diagnostics (per-model request counts, token usage, raw provider errors) are returned only to admins.
- **Row-level security does the filtering, not Python.** The app connects as `finpilot_app`, a role that isn't the table owner, isn't a superuser and can't bypass RLS; the backend refuses to start otherwise. Every transaction first sets `app.org_id`, `app.clearance` and `app.team_ids` for that transaction only (`set_config(..., true)`), from the user just loaded. `chunks` and `documents` (and the query and feedback logs, by org) are under `FORCE ROW LEVEL SECURITY` with one policy: same org, clearance at or below the user's, and the team empty (org-wide) or one of the user's. Both retrieval legs (the HNSW vector search with iterative scans, and the full-text search) run under it. A transaction with no context sees no rows.
- **The `documents` table is the source of truth for access.** Loading a document writes its org, team and clearance (required, no default) and its chunks in one transaction, the chunks copying those values from the document row, so they can't drift apart.
- **Owner credentials never reach the app.** Migrations, setting `finpilot_app`'s password and the embedding-model check run as a separate one-shot step (`backend/migrate.py`; the `migrate` service in docker-compose, which the backend waits for). The backend gets only `DATABASE_URL` and refuses to start if `MIGRATION_DATABASE_URL` is in its environment.

### Setting up the first admin

1. In the Firebase console, create the project with email/password sign-in enabled, and create the user.
2. Put the web config (`VITE_FIREBASE_*`), `FIREBASE_PROJECT_ID` and `GOOGLE_APPLICATION_CREDENTIALS` (the path to a service account key file kept outside the repo) in `.env`. docker-compose mounts that file read-only into the backend; it never goes into an image. Without these the backend refuses to start and says what is missing. CI fails if a key file is ever committed.
3. Run the migrate step, then map the user to an admin with the owner's database URL:
   `python scripts/setup_eval_admin.py --database-url OWNER_URL` (it looks the user's uid up by `EVAL_ADMIN_EMAIL`, creates org `eval`, and is safe to re-run; it accepts only `@documind.test` accounts). Further users and teams are rows in `users` and `team_memberships`, written as the owner; there's no admin page yet.

### Known limitations of the access model

- Clearance is a single ordered level with no compartments: a user at level N sees everything at N or below in their org and teams.
- Firebase's revoked-token check is a network call, so only `/admin/*` makes it. Elsewhere a revoked token keeps working until it expires (at most an hour); setting `users.active=false` is the immediate off switch everywhere.
- The embedding-model check in the migrate step reads every org's chunks, so the migration role must be able to bypass RLS (true for the Docker Postgres owner, a superuser).

## Testing

```bash
cd backend
pip install -r requirements-dev.txt
python -m pytest tests
```

`tests/unit/` covers chunking edge cases (empty/single-sentence/boundary-sized documents), input validation (empty and oversized queries), file-type validation for all supported and rejected formats, doc-scoping logic (explicit scope, default-to-newest, empty corpus), PII redaction patterns (including false-positive guards on normal technical text), the small-talk classifier in both directions, and the embedding pacing used by the loaders.

`tests/integration/` runs the store (documents and chunks into Postgres + pgvector, a fresh database per test; embeddings stubbed, everything else real), row-level security as the app role, and the full query path through the FastAPI app with LLM calls replaced by a deterministic fake of the LLM gateway: the small-talk shortcut (asserting **zero** LLM calls), the grading-fails refusal, the ungrounded-retry-then-refuse path (asserting the documented five-call worst case), the normal answer path (asserting checkers ride `GRADING_MODEL` and generation rides `GENERATION_MODEL`), doc scoping end to end, PII never reaching a prompt or the log, and the feedback upsert with its stats aggregates. CI runs these against a `pgvector/pgvector:0.8.7-pg16` service container (the compose image) and fails if the Postgres tests would skip; locally, set `TEST_POSTGRES_URL` to a pgvector database. Unit tests that only need the log tables run on SQLite.

`tests/unit/test_llm_gateway.py` runs both real gateway classes, Gemini and Anthropic, against stubbed SDK clients through one shared contract. It covers text and structured-JSON calls, fail-safe results on refusal/block/truncation/malformed output, bounded backoff on rate limits (including Gemini's server-suggested retry delay and fail-fast on daily quotas), checker-only thinking config, and provider selection.

The suite runs in GitHub Actions on every push (`.github/workflows/tests.yml`), needs no secrets, and costs nothing. The RAGAS eval is deliberately **not** part of it — see below.

## API

| Endpoint | Purpose |
|---|---|
| `GET /documents` | Documents the caller can see, newest first; the first entry is the default query scope |
| `POST /query` | Full pipeline, single JSON response. Body: `{question, doc_ids?}`; omitting `doc_ids` scopes to the newest document. Returns `query_id` for feedback |
| `POST /query/stream` | SSE: one `stage` event per completed node, then the verified answer as `token` events, then `citations` and `done` (which carries `query_id`) |
| `POST /feedback` | `{query_id, thumbs_up, comment?}`: upserts the verdict for that query |
| `GET /admin/logs` | Recent query logs of the admin's org |
| `GET /admin/stats` | Outcome/route counts, feedback aggregates, per-stage latency percentiles |
| `GET /me` | Who the token belongs to and whether they're set up |
| `GET /health` | Liveness |

## Free-tier limits

All reasoning and embeddings run on the Gemini free tier. Limits are per project and per model and reset at midnight Pacific; failed attempts count against them. Measured so far: `gemini-3.6-flash` 20 requests/day (why it isn't the default), `gemini-3.5-flash-lite` 15 requests/minute, embeddings about 100 texts and 30,000 tokens a minute. Latency varies widely and 503 "high demand" responses are common; backoff makes this survivable, not fast. Free-tier usage may be used to improve Google's products, so no real customer data ever goes in: FinPilot's customers are synthetic.
