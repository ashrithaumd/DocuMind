# DocuMind Enterprise — progress log

## Phase 0

### LLM provider changed: Anthropic → Gemini (2026-10-22)

All reasoning calls now default to the **Gemini API free tier** (Google AI
Studio key, no billing), selected by `LLM_PROVIDER=gemini` behind the existing
gateway in `backend/llm.py`. Anthropic stays implemented and tested;
`LLM_PROVIDER=anthropic` switches back.

| Role | Gemini (default) | Anthropic (previous) |
|---|---|---|
| Generation (`GENERATION_MODEL`) | `gemini-3.5-flash-lite` (see decision below) | `claude-opus-4-8` |
| Relevance + groundedness checks (`GRADING_MODEL`) | `gemini-3.5-flash-lite` | `claude-haiku-4-5` |
| RAGAS judge (`JUDGE_MODEL`) | `gemini-3.1-flash-lite` | `claude-sonnet-4-6` |

Embeddings are unchanged (local `all-MiniLM-L6-v2`), so existing indexes stay valid.

**The July 2026 eval numbers are historical and not comparable.** The scores
in `backend/eval/report.*` and the README (faithfulness 1.0, context precision
0.917, 12/14 answered, 2026-07-15) were produced by Claude generation/grading
**and** a Claude judge. The new baseline changes both the system under test and
the judge, so a difference between the two runs can't be attributed to either one.
The Phase 0 baseline will be re-measured on Gemini.

### Decision: RAGAS judge runs at Gemini's default temperature (2026-10-22)

RAGAS 0.4's `LangchainLLMWrapper` overwrites the judge's temperature on every
call (0.01, or 0.3 when it asks for several completions). Gemini accepts that,
but Google strongly recommends keeping Gemini 3 models at their default 1.0,
because lower values can cause looping or degraded reasoning.

**Chosen:** `LangchainLLMWrapper(..., bypass_temperature=True)` with no
temperature on `ChatGoogleGenerativeAI`, so judge calls carry no temperature
and Gemini uses its default of 1.0. A one-sample probe confirmed it: both judge
calls had `temperature=None`. The cost is less deterministic judge scores, so
the baseline runs 3 passes and reports mean with min/max to show that variance.
The Anthropic judge path is unchanged (`claude-sonnet-4-6`, temperature 0).

### Decision: generate on gemini-3.5-flash-lite, project-wide (2026-10-22)

The first Phase 0 eval pass hit `gemini-3.6-flash`'s free-tier daily cap at
question 10 of 14. The quota id was
`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, with limit **20**. By the
request tally, failed attempts (503/500) count against it too.

**Chosen:** `GENERATION_MODEL` now defaults to `gemini-3.5-flash-lite`, the same
model as grading, everywhere: config, README and `.env.example`, not just the
eval. `gemini-3.6-flash` remains an opt-in for demos.

**Why:** the baseline has to be re-run after Phase 1, Phase 6 and Phase 10 with
the same generator, and development queries share the quota. A 20-request/day
generator would make every comparison take several days. The resume-critical
metrics (recall@k, MRR, ACL leakage, `retrieval_ms`) don't depend on the
generator.

**Known limitation:** the generator (`gemini-3.5-flash-lite`) and the judge
(`gemini-3.1-flash-lite`) are both small Gemini Flash-Lite models, so
self-preference bias in the RAGAS scores is possible. Every eval report states
this caveat.

### Phase 0 baseline (2026-10-22)

Artifacts: `docs/baseline/phase0/` — `retrieval_bench.json` (retrieval-only
benchmark), `results.jsonl` (one record per question), `report.json` and
`report.md` (RAGAS). One RAGAS pass; the backend ran with
`EXPOSE_LLM_DIAGNOSTICS=true` so the report has per-model request counts and
full quota errors. Generator and checkers: `gemini-3.5-flash-lite`; judge:
`gemini-3.1-flash-lite`.

**Retrieval latency (the latency baseline).** `HybridStore.search()` scoped to
`sample.pdf` (5 chunks, `all-MiniLM-L6-v2`, 12 candidates per retriever, top 6),
no LLM calls, 14 questions × 20 timed reps after 1 warm-up each:

| n | p50 | p95 | p99 | min | max |
|---|---|---|---|---|---|
| 280 | 14.9 ms | 19.0 ms | 21.9 ms | 12.3 ms | 24.3 ms |

Inside the live pipeline `retrieval_ms` was p50 26.4 / p95 30.7 ms (n=14); it
also covers the default-scope lookup and PII redaction.

**RAGAS (14 questions).** 12 answered, 2 refused, 0 errored, 0 not run.

| Metric | Mean | Scored |
|---|---|---|
| Faithfulness | 1.0 | 12/12 |
| Context precision | 0.958 | 12/12 |

- Context precision is 1.0 on every answered question except Q11 (CO2 flooding
  vs. water mist), 0.5.
- Q7 (out-of-range environmental parameter) and Q12 (emergency return vehicle)
  were refused `refused_no_context`, same as in pass 1. Both have reference
  answers in the document, so these are false refusals. Cause: the grader
  sees only the first 700 characters of each excerpt, and both answers lie
  past that cut (see Phase 6 notes).
- LLM stage latencies (grading p50 1006 ms, generation p50 811 ms,
  groundedness p50 827 ms, total p50 2790 ms) are free-tier queue-dominated
  and not a baseline.
- Requests: app `gemini-3.5-flash-lite` 38, 0 failed; judge
  `gemini-3.1-flash-lite` 115, 31 failed. 17 of the failures were the judge's
  per-minute quota, `GenerateRequestsPerMinutePerProjectPerModel-FreeTier`,
  limit **15**. No daily quota was hit.
- Caveat (also in the report): generator and judge are both small Gemini
  Flash-Lite models, so self-preference bias in the RAGAS scores is possible.

**Not yet done:** the judge-temperature decision above calls for 3 passes
reported as mean with min/max. This is a single pass, so it shows no judge
variance.

## Phase 1

Postgres + pgvector + full-text search replace Chroma and the in-memory BM25.

### Decisions (2026-10-23)

- **Embeddings: `gemini-embedding-2` at `output_dimensionality=1536`, stored
  as `vector(1536)`.** It takes 8,192 input tokens, so 600-token chunks are
  embedded whole (MiniLM truncated at 256 word-pieces), and it normalizes
  reduced dimensions itself. 1536 is one of Google's recommended sizes and
  fits pgvector's 2,000-dimension HNSW limit for `vector`, so there are no
  `halfvec` casts. A 3072-dimension `halfvec` variant stays possible later
  as a migration plus re-embed.
- **Deviation from the build prompt: AI Studio key, not Vertex AI.** The
  prompt names Vertex Gemini as the primary embedding provider. Vertex AI
  embeddings need a billing account, and Vertex express mode (the no-billing
  route) supports only listed Gemini text/image models, no embedding model;
  this project runs on the free tier with no billing. The Gemini provider
  uses the `google-genai` SDK, which serves both, so Vertex is a config switch
  inside the same provider with no code change.
- **Data-use caveat accepted.** Free-tier Gemini API content is "Used to
  improve our products". Acceptable here because every corpus in this project
  is fictional.
- **Embedding quota is discovered, not looked up.** Google doesn't publish
  free-tier embedding limits (only AI Studio shows them). `sample.pdf` is 5
  chunks, so the limit is read from the first real calls and their quota
  errors, as in Phase 0, including whether one batch request counts as one
  request.
- **Embedding free-tier quota, measured 2026-10-23.** A 429 after 22 batch
  requests of 5 texts reported
  `EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier`, limit
  **100** per minute (metric `embed_content_free_tier_requests`). Every text
  in a batch counts as one request: batching saves round trips, not quota.
  The daily limit hasn't been hit yet; the gateway stops on a daily 429 and
  logs its quota id and value.
- **`gemini-embedding-2` task prefixes.** Queries:
  `task: question answering | query: ...` (queries are questions answered
  from the passages); documents: `title: none | text: ...` (chunks carry no
  real title). Each text is sent as its own `Content`: the SDK merges a
  plain list of strings into one multi-part content and returns a single
  combined embedding for this model (caught by a count check on the first
  real call). Vertex's `embedContent` takes one content per call, so the
  Vertex path sends one text per request (untested here: no billing).
- **Full-text search config: `english`** (stemming + English stop words).
  BM25 tokenized without stemming, so the sparse leg's ranking changes too;
  the benchmark reports each leg's ranks separately so that shows up.
- **Full-text query semantics: any term (OR), ranked by `ts_rank_cd`.**
  `plainto_tsquery`/`websearch_to_tsquery` AND every term together; on the
  sample, the gold page of only 4 of 14 questions contained every query
  term, so the sparse leg would go silent on most questions. The store
  turns the `plainto_tsquery` ANDs into ORs: same stemming and stop words,
  BM25-like any-term matching. Only matching chunks get a sparse rank.
- **Filtered HNSW search uses iterative scans** (`SET LOCAL
  hnsw.iterative_scan = relaxed_order`, re-sorted by exact distance in a
  materialized CTE). A test with the HNSW index forced shows the control
  query without them losing a small document's rows to the `doc_id` filter.
- **Comparison design.** Run A: Chroma + BM25 + MiniLM (Phase 0). Run B:
  pgvector + FTS + MiniLM, the migration effect (B vs A). Run C: pgvector +
  FTS + `gemini-embedding-2`, the embedding effect (C vs B). The latency
  claim uses `search_ms`; query embedding (`embed_ms`) is reported
  separately, since Gemini's is a network call. Retrieval quality is MRR and
  recall@1/@3 against hand-verified gold pages. RAGAS: one pass each for A
  and B, three passes (mean, min/max) for C.
- **Free memory before long eval runs: 2 GB for Gemini-embedding runs, 4 GB
  for anything with `EMBEDDING_PROVIDER=local`.** The 4 GB rule was sized
  for runs that load torch (run B's MiniLM backend: ~690 MB resident). A
  Gemini-embedding backend (~240 MB) and the eval harness (~230 MB) don't
  load it. Below the threshold, the run doesn't start. Claude Code's
  low-memory reaper stays enabled.
- **Before pushing, run the full suite in an environment that matches CI**:
  a fresh environment with exactly the files CI installs, or the backend
  Docker image, against pgvector. The local venv isn't enough, because it
  has packages CI doesn't: `test_eval_harness_does_not_load_torch`
  (5c6a524) passed locally and failed in CI, which never installed
  `eval/requirements.txt`.
  Changes to the CI workflow itself are verified by the CI run.
- **Phase 1 close-out suite run (2026-11-05, 15053b8): 200 passed, 0 skipped.**
  It ran on a clean export of the commit (no `.env`) in a fresh venv built with
  CI's three install commands, against `pgvector/pgvector:0.8.7-pg16` with
  `REQUIRE_POSTGRES_TESTS=1`. The host was Windows, not CI's `ubuntu-24.04`, so
  the CI run still has the last word. The eval install uninstalled no app
  package.
  - **Memory limit, this run only:** at least 3.5 GB free to start, with Docker
    up, and stop below 1 GB free. The usual 4 GB rule was sized for runs that
    load torch's model weights; this suite's heaviest measured process was far
    smaller.
  - **Measured (2-second samples, so short spikes can be missed):** 5.18 GB free
    at the start, 4.29 GB at the lowest. The pytest process tree peaked at
    0.47 GB and the Docker VM at 1.52 GB (during install; 1.23 GB during
    pytest). The stop rule never fired.

### Results: runs B and C (2026-10-23)

Artifacts in `docs/baseline/phase1/`: `retrieval_bench_b.json`,
`retrieval_bench_c.json` (retrieval-only benchmark), `ragas_b/` (one RAGAS
pass) and `ragas_c/pass1..3/`. Every run used the same 5-chunk `sample.pdf`, 14
questions, generator and checkers `gemini-3.5-flash-lite`, judge
`gemini-3.1-flash-lite`, on the same machine. Runs B and C ran before the
grader fix (8cbc2e4), so Q7 and Q12 are still refused in both (see below).

**Retrieval benchmark** (no LLM calls; 14 questions × 20 timed searches
after a warm-up, `search_ms` on a stored query vector; `embed_ms` timed
separately, 3 reps per question):

| Run | Store | Embeddings | search p50 / p95 | embed p50 | Fused MRR | Fused recall@1 | Dense MRR | Sparse MRR |
|---|---|---|---|---|---|---|---|---|
| A | Chroma + BM25 | MiniLM | 3.2 / 3.7 ms | 12.1 ms | 0.929 | 0.857 | 0.893 | 1.0 |
| B | pgvector + FTS | MiniLM | 8.3 / 9.8 ms | 13.8 ms | 1.0 | 1.0 | 0.893 | 0.964 |
| C | pgvector + FTS | gemini-embedding-2 | 11.5 / 15.9 ms | 293.2 ms | 0.929 | 0.857 | 0.929 | 0.964 |

(Run A is `docs/baseline/phase0/retrieval_bench_split.json`, the same
embed/search split.) The migration costs about 5 ms of `search_ms` at p50
on this corpus: two SQL round trips to a Postgres container instead of an
in-process Chroma query and BM25 scoring. At 5 chunks that is connection
and query overhead, not index work; it says nothing about how either store
scales. Recall@3 is 1.0 in every run and leg.

**RAGAS.** Run B (one pass): 12 answered, 2 refused; faithfulness **1.0**
(12/12), context precision **1.0** (12/12). Run C (three passes): see the
table under "Judge variance" below.

**Q7 and Q12 are false refusals from the 700-character grader cut, not
retrieval misses.** Both are `refused_no_context` in runs B and C (and in
run A). In run B's `results.jsonl`, the gold chunk is retrieved at rank 1
for both: Q7's page 2 chunk (1,008 characters) has "5 minutes" at
character 942 and "Tier-2" at 963; Q12's page 4 chunk has "4 minutes" at
756. The grader saw only the first 700 characters of each excerpt. In run
A, retrieval ranked Q7's gold page 2nd. The fix and its before/after pass
are below.

**Run C's Q3 and Q9 context precision (0.5) is an RRF tie artifact, not
worse retrieval from Gemini embeddings.** In run C's benchmark, Q3 and Q9
are the two questions whose gold rank is decided by a tie
(`questions_gold_rank_decided_by_tie: 2`): the gold page is dense rank 2,
sparse rank 1, and the other chunk is dense 1, sparse 2, so both score
1/61 + 1/62. The dense-first tie-break (`_fused_order` in
`ingestion/store.py`) puts the other chunk first, and RAGAS scores the
gold chunk at position 2 as 0.5. Run B has no ties. Gemini's dense leg
does rank the gold page 2nd on those two questions, so a sparse-first rule
would flip both to 1.0; but no tie-break rule is evidence about embedding
quality. The dense legs differ by one question (dense MRR 0.929 for Gemini vs 0.893
for MiniLM), which on 5 chunks is noise, not evidence: one question moves
MRR by 0.036, and a tie-break rule moves context precision by 0.083.
**This corpus can't separate embedding models in either direction.** That
comparison belongs to the Phase 3 corpus.

**Two latency numbers, not comparable.** Run B's in-pipeline
`retrieval_ms` is p50 46.8 / p95 83.6 ms, against a benchmark `search_ms`
p50 of 8.3 ms. `retrieval_ms` times the whole retrieve node: the
default-scope lookup (`list_documents`, a SQL query), the query embedding
(MiniLM on the CPU, ~14 ms), the search, and PII redaction of each
retrieved chunk, while the same machine runs the eval harness and its
judge calls. (Redaction of the question happens earlier, in
input validation, and isn't included.) `search_ms` is the store's hybrid
search alone on a precomputed vector, with nothing else running. Compare
stores with `search_ms`; use `retrieval_ms` for what a user waits for. In
run C the query embedding is a network call (~293 ms p50), so run C's
`retrieval_ms` p50 is 418.7–466.9 ms across passes.

**Judge variance (run C, 3 passes).** All three passes produced identical
scores on every question:

| Metric | Mean of 3 passes | Min | Max | Scored |
|---|---|---|---|---|
| Faithfulness | 1.0 | 1.0 | 1.0 | 12/12 each |
| Context precision | 0.917 | 0.917 | 0.917 | 12/12 each |

Outcomes were identical too (12 answered, 2 refused each pass). Judge
traffic varied (98 / 94 / 85 requests; 14 / 10 / 1 failed attempts, all
the 15-per-minute quota, all retried); the scores didn't. This doesn't
mean the judge is deterministic: it runs at temperature 1.0. These 12
questions didn't show judge variance: every claim is a short fact lifted
from one page, and context precision on 5 chunks has few possible values.
Q7, which these passes refused, did show it once answered (see "Q7
re-judge" below). A harder corpus is needed before the min/max columns say
anything.

### Grader fix: before/after (2026-10-23)

Fix: 8cbc2e4, "fix(graph): grade full excerpts instead of the first 700
characters", with a regression test that places the answer past character
700 (`tests/unit/test_grading.py`). The after pass ran on run C's setup
(`ragas_c` database, `gemini-embedding-2`, same 5 chunks, no re-upload) on
committed code: 13 questions on `9046638` (clean), and Q7's final record on
`8874fb7`, which has no `backend/` diff from it. The report records
`8874fb7` with its own `results.jsonl` as the only uncommitted file.
Artifacts: `docs/baseline/phase1/ragas_c_fix/` and
`docs/baseline/phase1/grading_tokens/`.

**The first after-pass report was replaced.** Doc drafts (`README.md`,
`docs/PROGRESS.md`, `.env.example`) were edited while the pass ran, so its
report recorded `9046638` *with uncommitted changes*: the same code
commit, no code changes, only tracked docs. The drafts were stashed and
the pass re-run with `--resume` on the clean tree, which rewrote the
report as `9046638 (clean)`. `--resume` keeps finished questions, so the
13 fully scored records from the first run were reused, all on the same
code; only Q7 (unscored, see below) was queried and judged again.
**Rule: no edits to tracked files while an eval runs.**

**Outcomes.** Before is run C (all three passes identical); after is one
pass.

| | Before (run C) | After |
|---|---|---|
| Answered | 12/14 | **14/14** |
| False refusals | 2 (Q7, Q12) | **0** |
| Faithfulness | 1.0 (12/12) | 0.964 (14/14) |
| Context precision | 0.917 (12/12) | 0.929 (14/14) |

- **Q12** is answered and fully scored: faithfulness 1.0, context
  precision 1.0.
- **Q7** is answered ("…triggers a Tier-2 alarm to the Systems Engineer on
  duty [1]", matching the reference) and fully scored: faithfulness 0.5,
  context precision 1.0. It took three attempts. The first attempt's two
  judge jobs timed out (RAGAS `TimeoutError`); the first `--resume` scored
  faithfulness 1.0, then hit the judge's daily quota
  (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, limit 500) on
  context precision; the second, after the reset, gave the final record.
- **No other question's outcome or score changed**: the other 12 match all
  three run C passes exactly, including Q3 and Q9 at 0.5 (the tie
  artifact above). Context precision moves from 0.917 to 0.929 only
  because Q7 and Q12 each add a 1.0; over the same 12 questions it's
  unchanged. Faithfulness moves from 1.0 to 0.964 only because of Q7's 0.5.

**Q7 re-judge (2026-11-05).** Q7 scored faithfulness 1.0 and then 0.5 on
essentially the same answer, so `eval/rejudge_faithfulness.py` (f727f1a)
re-judged the stored answer against its stored contexts 5 times, with no
pipeline call, keeping each statement's verdict
(`docs/baseline/phase1/ragas_c_fix/q7_rejudge.jsonl`). Scores: **1.0, 1.0,
0.0, 1.0, 1.0** (mean 0.8; 10 judge requests).

This supports judge variance: the input never changed and the score covered
0.0 to 1.0. The answer restates the source sentence nearly word for word,
and four passes marked both extracted claims supported. Pass 3 extracted
the rule's condition as its own claim ("An environmental parameter is out
of range for more than five minutes"), then rejected both claims because
the context doesn't say that event occurred, while quoting the sentence
that supports the answer. The 0.5 record kept no verdicts, so its cause
isn't known.

The reported mean stays 0.964, as measured. Q7 alone moves it between 0.929
(Q7 at 0.0) and 1.0 (Q7 at 1.0), so differences of that size on this dataset
aren't meaningful. Only Q7 was re-judged; the other 13 questions' variance
isn't measured.

**Grading-call input tokens per question.** Before/after numbers are
*counted with countTokens on reconstructed prompts*
(`eval/count_grading_tokens.py`): the full grading request (system prompt,
redacted question, numbered excerpts, JSON schema) over the same retrieved
chunks, cut to 700 characters (before) and whole (after), on
`gemini-3.5-flash-lite`.

| | Mean | Min | Max | Total (14) |
|---|---|---|---|---|
| Before: 700-char excerpts | 950.1 | 946 | 954 | 13,301 |
| After: full excerpts | 1,228.1 | 1,224 | 1,232 | 17,193 |
| Increase | **+278 (+29%)** | +278 | +278 | +3,892 |

The increase is the same for every question because every question
retrieves all 5 chunks (938–1,008 characters; 1,259 characters past the
cut in total, about 4.5 characters per token).

After-pass numbers *measured from live calls* (provider usage metadata,
recorded since b79d2b8): 14 grade calls, 17,207 prompt tokens, mean
1,229.1, no thinking tokens. **Gap: live is 1 token above the countTokens
full-excerpt figure on every question** (+14 in total, 0.08%). The count
is otherwise exact; the cause of the constant +1 isn't established.
For context, generate calls averaged 1,254.1 and groundedness calls
1,254.4 prompt tokens: both already saw full excerpts before the fix.

**How it scales.** The +29% is specific to this corpus, whose pages are
about 1,000 characters, so the cut removed only ~300 characters per
excerpt. At the configured 600-token chunks (~2,400 characters) the 700-
character cut kept about 29% of each excerpt, so full-excerpt grading
costs roughly 3.4x the old version per excerpt. With the fixed prompt
overhead measured here (~180 tokens) and 6 excerpts, that is roughly
1,100 → 3,350 input tokens per grading call, about 3x. The fix is still
correct: a grader that can't see the answer is broken. Cheaper grading
belongs in Phase 7 (see Phase 7 notes), measured against the
false-refusal rate.

## Phase 2 plan

Status (2026-11-07): plan and RLS decision recorded; commits 1-8 done (8: eval admin; its live one-question eval run is
still pending; 9: frontend lint fix and frontend CI; 10: frontend login) (CI chore + guard,
migration 0003, get_principal and /me, revoked-token check on /admin/*,
clock-skew fix, live Firebase check script, documind_app role and RLS on
chunks and documents, separate migrate step, documents table as the
write path with migration 0006, log tables under RLS by org and
admin-only /admin/* and diagnostics with migration 0007).

### Auth provider: Firebase Auth (option A)

- **Firebase Auth only identifies the user.** Org, teams, role and clearance
  live in Postgres and are read on every request to set the RLS context.
  No custom claims for authorization: a claim change reaches a token only
  at the next sign-in or token refresh, and claims are capped at 1,000
  bytes.
- **Plain Firebase Auth, no billing, until Phase 9.** The free Spark plan
  allows 3,000 daily active users. Identity Platform (multi-tenancy, MFA,
  SAML/OIDC) isn't needed: org lives in our own tables.
- **Rejected:** Keycloak (about 1 GB of RAM per instance at idle on top of
  Docker's 1.2-1.5 GB; another service to run on Cloud Run) and app-issued
  JWTs (we would own password storage, resets and lockout).

### Required behavior

1. **Disabling a user takes effect immediately.** A `users.active` flag is
   checked on every request, and admin routes also use Firebase's
   revoked-token check, so neither waits for the token to expire. Both are
   tested.
2. **Deny by default.** A valid Firebase user who isn't in our tables, or has
   no org, gets 403 on everything except a "who am I" endpoint (`GET /me`).
   Tested.
3. **No emulator in tests.** Tests replace the token check with a fake
   verifier: no emulator, no network. The emulator is optional, only for
   trying the login UI locally.
4. **The eval harness gets a seeded admin test user**, so it can read
   diagnostics once they are admin-only.

### Decision: RLS policy shape (decided 2026-11-05)

**Option 1: per-request settings checked against access columns on each
`chunks` row.**

- **Context per request.** Each request sets `app.org_id`, `app.clearance`
  and `app.team_ids` for its transaction only (`set_config(..., true)`),
  from the user rows it has just read. Unset settings read as NULL, so no
  rows match.
- **Policy, roughly:** `org_id = <org> AND clearance <= <clearance> AND
  (team_id IS NULL OR team_id = ANY(<teams>))`. Each row is checked on its
  own columns with no join, which works with the HNSW iterative scans.
- **Trade-off accepted:** the policy trusts the values the backend sets.
  They come from Postgres in the same request, and `documind_app` can't
  bypass RLS.
- **Rejected:** a policy that looks up teams and clearance itself from
  `app.user_id` (a lookup per candidate row in the vector search, and the
  app role would need to read the user tables).

The user's four changes on top:

1. **A `documents` table is the source of truth for access.** Its org, team
   and clearance are copied onto every chunk of that document in the same
   transaction, on upload and on any access change. A test checks that an
   access change updates every chunk at once. The policies read only the
   copy on each row, so they still need no join.
2. **RLS on the query and feedback log tables, by org**, with a leakage test.
3. **Clearance is an integer level** (higher sees more).
4. **Credentials stay out of the repo.** The service account key file
   pattern goes in `.gitignore` and the variables in `.env.example`, and a
   CI check fails if a key file is committed.

### Known limitations (accepted for Phase 2)

- **One team per document.** A document is either org-wide (`team_id` NULL)
  or belongs to exactly one team; sharing it with two teams isn't possible.
- **Clearance is a single ordered level**, with no compartments: anyone at
  level N sees everything at N or below in their org and teams.

### Prerequisites from the user

- A Firebase project on the Spark plan with email/password sign-in enabled,
  and its web config (not secret).
- A service account key for the revoked-token check, kept outside the repo.
  Phase 9 replaces it with Workload Identity.

### Checklist (one commit each)

**Test rule from commit 5 on (decided 2026-11-06):** permission tests sign in
explicitly as a viewer (or whichever role the test is about), never through
the `client` fixture's default admin, so a test can't pass because the
caller happened to have every permission.

1. **CI chore, credential guard and Phase 2 decisions.** Bump
   `actions/checkout` and `setup-python` to their Node 24 versions. Add the
   service account key file pattern to `.gitignore` and the Firebase
   variables to `.env.example`. Add a CI step that fails if a key file is
   committed, matched by file name and by content (a JSON file with
   type `service_account`, or one with a `private_key` field), so a renamed key
   is caught too. Record the decisions above. This lands before any key
   exists on the machine.
2. **Migration 0003: user and access tables.** `orgs`, `teams`, `users`
   (`firebase_uid` unique, `org_id`, `role` viewer/editor/admin, integer
   `clearance`, `active`) and `team_memberships`. A `documents` table
   (`doc_id`, `filename`, `org_id`, `team_id` nullable, integer `clearance`,
   `uploaded_by`, `uploaded_at`) as the source of truth for access, plus
   `org_id`, `team_id` and `clearance` copied onto `chunks`. Existing chunks
   get no org, so they're invisible.
3. **Identifying the user on each request.** A token-checker interface with
   a real Firebase checker and a fake one for tests. A `get_principal`
   dependency checks the token, loads the user, and checks `active` and org
   on every request; otherwise 403. `GET /me` always answers and says
   whether the user is set up. Replaces `require_api_key`. Tests: unknown
   user, no org, inactive user, `/me`.
4. **Revoked-token check on admin routes**
   (`verify_id_token(check_revoked=True)`). Tests: a revoked token and a
   deactivated user, each refused on the very next request.
5. **`documind_app` role and RLS on `chunks` and `documents`.** Migrations
   run as the owner; the app connects as `documind_app` (not the owner,
   can't bypass RLS), with `FORCE ROW LEVEL SECURITY` on both tables.
   Separate connection URLs for migrations and the app (both in
   `.env.example`). The store sets the context in every transaction.
   Leakage tests on real Postgres: another org, another team, higher
   clearance, no context (zero rows), and the app role can't bypass the
   policy.
6. **Uploads and access changes keep chunks in step with `documents`.**
   Uploads: editors and admins only; the `documents` row and its chunks are
   written in one transaction, with no team you're not in and no clearance
   above your own. Access changes: one store function updates the
   `documents` row and every chunk of that document in the same
   transaction (used by commit 8's script). Tests: an access change updates
   every chunk at once, and no chunk's access ever differs from its
   document's. `/documents` reads the `documents` table under RLS.
   **Also in this commit: drop the clearance default (migration 0006;
   0004 is commit 5's RLS, 0005 drops its SECURITY DEFINER function).**
   0003 gave `chunks.clearance` and `documents.clearance` `NOT NULL DEFAULT
   0`. The commit that makes chunk writes copy org, team and clearance from
   the `documents` row also drops both defaults, so a write that forgets
   clearance fails instead of defaulting to the lowest level. Tests: a chunk
   insert with no `documents` row fails, and a chunk or document insert
   without clearance fails. Deferred from before commit 3 (decided
   2026-11-06): until this commit the write path has no `documents` row to
   copy from, and the default can't leak anything in the meantime, because
   new chunks have no org and are invisible under RLS.
7. **Admin-only diagnostics, and RLS on the log tables.** Query and
   feedback logs record `user_id` and `org_id`, under RLS by org with
   `FORCE ROW LEVEL SECURITY`. `/admin/*` is admin-only. LLM diagnostics go
   to admins instead of following `EXPOSE_LLM_DIAGNOSTICS`. Tests: a
   leakage test showing one org's admin can't read another org's query or
   feedback rows, and non-admins are refused.
8. **Eval admin user.** A script sets up an `eval` org and an admin mapped
   to a real Firebase test account (also how the first admin is created),
   and changes a document's access through commit 6's function.
   The harness signs in through Firebase's REST API and refreshes the token,
   since runs can last over an hour. `sample.pdf` is uploaded again into the
   eval org. Checked with a one-question eval run.
9. **Frontend lint fix and frontend CI.** Fix `src/App.tsx:24`; add
   `npm ci && npm run lint && npm run build` to CI.
10. **Frontend login.** Firebase JS SDK sign-in, the ID token on every
    request, and a "not set up yet" screen for 403s.
11. **Docs and the Phase 2 PR.** README auth section, including the known
    limitations above; Phase 2 results here; the full suite in a
    CI-matching environment; then the PR.

Not in Phase 2: an admin web page for assigning users (commit 8's script
does it). Until commit 8 lands, the eval harness can't authenticate.

## Phase 2 results

Status (2026-11-07): Phase 2 complete on `enterprise`, every push CI
green; commit 11 (these docs) opens the Phase 2 PR.

### Commits

| # | Commit | What |
|---|---|---|
| 1 | 96ab3cb, afff9f8, 8a4b655 | Node 24 actions, credential guard (key files by name and content), this plan |
| 2 | 33abbee | Migration 0003: orgs, teams, users, team_memberships, documents, access columns on chunks |
| - | c7be876 | Plan: the clearance-default drop moves to commit 6 |
| 3 | 31427c3 | Firebase token checker, `get_principal`, `GET /me`; replaces the shared API key |
| 4 | 7dd619e | Revoked-token check on `/admin/*` |
| - | 222e76a | Fix: 5 s clock-skew tolerance on token times (found by the live check) |
| - | 5b268ae | `scripts/live_firebase_check.py` |
| 5 | e22d501 | Migration 0004: `documind_app` role, RLS on chunks and documents |
| - | 8e077f6 | One-shot migrate step; owner credentials out of the backend; 0005 drops 0004's SECURITY DEFINER function |
| 6 | 9efc971 | Migration 0006: documents table as the write path, no clearance default |
| 7 | 60bb864 | Migration 0007: RLS on the log tables; admin-only `/admin/*` and diagnostics; editor/admin-only access changes |
| 8 | caad3c4 | Eval admin setup script; the eval harness signs in and refreshes its token |
| 9 | 7853dd1 | Frontend lint fix (`App.tsx` setState in an effect); frontend CI job |
| 10 | e40d1fe | Frontend login, signed API calls, "not set up yet" screen |
| - | b5bcecb | Frontend on Node 26 (current LTS; Node 20 is end of life) |
| - | 379059c | Fix: Firebase key mounted read-only in compose; startup fails fast on missing Firebase config |
| 11 | (this commit) | README auth section, these results |

### Decisions made during the build

- **Clearance default dropped with the write path (2026-11-06).** Dropping
  it before commit 6 would have broken uploads for three commits; until
  then new chunks had no org, so the default couldn't expose anything.
- **Trailers follow the model running the session (2026-11-06);** pushed
  commits weren't rewritten.
- **Permission tests sign in explicitly as the role under test**, never
  through the client fixture's default admin (from commit 5 on).
- **Clock skew of 5 s (2026-11-06).** The live check found that this
  machine's clock ran about 1.5 s behind Google's, and a token used right
  after sign-in failed as "used too early". Configurable, capped at
  Firebase's 60 s.
- **Owner credentials out of the app process (2026-11-07):** a separate
  migrate step; the backend refuses `MIGRATION_DATABASE_URL`. This moved the
  embedding-model check out of the store and let migration 0005 drop the
  SECURITY DEFINER function 0004 had added for it.
- **`APP_DB_PASSWORD` and `POSTGRES_PASSWORD` are required by docker-compose
  (no defaults)**, and the default Postgres owner password was rotated in
  the local volume.
- **Changing a document's access is editor/admin only**, enforced in the
  API and in the store, so a script can't skip it (2026-11-07).
- **The upload form requires a clearance**: who may read a document is
  always an explicit choice.

### Bugs the tests and live checks found

- A token used within a second of sign-in was rejected (clock skew; live
  check) — fixed in 222e76a.
- Firebase's `UserDisabledError` isn't an invalid-token error and would
  have surfaced as a 500 on admin routes — caught and tested in 7dd619e.
- Setting the app role's password broke on a `%` in it (SQLAlchemy
  %-formatting) — found by a test with a deliberately awkward password,
  fixed in e22d501.
- The guard's own test file contained the key patterns and failed CI on
  the first push — the test now builds them at runtime (8a4b655).
- In the compose stack, `GET /me` returned 500 for every signed-in user
  (found in the browser walkthrough). Commit 3 assumed a token's signature
  check needs only the project id and that credentials load lazily; in
  fact firebase-admin loads credentials when its auth client is created,
  and the backend container had no `GOOGLE_APPLICATION_CREDENTIALS`. The
  live check passed only because it ran on the host with the key set, and
  the unit tests replace `verify_id_token`. Fixed in 379059c: compose
  mounts the key read-only, and the backend checks Firebase config at
  startup, before the database, and exits with what to set.

### Test results

- Backend suite, locally against a fresh pgvector container with Postgres
  tests required: 332 passed, 0 skipped at 379059c, the last backend
  change. Earlier counts: 235 (commit 3), 242 (4), 245 (clock-skew fix),
  275 (5), 278 (migrate step), 298 (6), 319 (7), 327 (8).
- **CI-matching run for the Phase 2 PR** (2026-11-07, at 379059c plus these
  docs): Linux containers like the CI job: Python 3.12.8 with CPU-only
  torch, the app, test and eval requirements installed with
  `constraints.txt` as in the workflow, git for the credential-guard tests,
  and a pgvector 0.8.7 service with `REQUIRE_POSTGRES_TESTS=1`: 332 passed,
  0 skipped. Frontend on Node 26.13.0: `npm ci`, lint and build pass.
- CI green on every push: backend job throughout, plus the frontend job
  (`npm ci`, lint, build) from commit 9.
- Live Firebase check (`scripts/live_firebase_check.py`, real Firebase, a
  throwaway database and backend): passed after commits 4, the clock-skew
  fix, 5, the migrate step, 6 and 7. It signs in, gets `unknown_user` from
  `/me`, sets the user up as an admin, reads `/admin/stats`, revokes the
  user's tokens (old token: 401 on `/admin/stats`, 200 on `/documents` by
  design), and a fresh sign-in used immediately works.
- Leakage tests on real Postgres cover another org, another team, higher
  clearance and no context (zero rows) on both retrieval legs and the API;
  the app role's attempts to bypass or drop the policies; the access
  context not surviving on a pooled connection; and the log tables, for a
  viewer and for another org's admin.

- **One-question eval, signed in as the eval admin** (2026-11-07,
  `run_eval.py --ingest --clearance 0` on a throwaway stack: migrate step
  as the owner, backend as documind_app, `setup_eval_admin.py`, the first
  question of `dataset.json`). Sign-in through the Firebase REST API, the
  upload of `sample.pdf` into the eval org (5 chunks) and the query all
  worked, and the report again has per-model request counts and token
  usage, now from the admin diagnostics. No token or password appears in
  the report or results files.
  - First run (code b5bcecb, before 379059c): answered, but not scored. The
    judge, `gemini-3.1-flash-lite`, returned 503 "high demand" on all 20
    attempts (free-tier load, not a code problem).
  - Re-run (code 379059c, with this commit's docs uncommitted), same
    judge: scored.

    | | |
    |---|---|
    | Outcome | answered |
    | Faithfulness | 1.0 (1/1 scored) |
    | Context precision | 1.0 (1/1 scored) |
    | Requests: gemini-embedding-2 (app) | 1, 0 failed |
    | Requests: gemini-3.5-flash-lite (app) | 3, 0 failed |
    | Requests: gemini-3.1-flash-lite (judge) | 7, 0 failed |
    | Tokens: grade | 1,226 prompt, 5 output |
    | Tokens: generate | 1,251 prompt, 25 output |
    | Tokens: groundedness | 1,249 prompt, 20 output |
    | retrieval_ms | 352.2 |

    One question proves the harness end to end; it says nothing about
    quality. The full eval belongs to Phase 6.

- **Browser walkthrough** (2026-11-07, by the user, on the docker-compose
  stack at localhost:3001, after 379059c):
  - Signed in as eval-admin: the header showed
    "eval-admin@documind.test · admin" and Sign out.
  - Uploaded `sample.pdf` (whole organization, clearance 0, the only level
    offered to a clearance-0 admin): "5 pages · 5 chunks indexed", listed
    under Documents.
  - Asked "At what altitude does the Aurora-9 station orbit?": answered
    "410 kilometers [1]", cited to `sample.pdf` p. 1, every pipeline stage
    shown, 3.2 s.
  - Signed out: back to the sign-in screen.
  - Signed in as `not-set-up@documind.test` (in Firebase, no DocuMind row):
    "Not set up yet", Reason: unknown_user, with Sign out.
  - Before the fix, the same walkthrough stopped at sign-in with "Can't
    reach DocuMind. The backend answered 500." (see bugs above).

## Phase 2 notes

- **LLM diagnostics become admin-only.** Phase 0 gates the raw provider error
  text and per-request `llm_usage` in `/query` behind `EXPOSE_LLM_DIAGNOSTICS`
  (default `false`; only the eval harness's backend turns it on). With the
  flag off, clients get provider-neutral errors and the details stay in the
  server log. Phase 2 replaces the global flag with a role check: only admin
  callers get diagnostics.
- **Fix the pre-existing frontend lint error.** `src/App.tsx:24` calls
  setState synchronously inside an effect (`react-hooks` rule). It predates
  the Phase 0 dependency pins, which didn't change any resolved version.
- **Add the frontend to CI:** `npm ci && npm run lint && npm run build`
  (today CI runs only the backend suite, so this error went unnoticed).
- **CI chore: move the actions off Node 20.** CI warns that
  `actions/checkout@v4` and `actions/setup-python@v5` target the deprecated
  Node.js 20 and are being forced onto Node 24. Bump both to their Node 24
  majors, checked against each action's release notes.

- **Owner credentials stay out of the app process (decided 2026-11-07).**
  Migrations, setting documind_app's password and the embedding-model check
  run as a one-shot migrate step (`backend/migrate.py`; the `migrate` service
  in docker-compose, which the backend waits for with
  `service_completed_successfully`). The backend gets only `DATABASE_URL`
  (documind_app) and refuses to start if `MIGRATION_DATABASE_URL` is set.

## Phase 3 plan

Status (2026-11-07): plan and decisions recorded; work on `phase3-corpus`
(main merged in at d74cdd7). One commit per item; CI green after each push.

Goal: a realistic two-org corpus (kb/: 50 documents, 303 chunks at the
default chunking, access in kb/manifest.yaml) loaded for real, and proof,
on that corpus, that access control holds: no leak across orgs, teams or
clearance levels, under adversarial prompts, through versioning and
deletion, and with the vector index still used under RLS.

Quota: items marked (quota) use real Gemini calls; everything else runs
on fakes in CI. Memory rules: at least 2 GB free for runs that call Gemini,
4 GB for anything that loads torch; checked before every run.

### Decisions (2026-11-07)

- **A dedicated Phase 3 database**: a local pgvector container with a
  published port (localhost only) and its own volume, separate from the
  compose stack used for demos.
- **Test users** for every access boundary in `kb/test-users.yaml`, seeded
  through the owner path; the automated tests use the same file with the
  fake token checker. **Three real Firebase demo users**:
  `aurora-viewer@documind.test` (org-wide, clearance 0),
  `bluefin-finance@documind.test` (finance, 3) and
  `bluefin-ml@documind.test` (ml, 3: must not see the finance-only
  Gullwing plan). Their passwords are generated, stored only in the local
  `.env`, never printed.
- **Canaries live in the corpus files**: unique strings near the start and
  end of each restricted document, listed in `kb/canaries.yaml`. Every
  negative canary test has a positive control: a user who can see the
  document retrieves its canary, so the negative tests can't pass
  vacuously.
- **Versioning**: an upload may name the document it supersedes; the new
  row records its version and `supersedes_doc_id`; default retrieval uses
  only the latest version of each chain; older versions answer only when
  asked for by id. The corpus's 2025/2026 handbooks and expense policies
  stay unlinked (independent documents), so the version-trap eval items
  stay traps.
- **Deletion**: a hard delete of the document and all its chunks in one
  transaction, under RLS, with an audit row in a new `document_events`
  table (who, when, doc id, filename; under the same org RLS). Only the
  document's uploader or an admin of its org may delete it: another
  editor who can see it gets 403, and anyone who can't see it gets 404
  (indistinguishable from a document that doesn't exist).

### Checklist (one commit each)

1. **This plan.**
2. **Phase 3 database**: a compose file with only a pgvector service
   (localhost port, named volume, required passwords generated into the
   local `.env`), and a helper that starts it, runs the migrate step and
   runs commands with its URLs in the environment, without printing them.
3. **Loader: pacing, resume, reporting.** Embedding requests paced under
   the measured free-tier limit of 100 per minute (default 90, every text
   counts as one request); per-minute 429s retried with backoff; a
   daily-quota error stops cleanly with how far it got and a distinct exit
   code, and a re-run resumes (each document loads in its own transaction,
   so nothing half-loaded is left). Reports chunks per org and embedding
   requests used. The embedder can be injected, so tests load the real
   corpus with fake embeddings. Tests: the pacing with a fake clock, a
   quota stop then a resume with no duplicates, the counts.
4. **(quota) Load the corpus** into the Phase 3 database with real
   embeddings (about 303 requests); record chunks per org and requests used.
5. **Test users** (`kb/test-users.yaml`, `scripts/seed_test_users.py`, owner
   path), plus the three Firebase demo users.
6. **Leakage tests on the real corpus, through the API**: cross-org
   (mirrored facts), cross-team (including the clearance-0 pair #13/#35),
   clearance (summary vs full report, salary bands), old versions that are
   team-only (#17/#18, #28/#29), and #26 (team plus clearance 3). Expected
   visibility comes from the manifest (`can_see`), not a second list.
7. **Canaries** in the restricted documents, with positive controls; never
   in an answer, citation, log row or admin view for a user who can't see
   the document.
8. **Adversarial prompts**: (a) in CI, with the fake LLM, retrieval,
   citations and logs contain only visible chunks whatever the prompt says;
   (b) (quota) a small live run against Gemini on the Phase 3 database:
   answers refuse or use only visible documents, zero canaries.
9. **Versioning** (supersedes chain, latest-only by default, older versions
   by id).
10. **Deletion** (uploader or org admin only; 403 for other editors who can
    see it, 404 for anyone who can't; audit row). Tests include an editor
    who can see but didn't upload (403) and an admin of the same org
    (allowed).
11. **Query plan**: EXPLAIN ANALYZE on the real corpus as documind_app with
    an access context, both retrieval legs, recorded here; any change to
    the indexes waits for a decision.
12. **Phase 3 results and the PR** to main.

## Phase 3 notes

- **Check the query plan under RLS on the real corpus.** Run EXPLAIN
  ANALYZE on both retrieval legs (the HNSW iterative scan and the full-text
  search) as documind_app with an access context set, on the Phase 3
  corpus, and record whether the HNSW and GIN indexes are used and the
  timings. In the Phase 2 tests, with seq scans disabled, the planner chose
  the `org_id` btree plus an exact sort over HNSW when every row was in the
  caller's org; at real scale that plan would be slow.

## Phase 6 notes

- **Q7 and Q12 are false refusals, caused by excerpt truncation in grading.**
  Both answers are in `sample.pdf`, and retrieval found them (Phase 0 run A:
  gold page ranked 1st for Q12, 2nd for Q7). But `grade_node` shows the
  relevance grader only the first 700 characters of each excerpt
  (`_numbered_context(..., max_chars=700)` in `graph/nodes.py`), and both
  answers lie past that cut in their chunk text:
  - Q7 (page 2, 1,008 chars): "out of range for more than 5 minutes" at
    915, "Tier-2 alarm" at 963.
  - Q12 (page 4, 986 chars): the sentence starts at 648, so the grader sees
    "The emergency return vehicle remains docked at all times, seat", but
    "seats all six crew members" starts at 706 and "undock within 4 minutes"
    at 742.
  Every answered question's answer ends before 700 (the latest is Q3 at 660;
  Q6's "recovers 93 percent" is at 636-655). That's why the July Claude run
  and both Gemini passes refused exactly these two. It is not the grader
  model; the README called them correct refusals.
  The fix ("grade full excerpts") lands in Phase 1 after run C, so the A/B/C
  comparison isn't confounded, with a regression test and a before/after
  RAGAS pass. **Done:** 8cbc2e4; both are answered in the after pass (see
  Phase 1, "Grader fix: before/after"). The Phase 6 eval still adds a **false-refusal rate** metric
  (questions with an answer in the document that end in a refusal), and the
  README wording is corrected when the eval is rewritten.
- **Plan for the judge's daily cap.** The judge (`gemini-3.1-flash-lite`)
  is capped at 500 requests a day on the free tier (hit in Phase 1), and a
  14-question pass used about 115 judge calls (Phase 0: 115; Phase 1:
  85–106). The 60+ question set at 3 passes needs about 1,500. Plan for it:
  spread passes across days, rotate judge models (each has its own quota),
  and use deterministic metrics (recall@k, MRR, refusal correctness, ACL
  leakage) wherever possible, keeping the LLM judge for faithfulness.
- **Check: no unjustified excerpt truncation.** Any truncation of excerpt or
  chunk text anywhere in the pipeline (grading, generation, groundedness,
  embedding input, logging) must be justified in a comment and covered by a
  test that places the answer past the cut.
- **Record per-statement faithfulness verdicts in the main harness.** Every
  pass keeps each statement's verdict, not just `rejudge_faithfulness.py`, and
  the report gives each question's min/max across passes. The Q7 re-judge
  (Phase 1) showed that one judge misread of a conditional rule can swing a
  question from 1.0 to 0.0, which a mean alone would hide.

## Phase 7 notes

- **Record embedding calls in `llm_usage` too.** Since b79d2b8 the gateway
  records prompt, output and thinking tokens per LLM call and per role
  (grade, generate, groundedness), but not embeddings. Cost per request has
  to include them: the query embedding on every request, and document
  embeddings at ingestion. Record per call the model, the texts sent, their
  token count (or characters, where the provider bills by character) and
  the computed cost, under its own role (`embed_query`, `embed_documents`).
- **Measure cheaper grading against the false-refusal rate.** Grading full
  excerpts costs more input tokens than the old 700-character cut (see the
  Phase 1 grader-fix results), and more as chunks grow. Cheaper options to
  measure, each against the false-refusal rate from the Phase 6 eval:
  grading only the top-N fused chunks, for example. An option that refuses
  answerable questions again isn't cheaper, it's broken.

## Phase 8 notes

- **Citation tooltips should show the cited sentence.** The tooltip shows the
  first 200 characters of the chunk (`generate_node` in `graph/nodes.py`), not
  the sentence the answer cites. Show that sentence or a window around it,
  with a test using eval Q7, whose cited sentence is near character 940, so
  today's tooltip doesn't show it.

- **The chat's empty state doesn't notice an upload.** After uploading a
  document, the chat still says "Upload a document… then ask anything about
  it." It should switch to a prompt to ask about the uploaded document once
  one exists (seen in the Phase 2 browser walkthrough).

## Phase 9 notes

- **Replace the mounted Firebase key with Workload Identity.** Locally,
  docker-compose bind-mounts the service account key read-only (379059c);
  on Cloud Run the service account attached to the service should
  authenticate with no key file at all.
- **Run the migrate step as a Cloud Run job**, not inside the service: the
  job gets the owner credentials (`MIGRATION_DATABASE_URL`), the service
  only documind_app's (`DATABASE_URL`), each from Secret Manager, and a
  deploy runs the job to completion before rolling out the service. On a
  managed Postgres whose owner isn't a superuser, the migration role needs
  BYPASSRLS for the embedding-model check (`db.session._check_embedding_models`).
