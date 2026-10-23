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

**Why:** the baseline has to be re-run after Phase 1, Phase 6 and Phase 12 with
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
  RAGAS pass. The Phase 6 eval still adds a **false-refusal rate** metric
  (questions with an answer in the document that end in a refusal), and the
  README wording is corrected when the eval is rewritten.
- **Check: no unjustified excerpt truncation.** Any truncation of excerpt or
  chunk text anywhere in the pipeline (grading, generation, groundedness,
  embedding input, logging) must be justified in a comment and covered by a
  test that places the answer past the cut.
