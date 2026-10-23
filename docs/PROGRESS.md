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
  answers in the document, so these are false refusals by the relevance
  grader: with a 5-chunk index, retrieval handed it the whole document.
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

- **Q7 and Q12 are false refusals.** Both answers are in `sample.pdf`, but the
  relevance grader (`gemini-3.5-flash-lite`) rejected the retrieved chunks
  (`refused_no_context`) in both Phase 0 passes. The July 2026 Claude run
  refused the same two questions, and the README described those as correct
  refusals ("refused rather than guessed, consistent with the refusal-first
  design"). The Phase 6 eval adds a **false-refusal rate** metric (questions
  with an answer in the document that end in a refusal), and the README
  wording is corrected when the eval is rewritten. No fix now.
