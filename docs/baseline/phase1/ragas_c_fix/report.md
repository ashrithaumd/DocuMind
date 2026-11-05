# DocuMind RAGAS Evaluation Report

Dataset: `dataset.json` over `sample.pdf` — 14 questions: 14 answered, 0 refused, 0 errored, 0 not run

Code: `8874fb7ec7799485a111a3b5bc3f262ed66926bf` **with uncommitted changes:** `docs/baseline/phase1/ragas_c_fix/results.jsonl`

**Caveat:** The default generator (gemini-3.5-flash-lite) and the judge (gemini-3.1-flash-lite) are both small Gemini Flash-Lite models, so self-preference bias in the RAGAS scores is possible.

| Metric | Mean | Scored |
|---|---|---|
| Faithfulness | **0.964** | 14/14 |
| Context precision | **0.929** | 14/14 |

Latency per stage, all questions that ran (ms). LLM stages on the Gemini free
tier are queue-dominated and not representative; `retrieval_ms` is the baseline.

| Stage | n | p50 | p95 | |
|---|---|---|---|---|
| retrieval_ms | 14 | 598.7 | 913.4 | baseline |
| grading_ms | 14 | 991.9 | 2575.6 | queue-dominated (free tier), not representative |
| generation_ms | 14 | 744.5 | 1315.8 | queue-dominated (free tier), not representative |
| groundedness_ms | 14 | 874.1 | 1112.2 | queue-dominated (free tier), not representative |
| total_ms | 14 | 3274.5 | 5045.0 | queue-dominated (free tier), not representative |

Requests per model (failed attempts included; full error texts are in report.json).

| Model | Role | Requests | Failed attempts |
|---|---|---|---|
| gemini-embedding-2 | app | 14 | 0 |
| gemini-3.5-flash-lite | app | 42 | 0 |
| gemini-3.1-flash-lite | judge | 100 | 2 |

App tokens per role, measured from live calls (provider usage metadata).

| Role | Calls | Prompt tokens | Mean prompt/call | Output tokens | Thinking tokens |
|---|---|---|---|---|---|
| grade | 14 | 17207 | 1229.1 | 82 | 0 |
| generate | 14 | 17557 | 1254.1 | 425 | 0 |
| groundedness | 14 | 17561 | 1254.4 | 262 | 0 |

- Quota hit: `GenerateRequestsPerMinutePerProjectPerModel-FreeTier (limit 15)` ×2

| # | Question | Outcome | Faithfulness | Context precision | Notes |
|---|---|---|---|---|---|
| 1 | At what altitude does the Aurora-9 station orbit? | answered | 1.0 | 1.0 |  |
| 2 | How many crop varieties are grown on the station, and across how many greenhouse modules? | answered | 1.0 | 1.0 |  |
| 3 | What is the growing area of Botany Bay A compared to the other modules? | answered | 1.0 | 0.5 |  |
| 4 | What CO2 concentration is maintained in the greenhouse during photoperiod hours? | answered | 1.0 | 1.0 |  |
| 5 | What are the day and night temperature setpoints in the greenhouse modules? | answered | 1.0 | 1.0 |  |
| 6 | How much of the station's water does the reclamation loop recover? | answered | 1.0 | 1.0 |  |
| 7 | What happens if an environmental parameter stays out of range for more than five minutes? | answered | 0.5 | 1.0 |  |
| 8 | What roles make up the six-person crew? | answered | 1.0 | 1.0 |  |
| 9 | How often is preventive maintenance performed on the environmental control system, and who performs it? | answered | 1.0 | 0.5 |  |
| 10 | What must the crew do if an ammonia leak alarm sounds? | answered | 1.0 | 1.0 |  |
| 11 | Why does fire suppression in the greenhouse modules use CO2 flooding instead of water mist? | answered | 1.0 | 1.0 |  |
| 12 | How quickly can the emergency return vehicle undock, and how many people does it seat? | answered | 1.0 | 1.0 |  |
| 13 | Where are the primary and backup ground stations located? | answered | 1.0 | 1.0 |  |
| 14 | How long is science imagery stored onboard before archival downlink? | answered | 1.0 | 1.0 |  |

## Notes

- Two code SHAs, no code change between them. Questions 1-6 and 8-14 ran on 9046638 (clean); Q7's final record ran on 8874fb7. `git diff --stat 9046638 8874fb7 -- backend/` is empty: between them only docs, eval result files, .env.example and the CI workflow changed.
- The dirty file recorded above, docs/baseline/phase1/ragas_c_fix/results.jsonl, is this run's own output: --resume appends to it before the report records the git state. No other tracked file was modified during the run.
- Q7 has three records in results.jsonl (the last one counts): the first pass (judge timeouts, unscored); a --resume on 9046638 (faithfulness 1.0, context precision stopped by the judge's daily quota); and a --resume on 8874fb7 after the quota reset (faithfulness 0.5, context precision 1.0). Each resume re-queries the question and re-judges both metrics. The 1.0 and 0.5 faithfulness scores are for essentially the same answer, a near-verbatim restatement of the source sentence: judge variance, not a pipeline change.
- Re-judging Q7's stored answer 5 times on f727f1a, with no pipeline call, gave faithfulness 1.0, 1.0, 0.0, 1.0, 1.0 (q7_rejudge.jsonl keeps each statement's verdict), which supports the judge-variance reading. The 0.5 record kept no verdicts, so its cause isn't known.
