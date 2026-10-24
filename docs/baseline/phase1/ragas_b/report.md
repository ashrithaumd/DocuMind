# DocuMind RAGAS Evaluation Report

Dataset: `dataset.json` over `sample.pdf` — 14 questions: 12 answered, 2 refused, 0 errored, 0 not run

**Caveat:** The default generator (gemini-3.5-flash-lite) and the judge (gemini-3.1-flash-lite) are both small Gemini Flash-Lite models, so self-preference bias in the RAGAS scores is possible.

| Metric | Mean | Scored |
|---|---|---|
| Faithfulness | **1.0** | 12/12 |
| Context precision | **1.0** | 12/12 |

Latency per stage, all questions that ran (ms). LLM stages on the Gemini free
tier are queue-dominated and not representative; `retrieval_ms` is the baseline.

| Stage | n | p50 | p95 | |
|---|---|---|---|---|
| retrieval_ms | 14 | 46.8 | 83.6 | baseline |
| grading_ms | 14 | 1183.2 | 14571.8 | queue-dominated (free tier), not representative |
| generation_ms | 12 | 681.8 | 15379.6 | queue-dominated (free tier), not representative |
| groundedness_ms | 12 | 827.7 | 14911.3 | queue-dominated (free tier), not representative |
| total_ms | 14 | 2818.7 | 34750.1 | queue-dominated (free tier), not representative |

Requests per model (failed attempts included; full error texts are in report.json).

| Model | Role | Requests | Failed attempts |
|---|---|---|---|
| gemini-3.5-flash-lite | app | 38 | 0 |
| gemini-3.1-flash-lite | judge | 106 | 22 |


| # | Question | Outcome | Faithfulness | Context precision | Notes |
|---|---|---|---|---|---|
| 1 | At what altitude does the Aurora-9 station orbit? | answered | 1.0 | 1.0 |  |
| 2 | How many crop varieties are grown on the station, and across how many greenhouse modules? | answered | 1.0 | 1.0 |  |
| 3 | What is the growing area of Botany Bay A compared to the other modules? | answered | 1.0 | 1.0 |  |
| 4 | What CO2 concentration is maintained in the greenhouse during photoperiod hours? | answered | 1.0 | 1.0 |  |
| 5 | What are the day and night temperature setpoints in the greenhouse modules? | answered | 1.0 | 1.0 |  |
| 6 | How much of the station's water does the reclamation loop recover? | answered | 1.0 | 1.0 |  |
| 7 | What happens if an environmental parameter stays out of range for more than five minutes? | refused_no_context |  |  |  |
| 8 | What roles make up the six-person crew? | answered | 1.0 | 1.0 |  |
| 9 | How often is preventive maintenance performed on the environmental control system, and who performs it? | answered | 1.0 | 1.0 |  |
| 10 | What must the crew do if an ammonia leak alarm sounds? | answered | 1.0 | 1.0 |  |
| 11 | Why does fire suppression in the greenhouse modules use CO2 flooding instead of water mist? | answered | 1.0 | 1.0 |  |
| 12 | How quickly can the emergency return vehicle undock, and how many people does it seat? | refused_no_context |  |  |  |
| 13 | Where are the primary and backup ground stations located? | answered | 1.0 | 1.0 |  |
| 14 | How long is science imagery stored onboard before archival downlink? | answered | 1.0 | 1.0 |  |
