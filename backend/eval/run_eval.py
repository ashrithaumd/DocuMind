"""RAGAS evaluation harness.

Runs the hand-written QA set (dataset.json) against a LIVE DocuMind API — the
same code path users hit, not a parallel re-implementation — then scores the
results with RAGAS:

  - faithfulness:       are the answer's claims supported by the retrieved
                        chunks? (external check on the groundedness gate)
  - context precision:  did retrieval rank the truly relevant chunks at the
                        top? (the metric that would justify adding a reranker)

Usage:
    1. Start the stack (docker compose up, or uvicorn locally).
    2. Upload the sample document once:  see --ingest below.
    3. python eval/run_eval.py [--api http://localhost:8000] [--ingest]

Outputs eval/report.json and eval/report.md.
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

EVAL_DIR = Path(__file__).parent


def call_api(client: httpx.Client, api: str, question: str) -> dict:
    response = client.post(f"{api}/query", json={"question": question}, timeout=120)
    response.raise_for_status()
    return response.json()


# Judge per provider. Overridable with JUDGE_MODEL. Defaults:
#   gemini:    gemini-3.1-flash-lite — free tier, a stable (non-preview) ID,
#              and a different model from the app's generation/grading models,
#              so the judge draws on its own per-model rate-limit bucket
#              instead of starving the app's.
#   anthropic: claude-sonnet-4-6 rather than claude-opus-4-8 for a mechanical
#              reason: RAGAS's LangchainLLMWrapper injects `temperature` into
#              every judge call, and Opus 4.8 rejects sampling parameters with
#              a 400 — Sonnet 4.6 is the strongest Claude that still accepts them.
JUDGE_DEFAULTS = {"gemini": "gemini-3.1-flash-lite", "anthropic": "claude-sonnet-4-6"}

STAGES = ("retrieval_ms", "grading_ms", "generation_ms", "groundedness_ms", "total_ms")


class JudgeCallCounter:
    """LangChain callback counting every judge request per model, plus the
    errors (429 quota details included) — free-tier quotas count requests,
    so retries must be visible, not hidden inside the client."""

    def __new__(cls):
        from langchain_core.callbacks import BaseCallbackHandler

        class _Counter(BaseCallbackHandler):
            def __init__(self):
                self.requests: dict[str, int] = {}
                self.errors: list[str] = []

            def on_chat_model_start(self, serialized, messages, *, invocation_params=None, **kw):
                model = (invocation_params or {}).get("model", "?")
                self.requests[model] = self.requests.get(model, 0) + 1

            def on_llm_error(self, error, **kw):
                self.errors.append(f"{type(error).__name__}: {str(error)[:400]}")
                print(f"  judge error: {self.errors[-1][:200]}", flush=True)

        return _Counter()


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo), 1)


def build_judge():
    from ragas.llms import LangchainLLMWrapper

    provider = os.environ.get("LLM_PROVIDER", "gemini")
    model = os.environ.get("JUDGE_MODEL") or JUDGE_DEFAULTS[provider]
    print(f"RAGAS judge: {provider} / {model}")
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # No temperature: Google recommends keeping Gemini 3 models at their
        # default 1.0 (lower values risk looping/degraded reasoning), and
        # bypass_temperature below stops RAGAS overriding it with 0.01. The
        # resulting judge variance is measured by the 3-pass min/max.
        # max_retries=0: RAGAS's own retry (RunConfig, exponential backoff)
        # re-issues failed judge calls, so every request reaches the counter.
        chat = ChatGoogleGenerativeAI(
            model=model, api_key=os.environ["GEMINI_API_KEY"],
            max_tokens=4096, max_retries=0,
        )
        return LangchainLLMWrapper(chat, bypass_temperature=True)
    else:
        from langchain_anthropic import ChatAnthropic

        chat = ChatAnthropic(model=model, temperature=0, max_tokens=4096)
        return LangchainLLMWrapper(chat)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the RAGAS eval against a live API.")
    parser.add_argument("--api", default="http://localhost:8000", help="Backend base URL")
    parser.add_argument("--dataset", default=str(EVAL_DIR / "dataset.json"))
    parser.add_argument("--api-key", default=None, help="X-API-Key if the gate is enabled")
    parser.add_argument(
        "--out-dir", default=str(EVAL_DIR),
        help="Where report.json / report.md are written (default: eval/)",
    )
    parser.add_argument(
        "--ingest", action="store_true",
        help="Upload eval/sample.pdf to the API before querying",
    )
    args = parser.parse_args()

    load_dotenv()  # the RAGAS judge needs the provider key (see build_judge)

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    client = httpx.Client(headers=headers)

    if args.ingest:
        pdf_path = EVAL_DIR / dataset["document"]
        with pdf_path.open("rb") as fh:
            response = client.post(
                f"{args.api}/upload",
                files={"file": (pdf_path.name, fh, "application/pdf")},
                timeout=300,
            )
        response.raise_for_status()
        print(f"Ingested {pdf_path.name}: {response.json()}")

    # --- run every QA pair through the real pipeline -------------------------
    samples = []
    for i, pair in enumerate(dataset["pairs"], start=1):
        result = call_api(client, args.api, pair["question"])
        print(f"[{i:2d}/{len(dataset['pairs'])}] {result['outcome']:20s} {pair['question'][:60]}")
        samples.append(
            {
                "user_input": pair["question"],
                "reference": pair["reference"],
                "response": result["answer"],
                "retrieved_contexts": [c["text"] for c in result["chunks"]],
                "outcome": result["outcome"],
                "timings": result.get("timings") or {},
            }
        )

    answered = [s for s in samples if s["outcome"] == "answered"]
    refused = len(samples) - len(answered)
    if refused:
        print(f"WARNING: {refused} question(s) were refused; they are excluded from scoring.")
    if not answered:
        print("Nothing to score — did you forget --ingest?")
        return 1

    # --- score with RAGAS -----------------------------------------------------
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference
    from ragas.run_config import RunConfig

    judge = build_judge()
    judge_counter = JudgeCallCounter()
    judge.langchain_llm.callbacks = [judge_counter]
    eval_dataset = EvaluationDataset.from_list(
        [{k: s[k] for k in ("user_input", "reference", "response", "retrieved_contexts")}
         for s in answered]
    )
    result = evaluate(
        dataset=eval_dataset,
        metrics=[Faithfulness(), LLMContextPrecisionWithReference()],
        llm=judge,
        # timeout: long contexts can exceed the 180s default. max_workers:
        # RAGAS fires 16 judge calls at once by default — far above Gemini
        # free-tier RPM, so throttle there (JUDGE_WORKERS overrides).
        run_config=RunConfig(
            timeout=300,
            max_workers=int(os.environ.get("JUDGE_WORKERS")
                            or (2 if os.environ.get("LLM_PROVIDER", "gemini") == "gemini" else 16)),
        ),
    )
    df = result.to_pandas()

    # --- report ---------------------------------------------------------------
    faith_col = "faithfulness"
    precision_col = next(c for c in df.columns if "context_precision" in c)

    rows = []
    for sample, (_, scored) in zip(answered, df.iterrows()):
        rows.append(
            {
                "question": sample["user_input"],
                "outcome": sample["outcome"],
                "faithfulness": round(float(scored[faith_col]), 3),
                "context_precision": round(float(scored[precision_col]), 3),
                "timings": sample["timings"],
            }
        )

    summary = {
        "questions": len(samples),
        "answered": len(answered),
        "refused": refused,
        "mean_faithfulness": round(statistics.mean(r["faithfulness"] for r in rows), 3),
        "mean_context_precision": round(
            statistics.mean(r["context_precision"] for r in rows), 3
        ),
    }

    # Per-stage latency over ALL questions (refusals included: they exercise
    # retrieval + grading). On the Gemini free tier the LLM stages are
    # dominated by server-side queueing and are NOT representative of model
    # latency; retrieval_ms (local, no LLM) is the baseline number.
    latency = {}
    for stage in STAGES:
        values = [s["timings"][stage] for s in samples if stage in s["timings"]]
        if values:
            latency[stage] = {
                "n": len(values),
                "p50": percentile(values, 50),
                "p95": percentile(values, 95),
                "label": "baseline" if stage == "retrieval_ms"
                else "queue-dominated (free tier), not representative",
            }
    judge_calls = {"requests_per_model": judge_counter.requests, "errors": judge_counter.errors}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "latency_ms": latency,
                "judge_calls": judge_calls,
                "per_question": [
                    {"question": s["user_input"], "outcome": s["outcome"], "timings": s["timings"]}
                    for s in samples
                ],
                "results": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# DocuMind RAGAS Evaluation Report",
        "",
        f"Dataset: `{Path(args.dataset).name}` over `{dataset['document']}` "
        f"({len(samples)} questions, {len(answered)} answered, {refused} refused)",
        "",
        f"| Metric | Mean |",
        f"|---|---|",
        f"| Faithfulness | **{summary['mean_faithfulness']}** |",
        f"| Context precision | **{summary['mean_context_precision']}** |",
        "",
        "Latency per stage, all questions (ms). LLM stages on the Gemini free tier",
        "are queue-dominated and not representative; `retrieval_ms` is the baseline.",
        "",
        "| Stage | n | p50 | p95 | |",
        "|---|---|---|---|---|",
        *[f"| {stage} | {v['n']} | {v['p50']} | {v['p95']} | {v['label']} |"
          for stage, v in latency.items()],
        "",
        "| # | Question | Faithfulness | Context precision |",
        "|---|---|---|---|",
    ]
    for i, row in enumerate(rows, start=1):
        lines.append(
            f"| {i} | {row['question']} | {row['faithfulness']} | {row['context_precision']} |"
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nSummary:", json.dumps(summary, indent=2))
    print("Latency:", json.dumps(latency, indent=2))
    print("Judge calls:", json.dumps(judge_calls, indent=2))
    print(f"Wrote {out_dir / 'report.json'} and {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
