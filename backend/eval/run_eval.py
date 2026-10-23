"""RAGAS evaluation harness.

Runs the hand-written QA set (dataset.json) against a LIVE DocuMind API — the
same code path users hit, not a parallel re-implementation — then scores the
results with RAGAS:

  - faithfulness:       are the answer's claims supported by the retrieved
                        chunks? (external check on the groundedness gate)
  - context precision:  did retrieval rank the truly relevant chunks at the
                        top? (the metric that would justify adding a reranker)

Every question ends in exactly one recorded outcome: the pipeline's own
(answered / refused_*), "error" (with the failing stage and error kind), or
"not_run" (the run stopped first). Nothing is dropped silently. A daily-quota
error from any model stops the run; everything already scored is kept.

Durability: each question is queried and, if answered, scored straight away;
its record is then appended to results.jsonl in the output directory before
the next question starts. A killed run loses at most the question in flight.
--resume reloads results.jsonl and skips questions already finished (refused,
or answered and fully scored); errored, not-run and partly scored questions
are run again. The last record per question wins.

Usage:
    1. Start the stack (docker compose up, or uvicorn locally) with
       EXPOSE_LLM_DIAGNOSTICS=true, so /query returns llm_usage and full
       error text for the report. It is off by default for normal clients.
    2. Upload the sample document once:  see --ingest below.
    3. python eval/run_eval.py [--api http://localhost:8000] [--ingest] [--out-dir DIR] [--resume]

Outputs results.jsonl, report.json and report.md (default: eval/).
"""
import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

EVAL_DIR = Path(__file__).parent

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

RESULTS_FILE = "results.jsonl"
STAGES = ("retrieval_ms", "grading_ms", "generation_ms", "groundedness_ms", "total_ms")
METRICS = ("faithfulness", "context_precision")


# --- error classification ------------------------------------------------------


class DailyQuotaExhausted(Exception):
    """A per-day quota is spent: retrying before the reset is pointless."""


class TransientJudgeError(Exception):
    """A judge error worth retrying (rate limit, 500/503)."""


def is_daily_quota(text: str) -> bool:
    # Gemini's quota id, e.g. GenerateRequestsPerDayPerProjectPerModel-FreeTier.
    return "PerDay" in text


def is_transient(text: str) -> bool:
    return bool(re.search(r"\b(429|500|503|529)\b|RESOURCE_EXHAUSTED|UNAVAILABLE|INTERNAL|overloaded", text))


def quota_hits(text: str) -> list[str]:
    """'<quotaId> (limit N)' for each quota violation in a Gemini error text."""
    return [f"{quota_id} (limit {value})" for quota_id, value in
            re.findall(r"'quotaId': '([^']+)'.*?'quotaValue': '([^']+)'", text)]


# --- querying the live pipeline ------------------------------------------------


def call_api(client: httpx.Client, api: str, question: str) -> dict:
    response = client.post(f"{api}/query", json={"question": question}, timeout=600)
    response.raise_for_status()
    return response.json()


def _api_error(exc: httpx.HTTPError) -> dict:
    """stage, error kind, full error text and the request's LLM usage for a
    failed /query call."""
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            body = exc.response.json()
        except ValueError:
            body = {}
        return {
            "stage": body.get("stage", "api"),
            "error": body.get("error", f"http_{exc.response.status_code}"),
            "error_message": body.get("message") or exc.response.text,
            "llm_usage": body.get("llm_usage") or {},
        }
    # timeout / connection error
    return {"stage": "api", "error": type(exc).__name__,
            "error_message": f"{type(exc).__name__}: {exc}", "llm_usage": {}}


def is_finished(sample: dict) -> bool:
    """Nothing left to spend quota on: refused, or answered and fully scored."""
    if sample["outcome"].startswith("refused"):
        return True
    return sample["outcome"] == "answered" and all(sample.get(m) is not None for m in METRICS)


def load_results(path: Path) -> dict[str, dict]:
    """question -> its latest record in results.jsonl (later lines win). A
    torn last line from a killed run is skipped."""
    records: dict[str, dict] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        records[record["user_input"]] = record
    return records


def append_result(path: Path, sample: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(sample) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def run_questions(client: httpx.Client, api: str, pairs: list[dict], score=None,
                  record=None, done: dict[str, dict] | None = None) -> tuple[list[dict], str | None]:
    """Every pair through the live pipeline. Returns (samples, stop_reason):
    one sample per pair, always — a failed question is recorded as "error"
    and the run continues; after a daily-quota error the rest are "not_run".

    score(sample) scores an answered sample in place and returns a stop
    reason or None; record(sample) persists a finished question before the
    next one starts; `done` maps questions finished by an earlier run, which
    are reused rather than re-queried."""
    samples: list[dict] = []
    stop_reason = None
    warned_no_diagnostics = False
    done = done or {}
    for i, pair in enumerate(pairs, start=1):
        if pair["question"] in done:
            sample = done[pair["question"]]
            print(f"[{i:2d}/{len(pairs)}] {sample['outcome']:20s} {pair['question'][:60]} (resumed)",
                  flush=True)
            samples.append(sample)
            continue
        sample = {"user_input": pair["question"], "reference": pair["reference"], "timings": {}}
        if stop_reason:
            sample["outcome"] = "not_run"
        else:
            try:
                result = call_api(client, api, pair["question"])
            except httpx.HTTPError as exc:
                sample.update(outcome="error", **_api_error(exc))
                if sample["error"] == "daily_quota":
                    stop_reason = f"daily quota exhausted at question {i} (stage: {sample['stage']})"
            else:
                if "llm_usage" not in result and not warned_no_diagnostics:
                    warned_no_diagnostics = True
                    print("WARNING: /query returned no llm_usage. Restart the backend with "
                          "EXPOSE_LLM_DIAGNOSTICS=true, or the report's app call counts "
                          "and quota limits will be empty.", flush=True)
                sample.update(
                    outcome=result["outcome"],
                    response=result["answer"],
                    retrieved_contexts=[c["text"] for c in result["chunks"]],
                    timings=result.get("timings") or {},
                    llm_usage=result.get("llm_usage") or {},
                )
        detail = f" [{sample['stage']}: {sample['error']}]" if sample["outcome"] == "error" else ""
        print(f"[{i:2d}/{len(pairs)}] {sample['outcome']:20s} {pair['question'][:60]}{detail}", flush=True)
        if sample["outcome"] == "answered" and score:
            judge_stop = score(sample)
            if judge_stop and not stop_reason:
                stop_reason = f"{judge_stop} at question {i}"
        if record and sample["outcome"] != "not_run":
            record(sample)
        samples.append(sample)
    return samples, stop_reason


# --- judge ---------------------------------------------------------------------------


class JudgeCallCounter:
    """LangChain callback counting every judge request per model, plus the
    errors — free-tier quotas count requests, so retries must be visible, not
    hidden inside the client."""

    def __new__(cls):
        from langchain_core.callbacks import BaseCallbackHandler

        class _Counter(BaseCallbackHandler):
            def __init__(self):
                self.requests: dict[str, int] = {}
                self.errors: list[str] = []
                self.daily_quota_hit = False

            def on_chat_model_start(self, serialized, messages, *, invocation_params=None, **kw):
                model = (invocation_params or {}).get("model", "?")
                self.requests[model] = self.requests.get(model, 0) + 1

            def on_llm_error(self, error, **kw):
                # Full text, never truncated: it carries the quota id and limit.
                self.errors.append(f"{type(error).__name__}: {error}")
                hits = quota_hits(self.errors[-1])
                print(f"  judge error: {', '.join(hits) or self.errors[-1][:200]}", flush=True)
                if isinstance(error, DailyQuotaExhausted):
                    self.daily_quota_hit = True

        return _Counter()


def build_judge():
    from ragas.llms import LangchainLLMWrapper

    provider = os.environ.get("LLM_PROVIDER", "gemini")
    model = os.environ.get("JUDGE_MODEL") or JUDGE_DEFAULTS[provider]
    print(f"RAGAS judge: {provider} / {model}")
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        class QuotaAwareGemini(ChatGoogleGenerativeAI):
            """Re-raises judge errors as transient (RAGAS retries) or
            daily-quota (never retried, stops scoring)."""

            async def _agenerate(self, *args, **kwargs):
                try:
                    return await super()._agenerate(*args, **kwargs)
                except Exception as exc:
                    if is_daily_quota(str(exc)):
                        raise DailyQuotaExhausted(str(exc)) from exc
                    if is_transient(str(exc)):
                        raise TransientJudgeError(str(exc)) from exc
                    raise

        # No temperature: Google recommends keeping Gemini 3 models at their
        # default 1.0 (lower values risk looping/degraded reasoning), and
        # bypass_temperature below stops RAGAS overriding it with 0.01. The
        # resulting judge variance is measured by the 3-pass min/max.
        # max_retries=0: RAGAS's own retry (RunConfig, exponential backoff)
        # re-issues transient failures, so every request reaches the counter.
        chat = QuotaAwareGemini(
            model=model, api_key=os.environ["GEMINI_API_KEY"],
            max_tokens=4096, max_retries=0,
        )
        return LangchainLLMWrapper(chat, bypass_temperature=True)

    from langchain_anthropic import ChatAnthropic

    chat = ChatAnthropic(model=model, temperature=0, max_tokens=4096)
    return LangchainLLMWrapper(chat)


def judge_run_config():
    from ragas.run_config import RunConfig

    gemini = os.environ.get("LLM_PROVIDER", "gemini") == "gemini"
    return RunConfig(
        timeout=300,  # long contexts can exceed the 180s default
        # Only transient errors are retried; a daily quota is not.
        exception_types=(TransientJudgeError,) if gemini else (Exception,),
        # RAGAS fires 16 judge calls at once by default — far above Gemini
        # free-tier RPM, so throttle there (JUDGE_WORKERS overrides).
        max_workers=int(os.environ.get("JUDGE_WORKERS") or (2 if gemini else 16)),
    )


def score_sample(sample: dict, judge, counter, run_config) -> str | None:
    """Score one answered sample in place. A metric the judge failed on is
    None with the reason in sample["judge_error"]; every failed judge call's
    full text is kept in sample["judge_errors"], and the judge requests this
    sample cost in sample["judge_requests"] so a resumed run can still total
    them. Returns a stop reason on a judge daily quota."""
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference

    if counter.daily_quota_hit:
        sample.update(faithfulness=None, context_precision=None,
                      judge_error="not scored: judge daily quota exhausted")
        return "judge daily quota exhausted"
    dataset = EvaluationDataset.from_list(
        [{k: sample[k] for k in ("user_input", "reference", "response", "retrieved_contexts")}]
    )
    errors_before = len(counter.errors)
    requests_before = dict(counter.requests)
    result = evaluate(dataset=dataset, metrics=[Faithfulness(), LLMContextPrecisionWithReference()],
                      llm=judge, run_config=run_config, show_progress=False, raise_exceptions=False)
    row = result.to_pandas().iloc[0]
    precision_col = next(c for c in row.index if "context_precision" in c)
    for name, col in (("faithfulness", "faithfulness"), ("context_precision", precision_col)):
        value = float(row[col])
        sample[name] = None if math.isnan(value) else round(value, 3)
    sample.pop("judge_error", None)
    sample["judge_errors"] = counter.errors[errors_before:]
    if sample["faithfulness"] is None or sample["context_precision"] is None:
        last = sample["judge_errors"][-1] if sample["judge_errors"] else "judge returned no score"
        sample["judge_error"] = ", ".join(quota_hits(last)) or last.split("\n")[0][:200]
    sample["judge_requests"] = {
        model: n - requests_before.get(model, 0) for model, n in counter.requests.items()
        if n - requests_before.get(model, 0)
    }
    print(f"  scored: faithfulness={sample['faithfulness']} "
          f"context_precision={sample['context_precision']}", flush=True)
    return "judge daily quota exhausted" if counter.daily_quota_hit else None


def total_judge_requests(samples: list[dict]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for sample in samples:
        for model, n in sample.get("judge_requests", {}).items():
            totals[model] = totals.get(model, 0) + n
    return totals


def call_accounting(samples: list[dict]) -> dict:
    """Requests and failed attempts per model (app via /query's llm_usage,
    judge via the callback counter), and every distinct quota hit with how
    often it was seen."""
    app_requests: dict[str, int] = {}
    app_failures: dict[str, int] = {}
    tokens: dict[str, dict] = {}
    judge_failures = 0
    quotas: dict[str, int] = {}
    for sample in samples:
        usage = sample.get("llm_usage") or {}
        for model, n in usage.get("attempts_per_model", {}).items():
            app_requests[model] = app_requests.get(model, 0) + n
        for role, counts in usage.get("tokens_per_role", {}).items():
            totals = tokens.setdefault(role, dict.fromkeys(counts, 0))
            for field, n in counts.items():
                totals[field] = totals.get(field, 0) + n
        texts = []
        for failure in usage.get("failures", []):
            app_failures[failure["model"]] = app_failures.get(failure["model"], 0) + 1
            texts.append(failure["error"])
        judge_failures += len(sample.get("judge_errors", []))
        texts += sample.get("judge_errors", [])
        for hit in (h for text in texts for h in quota_hits(text)):
            quotas[hit] = quotas.get(hit, 0) + 1
    return {
        "app_requests_per_model": app_requests,
        "app_failed_attempts_per_model": app_failures,
        # Live counts from the provider's usage metadata (successful calls).
        "app_tokens_per_role": {
            role: {**t, "mean_prompt_tokens": round(t["prompt_tokens"] / t["calls"], 1)}
            for role, t in tokens.items() if t.get("calls")
        },
        "judge_requests_per_model": total_judge_requests(samples),
        "judge_failed_attempts": judge_failures,
        "quota_hits": quotas,
    }


# --- reporting -----------------------------------------------------------------------


def code_version() -> dict:
    """The commit this eval ran from and any tracked files changed since it
    (untracked eval output doesn't count). It describes the backend only if
    the backend was started from this same tree; git_sha is None outside a
    git checkout."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=EVAL_DIR, capture_output=True,
                              text=True, check=True).stdout
    try:
        sha = git("rev-parse", "HEAD").strip()
        changed = git("status", "--porcelain", "--untracked-files=no").splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": None, "dirty_files": []}
    return {"git_sha": sha, "dirty_files": [line[3:] for line in changed]}


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo), 1)


def summarize(samples: list[dict], stop_reasons: list[str]) -> dict:
    """Counts by outcome (every question lands in exactly one bucket) and
    metric means over the samples the judge actually scored."""
    outcomes = [s["outcome"] for s in samples]
    summary = {
        "questions": len(samples),
        "answered": outcomes.count("answered"),
        "refused": sum(o.startswith("refused") for o in outcomes),
        "errored": outcomes.count("error"),
        "not_run": outcomes.count("not_run"),
        "stopped": "; ".join(stop_reasons) or None,
    }
    for metric in METRICS:
        scored = [s[metric] for s in samples if s.get(metric) is not None]
        summary[f"{metric}_scored"] = len(scored)
        summary[f"mean_{metric}"] = round(statistics.mean(scored), 3) if scored else None
    return summary


def latency_table(samples: list[dict]) -> dict:
    # Over every question that ran (refusals included: they exercise
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
    return latency


def write_reports(out_dir: Path, dataset: dict, dataset_name: str, samples: list[dict],
                  summary: dict, latency: dict, calls: dict, caveats: list[str] = (),
                  code: dict | None = None) -> None:
    per_question = [
        {
            "question": s["user_input"],
            "outcome": s["outcome"],
            **{k: s[k] for k in ("stage", "error", "error_message", *METRICS, "judge_error",
                                 "judge_errors", "judge_requests", "llm_usage") if k in s},
            "timings": s["timings"],
        }
        for s in samples
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps({"code": code, "summary": summary, "latency_ms": latency, "calls": calls,
                    "caveats": list(caveats), "per_question": per_question}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# DocuMind RAGAS Evaluation Report",
        "",
        f"Dataset: `{dataset_name}` over `{dataset['document']}` — {summary['questions']} questions: "
        f"{summary['answered']} answered, {summary['refused']} refused, "
        f"{summary['errored']} errored, {summary['not_run']} not run",
    ]
    if code and code.get("git_sha"):
        dirty = code["dirty_files"]
        lines += ["", f"Code: `{code['git_sha']}`" + (
            f" **with uncommitted changes:** {', '.join(f'`{f}`' for f in dirty)}"
            if dirty else " (clean)")]
    if summary["stopped"]:
        lines += ["", f"**Run stopped early:** {summary['stopped']}"]
    for caveat in caveats:
        lines += ["", f"**Caveat:** {caveat}"]
    lines += [
        "",
        "| Metric | Mean | Scored |",
        "|---|---|---|",
        *[f"| {m.replace('_', ' ').capitalize()} | **{summary[f'mean_{m}']}** | "
          f"{summary[f'{m}_scored']}/{summary['answered']} |" for m in METRICS],
        "",
        "Latency per stage, all questions that ran (ms). LLM stages on the Gemini free",
        "tier are queue-dominated and not representative; `retrieval_ms` is the baseline.",
        "",
        "| Stage | n | p50 | p95 | |",
        "|---|---|---|---|---|",
        *[f"| {stage} | {v['n']} | {v['p50']} | {v['p95']} | {v['label']} |"
          for stage, v in latency.items()],
        "",
        "Requests per model (failed attempts included; full error texts are in report.json).",
        "",
        "| Model | Role | Requests | Failed attempts |",
        "|---|---|---|---|",
        *[f"| {model} | app | {n} | {calls.get('app_failed_attempts_per_model', {}).get(model, 0)} |"
          for model, n in calls.get("app_requests_per_model", {}).items()],
        *[f"| {model} | judge | {n} | {calls.get('judge_failed_attempts', 0)} |"
          for model, n in calls.get("judge_requests_per_model", {}).items()],
        "",
        *([
            "App tokens per role, measured from live calls (provider usage metadata).",
            "",
            "| Role | Calls | Prompt tokens | Mean prompt/call | Output tokens | Thinking tokens |",
            "|---|---|---|---|---|---|",
            *[f"| {role} | {t['calls']} | {t['prompt_tokens']} | {t['mean_prompt_tokens']} | "
              f"{t['output_tokens']} | {t['thinking_tokens']} |"
              for role, t in calls["app_tokens_per_role"].items()],
            "",
        ] if calls.get("app_tokens_per_role") else []),
        *[f"- Quota hit: `{hit}` ×{n}" for hit, n in calls.get("quota_hits", {}).items()],
        "",
        "| # | Question | Outcome | Faithfulness | Context precision | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for i, q in enumerate(per_question, start=1):
        note = (f"{q['stage']}: {q['error']}" if q["outcome"] == "error"
                else q.get("judge_error", ""))
        lines.append(f"| {i} | {q['question']} | {q['outcome']} | {q.get('faithfulness', '')} | "
                     f"{q.get('context_precision', '')} | {note} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- main ---------------------------------------------------------------------------


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
    parser.add_argument(
        "--resume", action="store_true",
        help=f"Continue from OUT_DIR/{RESULTS_FILE}, skipping finished questions",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / RESULTS_FILE
    if results_path.exists() and not args.resume:
        print(f"{results_path} already exists. Pass --resume to continue it, "
              "or move it away to start over.")
        return 2
    done = {q: r for q, r in load_results(results_path).items() if is_finished(r)}
    if done:
        print(f"Resuming: {len(done)} question(s) already finished in {results_path}")

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

    judging: dict = {}

    def score(sample: dict) -> str | None:
        if not judging:  # built on first use: a run with no answers needs no judge
            judging["judge"] = build_judge()
            judging["counter"] = JudgeCallCounter()
            judging["judge"].langchain_llm.callbacks = [judging["counter"]]
            judging["run_config"] = judge_run_config()
        return score_sample(sample, judging["judge"], judging["counter"], judging["run_config"])

    samples, stop = run_questions(client, args.api, dataset["pairs"], score=score,
                                  record=lambda s: append_result(results_path, s), done=done)
    stop_reasons = [stop] if stop else []

    answered = [s for s in samples if s["outcome"] == "answered"]
    if not answered:
        print("Nothing answered, nothing to score — did you forget --ingest?")
    calls = call_accounting(samples)

    summary = summarize(samples, stop_reasons)
    latency = latency_table(samples)
    caveats = []
    if os.environ.get("LLM_PROVIDER", "gemini") == "gemini":
        judge_model = os.environ.get("JUDGE_MODEL") or JUDGE_DEFAULTS["gemini"]
        caveats.append(
            f"The default generator (gemini-3.5-flash-lite) and the judge ({judge_model}) "
            "are both small Gemini Flash-Lite models, so self-preference bias in the "
            "RAGAS scores is possible."
        )
    write_reports(out_dir, dataset, Path(args.dataset).name, samples,
                  summary, latency, calls, caveats, code_version())

    print("\nSummary:", json.dumps(summary, indent=2))
    print("Latency:", json.dumps(latency, indent=2))
    print("Calls:", json.dumps(calls, indent=2))
    print(f"Wrote {results_path}, {out_dir / 'report.json'} and {out_dir / 'report.md'}")
    return 0 if answered else 1


if __name__ == "__main__":
    sys.exit(main())
