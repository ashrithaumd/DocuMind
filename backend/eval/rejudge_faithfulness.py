"""Re-judge one stored answer's faithfulness, keeping every statement's verdict.

run_eval.py records only the final score, so a low score can't say which claim
the judge rejected or why. This runs the judge's two steps from
ragas's Faithfulness._ascore (extract statements, then judge each against the
contexts) on the answer and contexts stored in results.jsonl, with no pipeline
call, and appends one record per pass before starting the next.

Usage:
    python eval/rejudge_faithfulness.py RESULTS_JSONL --question "five minutes" --out FILE.jsonl
"""
import argparse
import asyncio
import importlib.metadata
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

import run_eval

# judge_once calls private Faithfulness methods, verified only against this version.
TESTED_RAGAS = "0.4.3"


async def judge_once(metric, row: dict) -> dict:
    extracted = await metric._create_statements(row, None)
    if not extracted.statements:
        return {"statements": [], "verdicts": [], "score": None}
    verdicts = await metric._create_verdicts(row, extracted.statements, None)
    return {
        "statements": extracted.statements,
        "verdicts": [v.model_dump() for v in verdicts.statements],
        "score": metric._compute_score(verdicts),
    }


def rejudge(sample: dict, metric, counter, passes: int, out: Path, meta: dict) -> list[dict]:
    """Run the passes, persisting each one. A failed pass is recorded and the
    next one runs; a daily-quota error stops the run."""
    row = {k: sample[k] for k in ("user_input", "response", "retrieved_contexts")}
    records = []
    for n in range(1, passes + 1):
        requests_before = sum(counter.requests.values())
        errors_before = len(counter.errors)
        record = {"pass": n, **meta, "question": row["user_input"], "response": row["response"]}
        stop = False
        try:
            record.update(asyncio.run(judge_once(metric, row)))
        except Exception as exc:
            record.update(score=None, error=f"{type(exc).__name__}: {exc}")
            stop = run_eval.is_daily_quota(str(exc))
        record["judge_requests"] = sum(counter.requests.values()) - requests_before
        record["judge_errors"] = counter.errors[errors_before:]
        run_eval.append_result(out, record)
        records.append(record)
        print(f"pass {n}: score={record['score']} requests={record['judge_requests']}", flush=True)
        for verdict in record.get("verdicts", []):
            print(f"  [{verdict['verdict']}] {verdict['statement']}\n      {verdict['reason']}", flush=True)
        if stop:
            print("Stopped: judge daily quota exhausted.", flush=True)
            break
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-judge one answer's faithfulness.")
    parser.add_argument("results", help="results.jsonl holding the answer (its last record is used)")
    parser.add_argument("--question", required=True, help="Text that identifies exactly one question")
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    installed = importlib.metadata.version("ragas")
    if installed != TESTED_RAGAS:
        print(f"ragas {installed} installed, but this script calls private methods verified "
              f"only on {TESTED_RAGAS}: re-check judge_once, then update TESTED_RAGAS.")
        return 2

    out = Path(args.out)
    if out.exists():
        print(f"{out} already exists: move it away to start over.")
        return 2
    matches = [r for q, r in run_eval.load_results(Path(args.results)).items() if args.question in q]
    if len(matches) != 1:
        print(f"--question matched {len(matches)} questions, expected 1.")
        return 2

    load_dotenv()  # the judge needs the provider key (see build_judge)
    from ragas.metrics import Faithfulness

    judge = run_eval.build_judge()
    counter = run_eval.JudgeCallCounter()
    judge.langchain_llm.callbacks = [counter]
    metric = Faithfulness(llm=judge)
    metric.init(run_eval.judge_run_config())  # the harness's retries and timeout

    provider = os.environ.get("LLM_PROVIDER", "gemini")
    meta = {"code": run_eval.code_version(), "source": args.results,
            "judge_model": os.environ.get("JUDGE_MODEL") or run_eval.JUDGE_DEFAULTS[provider]}
    records = rejudge(matches[0], metric, counter, args.passes, out, meta)
    print(json.dumps({"scores": [r["score"] for r in records], "out": str(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
