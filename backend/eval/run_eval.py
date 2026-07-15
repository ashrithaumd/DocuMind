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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the RAGAS eval against a live API.")
    parser.add_argument("--api", default="http://localhost:8000", help="Backend base URL")
    parser.add_argument("--dataset", default=str(EVAL_DIR / "dataset.json"))
    parser.add_argument("--api-key", default=None, help="X-API-Key if the gate is enabled")
    parser.add_argument(
        "--ingest", action="store_true",
        help="Upload eval/sample.pdf to the API before querying",
    )
    args = parser.parse_args()

    load_dotenv()  # the RAGAS judge model needs ANTHROPIC_API_KEY

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
    from langchain_anthropic import ChatAnthropic
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference
    from ragas.run_config import RunConfig

    # Judge on Claude. claude-sonnet-4-6 rather than claude-opus-4-8 for a
    # mechanical reason: RAGAS's LangchainLLMWrapper injects `temperature`
    # into every judge call, and Opus 4.8 rejects sampling parameters with a
    # 400 — Sonnet 4.6 is the strongest Claude that still accepts them.
    judge = LangchainLLMWrapper(
        ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=4096)
    )
    eval_dataset = EvaluationDataset.from_list(
        [{k: s[k] for k in ("user_input", "reference", "response", "retrieved_contexts")}
         for s in answered]
    )
    result = evaluate(
        dataset=eval_dataset,
        metrics=[Faithfulness(), LLMContextPrecisionWithReference()],
        llm=judge,
        run_config=RunConfig(timeout=300),  # long contexts can exceed the 180s default
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

    (EVAL_DIR / "report.json").write_text(
        json.dumps({"summary": summary, "results": rows}, indent=2), encoding="utf-8"
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
        "| # | Question | Faithfulness | Context precision |",
        "|---|---|---|---|",
    ]
    for i, row in enumerate(rows, start=1):
        lines.append(
            f"| {i} | {row['question']} | {row['faithfulness']} | {row['context_precision']} |"
        )
    (EVAL_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nSummary:", json.dumps(summary, indent=2))
    print(f"Wrote {EVAL_DIR / 'report.json'} and {EVAL_DIR / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
