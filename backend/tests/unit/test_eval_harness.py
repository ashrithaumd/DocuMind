"""The eval harness's bookkeeping (no RAGAS, no network): every question
lands in exactly one recorded outcome, a failed question doesn't stop the
run, a daily-quota error does, and errored/not-run questions are counted and
reported — never dropped."""
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
import run_eval  # noqa: E402

PAIRS = [{"question": f"q{i}", "reference": f"r{i}"} for i in range(1, 5)]


def answered(question):
    return httpx.Response(200, json={
        "outcome": "answered", "answer": f"answer to {question}",
        "chunks": [{"text": "ctx"}],
        "timings": {"retrieval_ms": 20.0, "grading_ms": 900.0, "total_ms": 5000.0},
    })


def failed(status, stage, error):
    return httpx.Response(status, json={"detail": "Pipeline failed.", "stage": stage, "error": error})


def client_for(responses: dict):
    """q -> response; unknown questions get an 'answered' reply."""
    def handler(request):
        question = json.loads(request.content)["question"]
        return responses.get(question) or answered(question)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_failed_question_is_recorded_and_the_run_continues():
    client = client_for({"q2": failed(503, "generate", "unavailable")})

    samples, stop = run_eval.run_questions(client, "http://api", PAIRS)

    assert stop is None
    assert [s["outcome"] for s in samples] == ["answered", "error", "answered", "answered"]
    assert samples[1]["stage"] == "generate"
    assert samples[1]["error"] == "unavailable"


def test_daily_quota_stops_the_run_and_marks_the_rest_not_run():
    client = client_for({"q2": failed(503, "generate", "daily_quota")})

    samples, stop = run_eval.run_questions(client, "http://api", PAIRS)

    assert [s["outcome"] for s in samples] == ["answered", "error", "not_run", "not_run"]
    assert "daily quota" in stop and "question 2" in stop


def test_non_json_error_and_transport_failure_still_recorded():
    def handler(request):
        if json.loads(request.content)["question"] == "q1":
            return httpx.Response(500, text="Internal Server Error")
        raise httpx.ReadTimeout("timed out", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    samples, _ = run_eval.run_questions(client, "http://api", PAIRS[:2])

    assert (samples[0]["stage"], samples[0]["error"]) == ("api", "http_500")
    assert (samples[1]["stage"], samples[1]["error"]) == ("api", "ReadTimeout")


def test_summary_counts_every_outcome_and_averages_only_scored():
    samples = [
        {"outcome": "answered", "faithfulness": 1.0, "context_precision": 0.5, "timings": {}},
        {"outcome": "answered", "faithfulness": None, "context_precision": None,
         "judge_error": "boom", "timings": {}},
        {"outcome": "refused_no_context", "timings": {}},
        {"outcome": "error", "stage": "generate", "error": "unavailable", "timings": {}},
        {"outcome": "not_run", "timings": {}},
    ]

    summary = run_eval.summarize(samples, ["daily quota exhausted at question 5"])

    assert summary["questions"] == 5
    assert (summary["answered"], summary["refused"], summary["errored"], summary["not_run"]) == (2, 1, 1, 1)
    assert summary["mean_faithfulness"] == 1.0 and summary["faithfulness_scored"] == 1
    assert summary["mean_context_precision"] == 0.5
    assert summary["stopped"] == "daily quota exhausted at question 5"


def test_reports_list_errored_and_not_run_questions(tmp_path):
    samples = [
        {"user_input": "q1", "outcome": "answered", "faithfulness": 1.0,
         "context_precision": 1.0, "timings": {"retrieval_ms": 20.0}},
        {"user_input": "q2", "outcome": "error", "stage": "generate",
         "error": "unavailable", "timings": {}},
        {"user_input": "q3", "outcome": "not_run", "timings": {}},
    ]
    summary = run_eval.summarize(samples, [])

    run_eval.write_reports(tmp_path, {"document": "sample.pdf"}, "dataset.json", samples,
                           summary, run_eval.latency_table(samples), {})

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert [q["outcome"] for q in report["per_question"]] == ["answered", "error", "not_run"]
    assert report["per_question"][1]["stage"] == "generate"
    assert report["summary"]["errored"] == 1
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "1 errored, 1 not run" in markdown
    assert "generate: unavailable" in markdown


def test_judge_error_classification():
    daily = "429 RESOURCE_EXHAUSTED ... 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'"
    assert run_eval.is_daily_quota(daily)
    assert not run_eval.is_daily_quota("429 RESOURCE_EXHAUSTED GenerateRequestsPerMinutePerProject")
    assert run_eval.is_transient("Error calling model 'x' (UNAVAILABLE): 503 UNAVAILABLE")
    assert run_eval.is_transient("500 INTERNAL. Internal error encountered.")
    assert not run_eval.is_transient("400 INVALID_ARGUMENT: bad schema")
