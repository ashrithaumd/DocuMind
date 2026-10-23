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


def test_each_question_is_scored_and_persisted_before_the_next_runs(tmp_path):
    results = tmp_path / run_eval.RESULTS_FILE
    events = []

    def score(sample):
        events.append(("score", sample["user_input"]))
        sample.update(faithfulness=1.0, context_precision=1.0)

    def record(sample):
        events.append(("record", sample["user_input"]))
        run_eval.append_result(results, sample)

    client = client_for({"q2": failed(503, "generate", "unavailable")})
    run_eval.run_questions(client, "http://api", PAIRS, score=score, record=record)

    assert events == [("score", "q1"), ("record", "q1"), ("record", "q2"),
                      ("score", "q3"), ("record", "q3"), ("score", "q4"), ("record", "q4")]
    saved = run_eval.load_results(results)
    assert saved["q1"]["faithfulness"] == 1.0
    assert saved["q2"]["outcome"] == "error"


def test_judge_daily_quota_stops_the_run_after_saving_that_question(tmp_path):
    results = tmp_path / run_eval.RESULTS_FILE
    scored = []

    def score(sample):
        scored.append(sample["user_input"])
        return "judge daily quota exhausted" if sample["user_input"] == "q2" else None

    samples, stop = run_eval.run_questions(
        client_for({}), "http://api", PAIRS, score=score,
        record=lambda s: run_eval.append_result(results, s))

    assert scored == ["q1", "q2"]
    assert [s["outcome"] for s in samples] == ["answered", "answered", "not_run", "not_run"]
    assert stop == "judge daily quota exhausted at question 2"
    assert set(run_eval.load_results(results)) == {"q1", "q2"}  # not_run isn't persisted


def test_resume_skips_finished_questions_and_reruns_the_rest(tmp_path):
    results = tmp_path / run_eval.RESULTS_FILE
    base = {"reference": "r", "timings": {}}
    for record in (
        {"user_input": "q1", "outcome": "answered", "faithfulness": 1.0, "context_precision": 1.0},
        {"user_input": "q2", "outcome": "refused_no_context"},
        {"user_input": "q3", "outcome": "answered", "faithfulness": 1.0, "context_precision": None},
        {"user_input": "q4", "outcome": "error", "stage": "generate", "error": "unavailable"},
    ):
        run_eval.append_result(results, {**base, **record})
    with results.open("a", encoding="utf-8") as fh:
        fh.write('{"user_input": "q5", "outc')  # torn line from a killed run

    done = {q: r for q, r in run_eval.load_results(results).items() if run_eval.is_finished(r)}
    queried = []

    def handler(request):
        question = json.loads(request.content)["question"]
        queried.append(question)
        return answered(question)

    samples, _ = run_eval.run_questions(httpx.Client(transport=httpx.MockTransport(handler)),
                                        "http://api", PAIRS, done=done)

    assert set(done) == {"q1", "q2"}
    assert queried == ["q3", "q4"]  # partly scored and errored questions run again
    assert [s["outcome"] for s in samples] == ["answered", "refused_no_context", "answered", "answered"]


QUOTA_TEXT = ("ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'details': [{'@type': "
              "'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaId': "
              "'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaValue': '15'}]}]}}")


def test_failed_query_keeps_full_error_text_and_llm_usage():
    usage = {"attempts_per_model": {"flash-lite": 3},
             "failures": [{"model": "flash-lite", "attempt": 1, "error": QUOTA_TEXT}]}
    client = client_for({"q1": httpx.Response(503, json={
        "detail": "Pipeline failed.", "stage": "generate", "error": "unavailable",
        "message": "ServerError: 503 UNAVAILABLE. {...full body...}", "llm_usage": usage})})

    samples, _ = run_eval.run_questions(client, "http://api", PAIRS[:1])

    assert samples[0]["error_message"] == "ServerError: 503 UNAVAILABLE. {...full body...}"
    assert samples[0]["llm_usage"] == usage


def test_call_accounting_totals_requests_failures_and_quota_hits():
    samples = [
        {"llm_usage": {"attempts_per_model": {"flash-lite": 4},
                       "failures": [{"model": "flash-lite", "attempt": 1, "error": QUOTA_TEXT}]},
         "judge_requests": {"judge": 8}, "judge_errors": [QUOTA_TEXT, "ServerError: 503 UNAVAILABLE"]},
        {"llm_usage": {"attempts_per_model": {"flash-lite": 3}, "failures": []},
         "judge_requests": {"judge": 7}, "judge_errors": []},
    ]

    calls = run_eval.call_accounting(samples)

    assert calls["app_requests_per_model"] == {"flash-lite": 7}
    assert calls["app_failed_attempts_per_model"] == {"flash-lite": 1}
    assert calls["judge_requests_per_model"] == {"judge": 15}
    assert calls["judge_failed_attempts"] == 2
    assert calls["quota_hits"] == {"GenerateRequestsPerMinutePerProjectPerModel-FreeTier (limit 15)": 2}


def test_judge_requests_are_totalled_across_resumed_samples():
    samples = [
        {"judge_requests": {"gemini-3.1-flash-lite": 7}},
        {"judge_requests": {"gemini-3.1-flash-lite": 9}},
        {},
    ]
    assert run_eval.total_judge_requests(samples) == {"gemini-3.1-flash-lite": 16}


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


def test_reports_record_the_commit_and_uncommitted_changes(tmp_path):
    samples = [{"user_input": "q1", "outcome": "refused_no_context", "timings": {}}]
    summary = run_eval.summarize(samples, [])

    run_eval.write_reports(tmp_path, {"document": "sample.pdf"}, "dataset.json", samples,
                           summary, {}, {}, code={"git_sha": "abc123",
                                                 "dirty_files": ["backend/graph/nodes.py"]})

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["code"] == {"git_sha": "abc123", "dirty_files": ["backend/graph/nodes.py"]}
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Code: `abc123` **with uncommitted changes:** `backend/graph/nodes.py`" in markdown


def test_code_version_reads_this_checkout():
    code = run_eval.code_version()
    assert code["git_sha"] is None or len(code["git_sha"]) == 40
    assert isinstance(code["dirty_files"], list)
