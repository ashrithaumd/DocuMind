"""Full query path with LLM calls stubbed (deterministic, zero API cost):
routing shortcut, refusal paths, the normal answer path, doc scoping, PII
redaction, feedback — and the log rows each one must leave behind.

Each test runs against its own fresh Postgres + pgvector database (the
chunk store and the query log), with the LLM and embeddings faked.
"""
from db import session as db_session
from tests.helpers import make_docx


def _ingest_two_docs(client):
    """Two small docs with disjoint vocabulary; returns their doc_ids."""
    first = client.post(
        "/upload",
        files={"file": ("reactor.txt",
                        b"The reactor manual covers cooling. Cooling pumps run at 400 RPM.",
                        "text/plain")},
    ).json()
    second = client.post(
        "/upload",
        files={"file": ("handbook.docx",
                        make_docx(["The HR handbook covers vacation.",
                                   "Vacation days accrue at 1.5 per month."]),
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    ).json()
    return first["doc_id"], second["doc_id"]


def _last_log():
    logs = db_session.fetch_logs(1)
    assert logs, "expected a query log row"
    return logs[0]


# --- routing: small-talk shortcut ---------------------------------------------


def test_small_talk_skips_pipeline_entirely(client, app_env):
    response = client.post("/query", json={"question": "hello!"})
    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "small_talk"
    assert body["route"] == "routed_direct"
    assert "DocuMind" in body["answer"]
    assert body["chunks"] == [] and body["citations"] == []

    # The whole point: zero LLM traffic on this path.
    assert app_env.chat_calls == []
    assert app_env.embed_calls == []

    row = _last_log()
    assert row["route"] == "routed_direct"
    assert row["outcome"] == "small_talk"


# --- grading-fails refusal path -------------------------------------------------


def test_grading_fails_refusal(client, app_env):
    _ingest_two_docs(client)
    app_env.chat_calls.clear()
    app_env.relevant = False

    body = client.post("/query", json={"question": "What is the capital of France?"}).json()
    assert body["outcome"] == "refused_no_context"
    assert "don't have enough information" in body["answer"]
    assert len(app_env.chat_calls) == 1  # grade only — no generate, no groundedness

    row = _last_log()
    assert row["route"] == "full_pipeline"
    assert row["outcome"] == "refused_no_context"
    assert row["grade_relevant"] is False
    assert row["grounded"] is None


def test_empty_corpus_refusal_mentions_upload(client, app_env):
    app_env.relevant = False
    body = client.post("/query", json={"question": "What does the doc say?"}).json()
    assert body["outcome"] == "refused_no_context"
    assert "No documents have been uploaded yet" in body["answer"]


# --- normal generate + groundedness path ----------------------------------------


def test_normal_answer_path(client, app_env):
    _ingest_two_docs(client)
    app_env.chat_calls.clear()
    app_env.answer = "Vacation days accrue at 1.5 per month [1]."

    body = client.post("/query", json={"question": "How do vacation days accrue?"}).json()
    assert body["outcome"] == "answered"
    assert body["answer"] == "Vacation days accrue at 1.5 per month [1]."
    assert body["citations"], "expected citations"
    assert len(app_env.chat_calls) == 3  # grade + generate + groundedness

    # Model routing: checkers on GRADING_MODEL, generation on
    # GENERATION_MODEL (both Flash-Lite by default) — all via the gateway.
    models = [call["model"] for call in app_env.chat_calls]
    assert models[0] == "gemini-3.5-flash-lite"   # relevance grade
    assert models[1] == "gemini-3.5-flash-lite"   # generation
    assert models[2] == "gemini-3.5-flash-lite"   # groundedness check

    row = _last_log()
    assert row["outcome"] == "answered"
    assert row["route"] == "full_pipeline"
    assert row["grade_relevant"] is True
    assert row["grounded"] is True
    assert row["regenerated"] is False
    assert row["retrieved_chunks"], "retrieval detail must be logged"


def test_ungrounded_retries_once_then_refuses(client, app_env):
    _ingest_two_docs(client)
    app_env.chat_calls.clear()
    app_env.grounded = [False, False]  # fail the check, fail the retry

    body = client.post("/query", json={"question": "How do vacation days accrue?"}).json()
    assert body["outcome"] == "refused_ungrounded"
    # grade + (generate + groundedness) * 2 = 5 calls, the documented worst case
    assert len(app_env.chat_calls) == 5

    row = _last_log()
    assert row["outcome"] == "refused_ungrounded"
    assert row["regenerated"] is True


# --- doc scoping -----------------------------------------------------------------


def test_default_scope_is_newest_upload(client, app_env):
    reactor_id, handbook_id = _ingest_two_docs(client)

    body = client.post("/query", json={"question": "cooling pumps RPM"}).json()
    # Default = newest upload (the handbook), NOT the whole corpus — so the
    # reactor content must not be retrieved even though it matches the query.
    assert body["doc_ids"] == [handbook_id]
    assert all(c["chunk_id"].startswith(handbook_id) for c in body["chunks"])
    assert _last_log()["doc_ids"] == [handbook_id]


def test_explicit_doc_ids_scope_retrieval(client, app_env):
    reactor_id, handbook_id = _ingest_two_docs(client)

    body = client.post(
        "/query", json={"question": "cooling pumps RPM", "doc_ids": [reactor_id]}
    ).json()
    assert body["doc_ids"] == [reactor_id]
    assert body["chunks"], "expected retrieval hits in the scoped doc"
    assert all(c["chunk_id"].startswith(reactor_id) for c in body["chunks"])
    assert _last_log()["doc_ids"] == [reactor_id]


# --- PII redaction ---------------------------------------------------------------


def test_pii_never_reaches_llm_or_log(client, app_env):
    _ingest_two_docs(client)
    app_env.chat_calls.clear()

    ssn = "123-45-6789"
    client.post("/query", json={"question": f"Is SSN {ssn} in the vacation doc?"})

    assert ssn not in app_env.prompts_sent(), "raw SSN reached an LLM prompt"
    assert "[SSN_REDACTED]" in app_env.prompts_sent()

    row = _last_log()
    assert ssn not in row["question"]
    assert row["redacted"] is True
    assert row["redaction_categories"] == ["ssn"]


def test_clean_query_logs_no_redaction(client, app_env):
    _ingest_two_docs(client)
    client.post("/query", json={"question": "How do vacation days accrue?"})
    row = _last_log()
    assert row["redacted"] is False
    assert row["redaction_categories"] is None


# --- feedback ---------------------------------------------------------------------


def test_feedback_roundtrip_and_stats(client, app_env):
    _ingest_two_docs(client)

    query_id = client.post(
        "/query", json={"question": "How do vacation days accrue?"}
    ).json()["query_id"]
    assert isinstance(query_id, int)

    # Thumbs down, then add a comment — must upsert into ONE row.
    assert client.post(
        "/feedback", json={"query_id": query_id, "thumbs_up": False}
    ).status_code == 200
    assert client.post(
        "/feedback",
        json={"query_id": query_id, "thumbs_up": False, "comment": "missed the accrual rate"},
    ).status_code == 200

    stats = client.get("/admin/stats").json()
    feedback = stats["feedback"]
    assert feedback["total"] == 1
    assert feedback["thumbs_down"] == 1
    assert feedback["thumbs_down_pct"] == 100.0
    negative = feedback["recent_negative"][0]
    assert negative["query_id"] == query_id
    assert negative["comment"] == "missed the accrual rate"
    assert "vacation days" in negative["question"]

    # Changing the verdict updates the same row.
    client.post("/feedback", json={"query_id": query_id, "thumbs_up": True})
    stats = client.get("/admin/stats").json()
    assert stats["feedback"]["total"] == 1
    assert stats["feedback"]["thumbs_down"] == 0


def test_feedback_unknown_query_404(client):
    response = client.post("/feedback", json={"query_id": 424242, "thumbs_up": True})
    assert response.status_code == 404


# --- SSE stream carries the new fields --------------------------------------------


def test_stream_done_event_has_query_id_and_stages(client, app_env):
    _ingest_two_docs(client)

    import json as jsonlib

    with client.stream(
        "POST", "/query/stream", json={"question": "How do vacation days accrue?"}
    ) as response:
        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(jsonlib.loads(line[6:]))

    stages = [e["stage"] for e in events if e["type"] == "stage"]
    assert stages[:3] == ["input_validation", "route_query", "retrieve"]
    done = next(e for e in events if e["type"] == "done")
    assert done["outcome"] == "answered"
    assert isinstance(done["query_id"], int)


def _daily_quota_error():
    from google.genai import errors as genai_errors

    return genai_errors.ClientError(429, {"error": {
        "code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED",
        "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                     "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                                     "quotaValue": "20"}]}],
    }})


def _expose_llm_diagnostics(monkeypatch):
    import config

    monkeypatch.setenv("EXPOSE_LLM_DIAGNOSTICS", "true")
    config.get_settings.cache_clear()


def test_pipeline_failure_returns_stage_and_error_kind(client, app_env):
    _ingest_two_docs(client)
    app_env.generate_error = _daily_quota_error()

    response = client.post("/query", json={"question": "How do vacation days accrue?"})

    assert response.status_code == 503
    # Diagnostics off by default: provider details stay in the server log.
    assert response.json() == {"detail": "Pipeline failed.", "stage": "generate", "error": "daily_quota"}


def test_pipeline_failure_with_diagnostics_has_full_error_and_usage(client, app_env, monkeypatch):
    _ingest_two_docs(client)
    _expose_llm_diagnostics(monkeypatch)
    app_env.generate_error = _daily_quota_error()

    response = client.post("/query", json={"question": "How do vacation days accrue?"})

    assert response.status_code == 503
    body = response.json()
    assert (body["stage"], body["error"]) == ("generate", "daily_quota")
    # Full error text, so the eval report can record the quota id and limit.
    assert "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in body["message"]
    assert "'quotaValue': '20'" in body["message"]
    assert set(body["llm_usage"]) == {"attempts_per_model", "failures"}


def test_answer_has_no_llm_usage_by_default(client, app_env):
    _ingest_two_docs(client)

    body = client.post("/query", json={"question": "How do vacation days accrue?"}).json()

    assert body["outcome"] == "answered"
    assert "llm_usage" not in body


def test_answer_with_diagnostics_has_llm_usage(client, app_env, monkeypatch):
    _ingest_two_docs(client)
    _expose_llm_diagnostics(monkeypatch)

    body = client.post("/query", json={"question": "How do vacation days accrue?"}).json()

    assert body["outcome"] == "answered"
    assert set(body["llm_usage"]) == {"attempts_per_model", "failures"}


def test_unexpected_pipeline_failure_is_500_internal(client, app_env):
    _ingest_two_docs(client)
    app_env.generate_error = ValueError("bug")

    response = client.post("/query", json={"question": "How do vacation days accrue?"})

    assert response.status_code == 500
    assert response.json()["stage"] == "generate"
    assert response.json()["error"] == "internal"
