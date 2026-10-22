"""The real gateway classes (GeminiLLM, ClaudeLLM) against stubbed SDK
clients: same contract for both providers — text and structured JSON calls,
fail-safe results on refusal/block/malformed output, bounded backoff on
rate limits — plus provider selection. No keys, no network."""
from types import SimpleNamespace

import anthropic
import httpx
import pytest
from google.genai import errors as genai_errors
from google.genai import types as genai_types

import llm
from config import Settings
from llm import ClaudeLLM, GeminiLLM, build_llm

SCHEMA = {
    "type": "object",
    "properties": {"relevant": {"type": "boolean"}},
    "required": ["relevant"],
    "additionalProperties": False,
}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Backoff waits are recorded, not slept."""
    waits: list[float] = []

    async def fake_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(llm.asyncio, "sleep", fake_sleep)
    return waits


# --- stub SDK clients -------------------------------------------------------------


class _Scripted:
    """Async callable that returns (or raises) scripted results in order and
    records the kwargs of every call."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls: list[dict] = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def gemini_client(*results):
    create = _Scripted(*results)
    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=create)))
    return client, create


def gemini_reply(text="", finish=genai_types.FinishReason.STOP, block_reason=None):
    return genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(role="model", parts=[genai_types.Part(text=text)]),
                finish_reason=finish,
            )
        ],
        prompt_feedback=(
            genai_types.GenerateContentResponsePromptFeedback(block_reason=block_reason)
            if block_reason else None
        ),
    )


def gemini_error(code, quota_id=None, retry_delay=None):
    details = []
    if quota_id:
        details.append({
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaId": quota_id, "quotaValue": "20"}],
        })
    if retry_delay:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay})
    body = {"error": {"code": code, "message": "nope", "status": "X", "details": details}}
    cls = genai_errors.ClientError if code < 500 else genai_errors.ServerError
    return cls(code, body)


def claude_client(*results):
    create = _Scripted(*results)
    return SimpleNamespace(messages=SimpleNamespace(create=create)), create


def claude_reply(text="", stop_reason="end_turn"):
    return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])


def claude_error(status):
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.anthropic.com"))
    if status == 429:
        return anthropic.RateLimitError("rate limited", response=response, body=None)
    return anthropic.APIStatusError("overloaded", response=response, body=None)


# Both providers through one parametrized contract: (factory, ok_text,
# refusal reply, transient error).
PROVIDERS = {
    "gemini": (
        lambda *r: (lambda c, s: (GeminiLLM(c, max_attempts=3), s))(*gemini_client(*r)),
        gemini_reply,
        lambda: gemini_reply("", finish=genai_types.FinishReason.SAFETY),
        lambda: gemini_error(429),
    ),
    "anthropic": (
        lambda *r: (lambda c, s: (ClaudeLLM(c, max_attempts=3), s))(*claude_client(*r)),
        claude_reply,
        lambda: claude_reply("", stop_reason="refusal"),
        lambda: claude_error(429),
    ),
}


@pytest.fixture(params=sorted(PROVIDERS))
def provider(request):
    return PROVIDERS[request.param]


# --- shared contract ----------------------------------------------------------------


async def test_complete_text_returns_text(provider):
    make, reply, _, _ = provider
    gateway, create = make(reply("The pump runs at 400 RPM [1]."))
    assert await gateway.complete_text("m", "sys", "q") == "The pump runs at 400 RPM [1]."
    assert len(create.calls) == 1


async def test_complete_json_parses_structured_output(provider):
    make, reply, _, _ = provider
    gateway, _ = make(reply('{"relevant": true}'))
    assert await gateway.complete_json("m", "sys", "q", SCHEMA) == {"relevant": True}


async def test_refusal_fails_safe(provider):
    make, _, refusal, _ = provider
    gateway, _ = make(refusal(), refusal())
    assert await gateway.complete_text("m", "sys", "q") == ""
    assert await gateway.complete_json("m", "sys", "q", SCHEMA) == {}


async def test_malformed_json_fails_safe(provider):
    make, reply, _, _ = provider
    gateway, _ = make(reply('{"relevant": tr'))
    assert await gateway.complete_json("m", "sys", "q", SCHEMA) == {}


async def test_rate_limit_retried_with_exponential_backoff(provider, no_sleep):
    make, reply, _, transient = provider
    gateway, create = make(transient(), transient(), reply("ok"))
    assert await gateway.complete_text("m", "sys", "q") == "ok"
    assert len(create.calls) == 3
    assert no_sleep == [2.0, 4.0]


async def test_retries_are_bounded(provider):
    make, _, _, transient = provider
    gateway, create = make(transient(), transient(), transient(), transient())
    with pytest.raises(Exception):
        await gateway.complete_text("m", "sys", "q")
    assert len(create.calls) == 3  # max_attempts, then the error propagates


# --- Gemini specifics -----------------------------------------------------------------


async def test_gemini_sends_schema_as_structured_output():
    client, create = gemini_client(gemini_reply('{"relevant": false}'))
    await GeminiLLM(client).complete_json("gemini-x", "sys", "q", SCHEMA)
    config = create.calls[0]["config"]
    assert create.calls[0]["model"] == "gemini-x"
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == SCHEMA
    assert config.system_instruction == "sys"


async def test_gemini_checker_thinking_level_applies_to_checkers_only():
    client, create = gemini_client(gemini_reply('{"relevant": true}'), gemini_reply("answer"))
    gateway = GeminiLLM(client, checker_thinking_level="minimal")
    await gateway.complete_json("m", "sys", "q", SCHEMA)
    await gateway.complete_text("m", "sys", "q")
    checker_config, generation_config = (call["config"] for call in create.calls)
    assert checker_config.thinking_config.thinking_level == genai_types.ThinkingLevel.MINIMAL
    assert generation_config.thinking_config is None  # generation: model default


async def test_gemini_empty_thinking_level_leaves_model_default():
    client, create = gemini_client(gemini_reply('{"relevant": true}'))
    await GeminiLLM(client).complete_json("m", "sys", "q", SCHEMA)
    assert create.calls[0]["config"].thinking_config is None


async def test_gemini_blocked_prompt_fails_safe():
    blocked = genai_types.GenerateContentResponse(
        candidates=[],
        prompt_feedback=genai_types.GenerateContentResponsePromptFeedback(
            block_reason=genai_types.BlockedReason.SAFETY
        ),
    )
    client, _ = gemini_client(blocked, blocked)
    gateway = GeminiLLM(client)
    assert await gateway.complete_text("m", "sys", "q") == ""
    assert await gateway.complete_json("m", "sys", "q", SCHEMA) == {}


async def test_gemini_truncated_json_fails_safe():
    client, _ = gemini_client(
        gemini_reply('{"grounded": tru', finish=genai_types.FinishReason.MAX_TOKENS)
    )
    assert await GeminiLLM(client).complete_json("m", "sys", "q", SCHEMA) == {}


async def test_gemini_500_internal_is_retried():
    client, create = gemini_client(gemini_error(500), gemini_reply("ok"))
    assert await GeminiLLM(client).complete_text("m", "sys", "q") == "ok"
    assert len(create.calls) == 2


async def test_gemini_503_overloaded_is_retried():
    client, create = gemini_client(gemini_error(503), gemini_reply("ok"))
    assert await GeminiLLM(client).complete_text("m", "sys", "q") == "ok"
    assert len(create.calls) == 2


async def test_gemini_honours_server_retry_delay(no_sleep):
    client, _ = gemini_client(
        gemini_error(429, quota_id="GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                     retry_delay="17s"),
        gemini_reply("ok"),
    )
    assert await GeminiLLM(client).complete_text("m", "sys", "q") == "ok"
    assert no_sleep == [17.0]


async def test_gemini_daily_quota_fails_fast(caplog):
    client, create = gemini_client(
        gemini_error(429, quota_id="GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    )
    with pytest.raises(genai_errors.ClientError):
        await GeminiLLM(client).complete_text("m", "sys", "q")
    assert len(create.calls) == 1  # no retry: a daily cap won't clear in seconds
    assert "PerDay" in caplog.text and "limit 20" in caplog.text


async def test_gemini_non_retryable_error_propagates_immediately():
    client, create = gemini_client(gemini_error(400))
    with pytest.raises(genai_errors.ClientError):
        await GeminiLLM(client).complete_text("m", "sys", "q")
    assert len(create.calls) == 1


# --- Anthropic specifics ----------------------------------------------------------------


async def test_claude_overloaded_529_is_retried():
    client, create = claude_client(claude_error(529), claude_reply("ok"))
    assert await ClaudeLLM(client).complete_text("m", "sys", "q") == "ok"
    assert len(create.calls) == 2


async def test_claude_sends_schema_as_structured_output():
    client, create = claude_client(claude_reply('{"relevant": true}'))
    await ClaudeLLM(client).complete_json("claude-x", "sys", "q", SCHEMA)
    assert create.calls[0]["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}}


# --- provider selection -------------------------------------------------------------------


def test_default_provider_is_gemini_with_free_tier_models():
    settings = Settings(_env_file=None, gemini_api_key="k")
    assert settings.llm_provider == "gemini"
    assert settings.generation_model == "gemini-3.6-flash"
    assert settings.grading_model == "gemini-3.5-flash-lite"
    assert settings.grading_thinking_level == "minimal"
    gateway = build_llm(settings)
    assert isinstance(gateway, GeminiLLM)
    assert gateway._checker_thinking.thinking_level == genai_types.ThinkingLevel.MINIMAL


def test_anthropic_provider_keeps_claude_defaults():
    settings = Settings(_env_file=None, llm_provider="anthropic", anthropic_api_key="k")
    assert settings.generation_model == "claude-opus-4-8"
    assert settings.grading_model == "claude-haiku-4-5"
    assert isinstance(build_llm(settings), ClaudeLLM)


def test_explicit_models_override_provider_defaults():
    settings = Settings(_env_file=None, generation_model="g", grading_model="r")
    assert (settings.generation_model, settings.grading_model) == ("g", "r")


@pytest.mark.parametrize("provider_name", ["gemini", "anthropic"])
def test_missing_key_for_selected_provider_fails_loudly(monkeypatch, provider_name):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = Settings(_env_file=None, llm_provider=provider_name)
    with pytest.raises(RuntimeError, match="API_KEY is not set"):
        build_llm(settings)


# --- error classification -------------------------------------------------------------


def test_classify_error_kinds():
    from llm import classify_error

    daily = gemini_error(429, quota_id="GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    per_minute = gemini_error(429, quota_id="GenerateRequestsPerMinutePerProjectPerModel-FreeTier")
    assert classify_error(daily) == "daily_quota"
    assert classify_error(per_minute) == "rate_limited"
    assert classify_error(gemini_error(500)) == "unavailable"
    assert classify_error(gemini_error(503)) == "unavailable"
    assert classify_error(gemini_error(400)) is None
    assert classify_error(claude_error(429)) == "rate_limited"
    assert classify_error(claude_error(529)) == "unavailable"
    assert classify_error(ValueError("bug")) is None
