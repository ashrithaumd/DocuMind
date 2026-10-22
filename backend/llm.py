"""The single LLM gateway. Every model call in DocuMind goes through this
module — pipeline nodes never touch a provider SDK directly, so swapping
providers (as this project has done twice: OpenAI -> Anthropic -> Gemini)
means changing LLM_PROVIDER, not call sites.

Two operations cover everything the pipeline needs:
  - complete_text: generation (the answer itself)
  - complete_json: the two checker calls (relevance grade, groundedness),
    using each API's structured outputs so the reply is schema-valid JSON
    rather than "hopefully JSON".

Two providers implement them: GeminiLLM (default — Google AI Studio free
tier) and ClaudeLLM (Anthropic). build_llm() picks one from settings.

Failure discipline: a refusal, a safety block, or a malformed checker reply
must fail SAFE. complete_json returns {} (callers treat missing keys as "no"),
and a refused/blocked generation returns "" (which then fails the
groundedness gate) — the pipeline refuses rather than crashing or waving an
answer through.

Rate limits: transient errors (429 rate limit, Gemini 500 INTERNAL, 503/529 overloaded)
are retried with bounded exponential backoff, logged per attempt. A Gemini
per-DAY quota 429 is not retried — it won't clear for hours — and the quota
id/limit from the error is logged so the real free-tier numbers are visible.
"""
import asyncio
import json
import logging

import anthropic
from anthropic import AsyncAnthropic
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

logger = logging.getLogger(__name__)


class _BackoffLLM:
    """Shared retry loop. Subclasses say which exceptions are retryable."""

    def __init__(self, max_attempts: int = 5, base_delay_s: float = 2.0, max_delay_s: float = 60.0):
        self._max_attempts = max_attempts
        self._base_delay_s = base_delay_s
        self._max_delay_s = max_delay_s

    def _retry_delay(self, exc: Exception) -> float | None:
        """Seconds the server asked us to wait (0.0 = no hint), or None if
        `exc` must not be retried."""
        raise NotImplementedError

    async def _call(self, model: str, request):
        for attempt in range(1, self._max_attempts + 1):
            try:
                return await request()
            except Exception as exc:
                hint = self._retry_delay(exc)
                if hint is None or attempt == self._max_attempts:
                    raise
                delay = min(
                    self._max_delay_s,
                    hint or self._base_delay_s * 2 ** (attempt - 1),
                )
                logger.warning(
                    "%s on %s (attempt %d/%d); retrying in %.1fs",
                    type(exc).__name__, model, attempt, self._max_attempts, delay,
                )
                await asyncio.sleep(delay)


# --- Gemini (default) ------------------------------------------------------------

# Finish reasons that mean "the model declined / was stopped by policy" —
# treated exactly like an Anthropic refusal.
_GEMINI_BLOCKED = {
    genai_types.FinishReason.SAFETY,
    genai_types.FinishReason.RECITATION,
    genai_types.FinishReason.BLOCKLIST,
    genai_types.FinishReason.PROHIBITED_CONTENT,
    genai_types.FinishReason.SPII,
    genai_types.FinishReason.LANGUAGE,
    genai_types.FinishReason.OTHER,
}


def _gemini_error_details(exc: genai_errors.APIError) -> list[dict]:
    body = exc.details if isinstance(exc.details, dict) else {}
    details = body.get("error", {}).get("details", [])
    return details if isinstance(details, list) else []


class GeminiLLM(_BackoffLLM):
    def __init__(self, client: genai.Client, checker_thinking_level: str = "", **backoff):
        """`checker_thinking_level` applies to complete_json (the checker
        calls) only; "" leaves the model's default. Generation always runs
        at the model's default thinking level."""
        super().__init__(**backoff)
        self._client = client
        self._checker_thinking = (
            genai_types.ThinkingConfig(thinking_level=checker_thinking_level)
            if checker_thinking_level else None
        )

    def _retry_delay(self, exc: Exception) -> float | None:
        # 429 rate limit, 500 transient INTERNAL, 503 overloaded.
        if not isinstance(exc, genai_errors.APIError) or exc.code not in (429, 500, 503):
            return None
        hint = 0.0
        for detail in _gemini_error_details(exc):
            kind = detail.get("@type", "")
            if kind.endswith("QuotaFailure"):
                for violation in detail.get("violations", []):
                    quota_id = violation.get("quotaId", "")
                    logger.warning(
                        "Gemini quota hit: %s (limit %s)",
                        quota_id, violation.get("quotaValue", "?"),
                    )
                    if "PerDay" in quota_id:
                        return None  # daily cap: retrying within seconds is pointless
            elif kind.endswith("RetryInfo"):
                try:
                    hint = float(str(detail.get("retryDelay", "0s")).rstrip("s"))
                except ValueError:
                    pass
        return hint

    def _usable_text(self, model: str, response, what: str) -> str | None:
        """Response text, or None if the prompt/answer was blocked."""
        feedback = response.prompt_feedback
        if feedback is not None and feedback.block_reason:
            logger.warning("Gemini %s blocked a %s prompt: %s", model, what, feedback.block_reason)
            return None
        if not response.candidates:
            logger.warning("Gemini %s returned no candidates for a %s request", model, what)
            return None
        reason = response.candidates[0].finish_reason
        if reason in _GEMINI_BLOCKED:
            logger.warning("Gemini %s stopped a %s request: %s", model, what, reason)
            return None
        return response.text or ""

    async def complete_text(
        self, model: str, system: str, user: str, max_tokens: int = 4096
    ) -> str:
        config = genai_types.GenerateContentConfig(
            system_instruction=system, max_output_tokens=max_tokens
        )
        response = await self._call(
            model,
            lambda: self._client.aio.models.generate_content(
                model=model, contents=user, config=config
            ),
        )
        return self._usable_text(model, response, "generation") or ""

    async def complete_json(
        self, model: str, system: str, user: str, schema: dict, max_tokens: int = 1024
    ) -> dict:
        """One-shot JSON call constrained by `schema` (structured output)."""
        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            response_json_schema=schema,
            thinking_config=self._checker_thinking,
        )
        response = await self._call(
            model,
            lambda: self._client.aio.models.generate_content(
                model=model, contents=user, config=config
            ),
        )
        text = self._usable_text(model, response, "checker")
        if text is None:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Includes output truncated at max_tokens (Gemini counts thinking
            # tokens against that budget). The checker contract stands:
            # malformed replies are treated as a "no".
            logger.warning("Checker returned malformed JSON for model %s", model)
            return {}
        return parsed if isinstance(parsed, dict) else {}


# --- Anthropic ---------------------------------------------------------------------


class ClaudeLLM(_BackoffLLM):
    def __init__(self, client: AsyncAnthropic, **backoff):
        super().__init__(**backoff)
        self._client = client

    def _retry_delay(self, exc: Exception) -> float | None:
        if isinstance(exc, anthropic.RateLimitError):
            return 0.0
        if isinstance(exc, anthropic.APIStatusError) and exc.status_code == 529:  # overloaded
            return 0.0
        return None

    async def complete_text(
        self, model: str, system: str, user: str, max_tokens: int = 4096
    ) -> str:
        response = await self._call(
            model,
            lambda: self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            ),
        )
        if response.stop_reason == "refusal":
            logger.warning("Model %s refused a generation request", model)
            return ""
        return "".join(block.text for block in response.content if block.type == "text")

    async def complete_json(
        self, model: str, system: str, user: str, schema: dict, max_tokens: int = 1024
    ) -> dict:
        """One-shot JSON call constrained by `schema` (structured outputs)."""
        response = await self._call(
            model,
            lambda: self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"format": {"type": "json_schema", "schema": schema}},
            ),
        )
        if response.stop_reason == "refusal":
            logger.warning("Model %s refused a checker request", model)
            return {}
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Structured outputs should make this impossible, but the checker
            # contract stands: malformed replies are treated as a "no".
            logger.warning("Checker returned malformed JSON for model %s", model)
            return {}


# --- provider selection --------------------------------------------------------------


def build_llm(settings) -> GeminiLLM | ClaudeLLM:
    """The configured provider's gateway. Fails loudly if its key is missing."""
    backoff = {"max_attempts": settings.llm_max_attempts}
    if settings.llm_provider == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set")
        return GeminiLLM(
            genai.Client(api_key=settings.gemini_api_key),
            checker_thinking_level=settings.grading_thinking_level,
            **backoff,
        )
    if not settings.anthropic_api_key:
        raise RuntimeError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
    # SDK-level retries off: the gateway's own backoff is the single retry layer.
    return ClaudeLLM(AsyncAnthropic(api_key=settings.anthropic_api_key, max_retries=0), **backoff)
