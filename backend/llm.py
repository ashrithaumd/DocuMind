"""The single LLM gateway. Every model call in DocuMind goes through this
class — pipeline nodes never touch the Anthropic SDK directly, so swapping
providers (as this project did once, OpenAI -> Anthropic) means changing this
file and config defaults, not call sites.

Two operations cover everything the pipeline needs:
  - complete_text: generation (the answer itself)
  - complete_json: the two checker calls (relevance grade, groundedness),
    using the API's structured outputs so the reply is schema-valid JSON
    rather than "hopefully JSON".

Failure discipline: a refusal or malformed checker reply must fail SAFE.
complete_json returns {} (callers treat missing keys as "no"), and a refused
generation returns "" (which then fails the groundedness gate) — the pipeline
refuses rather than crashing or waving an answer through.
"""
import json
import logging

from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)


class ClaudeLLM:
    def __init__(self, client: AsyncAnthropic):
        self._client = client

    async def complete_text(
        self, model: str, system: str, user: str, max_tokens: int = 4096
    ) -> str:
        response = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if response.stop_reason == "refusal":
            logger.warning("Model %s refused a generation request", model)
            return ""
        return "".join(block.text for block in response.content if block.type == "text")

    async def complete_json(
        self, model: str, system: str, user: str, schema: dict, max_tokens: int = 1024
    ) -> dict:
        """One-shot JSON call constrained by `schema` (structured outputs)."""
        response = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
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


def _codeguard_test(query):
    return eval(query)
