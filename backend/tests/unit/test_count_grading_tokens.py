"""The token-count script rebuilds the grading request grade_node really sends."""
import sys
from pathlib import Path

import pytest

import config
from graph import nodes
from tests.helpers import FakeLLM

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
import count_grading_tokens  # noqa: E402

CHUNKS = [
    {"text": "A" * 700 + " late fact one", "filename": "sample.pdf", "page_number": 2},
    {"text": "B" * 900, "filename": "sample.pdf", "page_number": 4},
]


@pytest.fixture
def fake_llm(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    config.get_settings.cache_clear()
    llm = FakeLLM()
    monkeypatch.setattr(nodes, "get_llm", lambda: llm)
    yield llm
    config.get_settings.cache_clear()


async def test_full_variant_matches_grade_node(fake_llm):
    await nodes.grade_node({"question": "What is late?", "chunks": CHUNKS})

    (call,) = fake_llm.chat_calls
    assert call["system"] == nodes._GRADE_SYSTEM
    assert call["user"] == count_grading_tokens.grading_user_content("What is late?", CHUNKS)


def test_truncated_variant_matches_the_old_700_char_helper():
    user = count_grading_tokens.grading_user_content(
        "q", count_grading_tokens.truncated(CHUNKS, 700))

    # The pre-fix helper rendered chunk["text"][:700] under the same header.
    assert "[1] (sample.pdf, p. 2)\n" + CHUNKS[0]["text"][:700] + "\n\n[2]" in user
    assert user.endswith("[2] (sample.pdf, p. 4)\n" + "B" * 700)
    assert "late fact one" not in user
