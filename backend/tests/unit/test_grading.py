"""Grading, generation and groundedness see each retrieved chunk in full; only
the citation tooltip's snippet is cut."""
import pytest

import config
from graph import nodes
from tests.helpers import FakeLLM


@pytest.fixture
def fake_llm(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    config.get_settings.cache_clear()
    llm = FakeLLM()
    monkeypatch.setattr(nodes, "get_llm", lambda: llm)
    yield llm
    config.get_settings.cache_clear()


async def test_grader_sees_text_past_700_chars(fake_llm):
    # Regression: the grader used to get only each chunk's first 700
    # characters, so a fact late in a chunk (eval Q7 sits at char ~940)
    # never reached it and the question was refused.
    fact = "any parameter out of range for more than 5 minutes triggers a Tier-2 alarm"
    text = "Filler about the greenhouse. " * 40 + fact
    assert text.index(fact) > 700
    chunk = {"text": text, "filename": "sample.pdf", "page_number": 2}

    update = await nodes.grade_node({"question": "What happens after 5 minutes?", "chunks": [chunk]})

    assert update["grade_relevant"] is True
    (call,) = fake_llm.chat_calls
    assert fact in call["user"]


async def test_snippet_cut_does_not_reach_generation_or_groundedness(fake_llm):
    fact = "any parameter out of range for more than 5 minutes triggers a Tier-2 alarm"
    text = "Filler about the greenhouse. " * 10 + fact
    assert text.index(fact) > 200
    chunk = {"chunk_id": "c1", "text": text, "filename": "sample.pdf", "page_number": 2}
    state = {"question": "What happens after 5 minutes?", "chunks": [chunk]}

    update = await nodes.generate_node(state)
    await nodes.groundedness_check_node({**state, "answer": update["answer"]})

    generate, groundedness = fake_llm.chat_calls
    assert fact in generate["user"] and fact in groundedness["user"]
    assert update["citations"][0]["snippet"] == text[:200]
