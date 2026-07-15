"""Query routing classification: small talk vs. real document questions."""
import pytest

from graph.nodes import route_after_routing, route_query_node
from graph.routing import is_small_talk

SMALL_TALK = [
    "hi",
    "Hello!",
    "hey",
    "Good morning",
    "thanks!",
    "Thank you so much",
    "thx",
    "how are you?",
    "what's up",
    "ok",
    "cool",
    "got it",
    "bye",
    "goodbye!",
    "who are you?",
    "What can you do?",
    "help",
    "testing",
    "  Hi!!  ",
]

REAL_QUESTIONS = [
    "hello, what does section 3 say?",       # greeting + real question
    "What is the vacation policy?",
    "hi level threshold for the alarm?",      # starts with 'hi' but is a question
    "thanks to what mechanism does the pump cool?",
    "What's up with the Q3 budget numbers?",
    "help me understand the refund clause",
    "Who are you required to notify after an incident?",
    "summarize the document",
    "test coverage requirements in the spec?",
    "ok so what does the contract say about renewal?",
]


@pytest.mark.parametrize("question", SMALL_TALK)
def test_small_talk_detected(question):
    assert is_small_talk(question)


@pytest.mark.parametrize("question", REAL_QUESTIONS)
def test_document_questions_pass_through(question):
    # False positives are the costly error here — a real question must never
    # be answered with the canned small-talk response.
    assert not is_small_talk(question)


async def test_route_node_sets_direct_route():
    update = await route_query_node({"question": "hello!"})
    assert update == {"route": "routed_direct"}
    assert route_after_routing(update) == "small_talk"


async def test_route_node_sets_pipeline_route():
    update = await route_query_node({"question": "What is the refund policy?"})
    assert update == {"route": "full_pipeline"}
    assert route_after_routing(update) == "retrieve"
