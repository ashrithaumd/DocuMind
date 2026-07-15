"""The six pipeline nodes plus their routing functions.

This is a FIXED pipeline: every routing decision below is a deterministic
function of state (a boolean set by a cheap grading call), not an LLM choosing
tools at runtime. That keeps behavior auditable — for any logged query you can
replay exactly which path it took and why.

Failure behavior is explicit by design:
  - irrelevant retrieval  -> refuse ("I don't have enough information...")
  - ungrounded generation -> regenerate once, then refuse
Both refusals carry distinct `outcome` values so they are distinguishable from
real answers in the Postgres log.
"""
import logging
import re
import time
from dataclasses import asdict

from langgraph.graph import END

from config import get_settings
from deps import get_llm, get_store
from graph.redaction import redact
from graph.routing import SMALL_TALK_RESPONSE, is_small_talk
from graph.state import GraphState

logger = logging.getLogger(__name__)

REFUSAL_NO_CONTEXT = (
    "I don't have enough information in this document to answer that."
)
REFUSAL_NO_DOCUMENTS = (
    "No documents have been uploaded yet. Upload a PDF first, then ask again."
)
REFUSAL_UNGROUNDED = (
    "I couldn't produce an answer that is fully supported by the document, "
    "so I won't guess. Try rephrasing the question or asking about a "
    "narrower detail."
)


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def _numbered_context(chunks: list[dict], max_chars: int | None = None) -> str:
    """Render chunks as numbered excerpts the LLM can cite as [1], [2], ..."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        text = chunk["text"][:max_chars] if max_chars else chunk["text"]
        parts.append(f"[{i}] ({chunk['filename']}, p. {chunk['page_number']})\n{text}")
    return "\n\n".join(parts)


# --- 1. input_validation ------------------------------------------------------

async def input_validation_node(state: GraphState) -> dict:
    settings = get_settings()
    question = (state.get("question") or "").strip()
    if not question:
        return {"rejection": "Question is empty.", "outcome": "rejected_input"}

    # PII is masked here, at the front door: every downstream consumer — the
    # LLM calls AND the Postgres audit log — only ever sees the redacted
    # question. Rejected questions get redacted too, since they are logged.
    question, categories = redact(question)
    update: dict = {"question": question, "redaction_categories": categories}

    if len(question) > settings.max_query_chars:
        update |= {
            "rejection": (
                f"Question is too long ({len(question)} characters); "
                f"the limit is {settings.max_query_chars}."
            ),
            "outcome": "rejected_input",
        }
    else:
        update["rejection"] = None
    return update


def route_after_validation(state: GraphState) -> str:
    return END if state.get("rejection") else "route_query"


# --- 1b. route_query ------------------------------------------------------------

async def route_query_node(state: GraphState) -> dict:
    """Classify the input: real document question vs. greeting/small talk.

    The decision is recorded in state (and the audit log) so every logged
    query says which path it took. Heuristic by design — see graph/routing.py.
    """
    direct = is_small_talk(state["question"])
    return {"route": "routed_direct" if direct else "full_pipeline"}


def route_after_routing(state: GraphState) -> str:
    return "small_talk" if state["route"] == "routed_direct" else "retrieve"


async def small_talk_node(state: GraphState) -> dict:
    """Direct reply for conversational messages: no retrieval, no LLM calls."""
    return {
        "answer": SMALL_TALK_RESPONSE,
        "citations": [],
        "outcome": "small_talk",
    }


# --- 2. retrieve ---------------------------------------------------------------

async def retrieve_node(state: GraphState) -> dict:
    start = time.perf_counter()
    store = get_store()

    # Default scope is the single most recent upload, NOT the whole corpus —
    # querying everything by default invites answers stitched from unrelated
    # documents. Callers widen the scope explicitly by passing doc_ids.
    scoped = state.get("doc_ids") or []
    if not scoped:
        documents = store.list_documents()
        scoped = [documents[0]["doc_id"]] if documents else []

    results = await store.search(state["question"], doc_ids=scoped or None)

    # Redact retrieved text ONCE here, so every downstream LLM call (grade,
    # generate, groundedness) and every API response works from masked text.
    # PII inside a document can appear in a grounded answer otherwise.
    chunks = []
    categories = set(state.get("redaction_categories", []))
    for result in results:
        chunk = asdict(result)
        chunk["text"], chunk_categories = redact(chunk["text"])
        categories.update(chunk_categories)
        chunks.append(chunk)

    return {
        "chunks": chunks,
        "scoped_doc_ids": scoped,
        "redaction_categories": sorted(categories),
        "timings": {"retrieval_ms": _elapsed_ms(start)},
    }


# --- 3. grade ------------------------------------------------------------------

_GRADE_SYSTEM = """\
You are a relevance grader for a document Q&A system.
Given a user question and excerpts retrieved from uploaded documents, decide
whether the excerpts contain information that can actually answer the question.
Topical similarity is NOT enough — the excerpts must contain the substance of
an answer. Reply with JSON: {"relevant": true} or {"relevant": false}."""

_GRADE_SCHEMA = {
    "type": "object",
    "properties": {"relevant": {"type": "boolean"}},
    "required": ["relevant"],
    "additionalProperties": False,
}


async def grade_node(state: GraphState) -> dict:
    start = time.perf_counter()
    if not state["chunks"]:
        return {"grade_relevant": False, "timings": {"grading_ms": _elapsed_ms(start)}}

    # Truncated excerpts are plenty for a relevance judgment and keep the
    # grading call cheap.
    user = (
        f"Question: {state['question']}\n\nExcerpts:\n"
        f"{_numbered_context(state['chunks'], max_chars=700)}"
    )
    verdict = await get_llm().complete_json(
        get_settings().grading_model, _GRADE_SYSTEM, user, _GRADE_SCHEMA
    )
    return {
        "grade_relevant": bool(verdict.get("relevant", False)),
        "timings": {"grading_ms": _elapsed_ms(start)},
    }


def route_after_grade(state: GraphState) -> str:
    return "generate" if state["grade_relevant"] else "refuse"


# --- 4. generate ---------------------------------------------------------------

_GENERATE_SYSTEM = """\
You answer questions about uploaded documents.

Rules:
- Use ONLY the numbered excerpts below. Never use outside knowledge.
- Cite every factual claim inline with the excerpt number in square brackets,
  e.g. [1] or [2][3].
- If the excerpts only partially answer the question, answer what is supported
  and say explicitly what the document does not cover.
- Never invent names, numbers, or details that are not in the excerpts."""


async def generate_node(state: GraphState) -> dict:
    start = time.perf_counter()
    # "grounded" only exists in state after a groundedness check has run, so
    # its presence (as False) means this call is the one retry.
    is_retry = state.get("grounded") is False

    user = (
        f"Excerpts:\n{_numbered_context(state['chunks'])}\n\n"
        f"Question: {state['question']}"
    )
    if is_retry:
        claims = "\n".join(f"- {c}" for c in state.get("unsupported_claims", []))
        user += (
            "\n\nYour previous draft contained claims not supported by the "
            "excerpts:\n" + (claims or "- (unspecified)") +
            "\nRewrite the answer, keeping only claims directly supported by "
            "the excerpts."
        )

    answer = await get_llm().complete_text(
        get_settings().generation_model, _GENERATE_SYSTEM, user
    )

    # Citations list = the excerpts the answer actually references; fall back
    # to all retrieved chunks if the model produced no [n] markers.
    cited_numbers = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    citations = [
        {
            "n": i,
            "chunk_id": chunk["chunk_id"],
            "filename": chunk["filename"],
            "page_number": chunk["page_number"],
            "snippet": chunk["text"][:200],
        }
        for i, chunk in enumerate(state["chunks"], start=1)
        if not cited_numbers or i in cited_numbers
    ]

    # Accumulate across the retry so the log shows total generation cost.
    generation_ms = state.get("timings", {}).get("generation_ms", 0) + _elapsed_ms(start)
    return {
        "answer": answer,
        "citations": citations,
        "regenerated": is_retry,
        "timings": {"generation_ms": round(generation_ms, 1)},
    }


# --- 5. groundedness_check -----------------------------------------------------

_GROUNDEDNESS_SYSTEM = """\
You are a groundedness auditor for a document Q&A system.
Given source excerpts and a generated answer, verify that every factual claim
in the answer is directly supported by the excerpts. Statements about what the
document does NOT cover do not need support.
Reply with JSON:
{"grounded": true|false, "unsupported_claims": ["<claim>", ...]}"""

_GROUNDEDNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "grounded": {"type": "boolean"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["grounded", "unsupported_claims"],
    "additionalProperties": False,
}


async def groundedness_check_node(state: GraphState) -> dict:
    start = time.perf_counter()
    user = (
        f"Excerpts:\n{_numbered_context(state['chunks'])}\n\n"
        f"Answer to verify:\n{state['answer']}"
    )
    verdict = await get_llm().complete_json(
        get_settings().grading_model, _GROUNDEDNESS_SYSTEM, user, _GROUNDEDNESS_SCHEMA
    )
    grounded = bool(verdict.get("grounded", False))

    groundedness_ms = state.get("timings", {}).get("groundedness_ms", 0) + _elapsed_ms(start)
    update: dict = {
        "grounded": grounded,
        "unsupported_claims": verdict.get("unsupported_claims", []),
        "timings": {"groundedness_ms": round(groundedness_ms, 1)},
    }
    if grounded:
        update["outcome"] = "answered"
    return update


def route_after_groundedness(state: GraphState) -> str:
    if state["grounded"]:
        return END
    if not state["regenerated"]:
        return "generate"  # one retry with feedback about the failed claims
    return "refuse"


# --- refusal (explicit failure) -----------------------------------------------

async def refuse_node(state: GraphState) -> dict:
    """Terminal node for both failure paths. Refusing beats guessing."""
    if not state.get("grade_relevant"):
        no_docs = not state["chunks"] and get_store().count() == 0
        return {
            "answer": REFUSAL_NO_DOCUMENTS if no_docs else REFUSAL_NO_CONTEXT,
            "citations": [],
            "outcome": "refused_no_context",
        }
    return {
        "answer": REFUSAL_UNGROUNDED,
        "citations": [],
        "outcome": "refused_ungrounded",
    }
