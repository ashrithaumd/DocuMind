"""DocuMind API.

Endpoints:
    POST /upload        — validate + ingest a PDF/DOCX/TXT (chunks -> Chroma + BM25)
    GET  /documents     — uploaded documents, newest first
    POST /query         — run the full pipeline, return one JSON response
    POST /query/stream  — same pipeline over SSE: stage events, then the
                          verified answer, citations, and outcome
    GET  /admin/logs    — recent query logs (Postgres)
    GET  /admin/stats   — outcome counts + per-stage latency percentiles
    GET  /health        — liveness + index size

Streaming design note: the answer is only sent AFTER the groundedness check
passes, so we cannot forward raw LLM tokens as they are produced (they might
be retracted). Instead the SSE stream carries live per-stage progress events
while the graph runs, then the verified answer. Trust-first over latency-first
— that trade is deliberate and documented in the README.
"""
import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from config import get_settings
from db import session as db
from deps import get_store
from graph.pipeline import pipeline
from ingestion.chunking import chunk_pages
from ingestion.extract import IngestionError, validate_and_extract
from llm import classify_error

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("documind")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    db.init_db(settings)
    # Building the store loads the local embedding model (first run downloads
    # it from HuggingFace, then it's served from cache, fully offline) and
    # fails fast with a clear error if an existing Chroma collection was
    # built with a different embedding model.
    store = get_store()  # also rebuilds the BM25 index from Chroma
    logger.info("Store ready: %d chunks indexed.", store.count())
    if not settings.api_key:
        logger.warning("API_KEY not set — endpoints are unauthenticated (dev mode).")
    yield


app = FastAPI(title="DocuMind API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def require_api_key(x_api_key: str | None = Header(default=None)):
    """Single shared-key gate. When API_KEY is unset (local dev), it's open."""
    expected = get_settings().api_key
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


class QueryRequest(BaseModel):
    question: str
    # Documents to query. Omitted/empty -> the single most recent upload
    # (deliberately NOT the whole corpus, to avoid cross-document bleed).
    doc_ids: list[str] | None = None


class FeedbackRequest(BaseModel):
    query_id: int
    thumbs_up: bool
    comment: str | None = Field(default=None, max_length=2000)


# --- helpers -------------------------------------------------------------------


def _initial_state(request: QueryRequest) -> dict:
    return {"question": request.question, "doc_ids": request.doc_ids, "timings": {}}


def _merge_update(final: dict, delta: dict) -> None:
    """Fold one node's partial update into our running copy of the state."""
    for key, value in (delta or {}).items():
        if key == "timings":
            final.setdefault("timings", {}).update(value)
        else:
            final[key] = value


def _db_record(request: QueryRequest, state: dict, total_ms: float) -> dict:
    timings = state.get("timings", {})
    redaction_categories = state.get("redaction_categories", [])
    return {
        # state["question"] is the REDACTED question (set by input_validation);
        # the raw one must not reach the log. Fall back to the request only for
        # crashes before validation ran.
        "question": state.get("question", request.question),
        "redacted": bool(redaction_categories),
        "redaction_categories": redaction_categories or None,
        "answer": state.get("answer") or state.get("rejection"),
        "outcome": state.get("outcome", "error"),
        "route": state.get("route"),
        "doc_ids": state.get("scoped_doc_ids"),
        "retrieved_chunks": [
            {
                "chunk_id": c["chunk_id"],
                "page_number": c["page_number"],
                "score": round(c["score"], 5),
                "similarity": c["similarity"],
                "dense_rank": c["dense_rank"],
                "sparse_rank": c["sparse_rank"],
            }
            for c in state.get("chunks", [])
        ],
        "grade_relevant": state.get("grade_relevant"),
        "grounded": state.get("grounded"),
        "regenerated": state.get("regenerated", False),
        "retrieval_ms": timings.get("retrieval_ms"),
        "grading_ms": timings.get("grading_ms"),
        "generation_ms": timings.get("generation_ms"),
        "groundedness_ms": timings.get("groundedness_ms"),
        "total_ms": total_ms,
    }


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


_STAGE_LABELS = {
    "input_validation": "Validating question",
    "route_query": "Routing",
    "small_talk": "Replying",
    "retrieve": "Retrieving relevant chunks",
    "grade": "Grading relevance",
    "generate": "Generating answer",
    "groundedness_check": "Verifying groundedness",
    "refuse": "No grounded answer",
}


def _stage_detail(node: str, delta: dict) -> dict:
    """Small, safe-to-show facts about what each stage decided."""
    if node == "input_validation":
        # Category names only — never the redacted values themselves.
        return {"redacted": delta.get("redaction_categories", [])}
    if node == "retrieve":
        return {"chunks": len(delta.get("chunks", []))}
    if node == "grade":
        return {"relevant": delta.get("grade_relevant")}
    if node == "generate":
        return {"retry": delta.get("regenerated", False)}
    if node == "groundedness_check":
        return {"grounded": delta.get("grounded")}
    return {}


# --- endpoints -----------------------------------------------------------------


@app.post("/upload", dependencies=[Depends(require_api_key)])
async def upload_document(file: UploadFile = File(...)):
    settings = get_settings()
    file_bytes = await file.read()
    filename = file.filename or ""

    try:
        pages, warnings = validate_and_extract(file_bytes, filename, settings)
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chunks = chunk_pages(pages, filename, settings)
    try:
        await get_store().add_chunks(chunks)
    except Exception as exc:
        # Embeddings are local now, so a failure here is a server-side problem
        # (model cache, disk, Chroma) — surface that clearly, not a bare 500.
        logger.exception("Embedding/indexing failed during ingestion")
        raise HTTPException(
            status_code=500,
            detail="Embedding or indexing failed server-side — see backend logs.",
        ) from exc
    logger.info("Ingested %s: %d pages, %d chunks", filename, len(pages), len(chunks))

    return {
        "filename": filename,
        "doc_id": chunks[0].doc_id,
        "pages": len(pages),
        "chunks_created": len(chunks),
        "warnings": warnings,
    }


@app.get("/documents", dependencies=[Depends(require_api_key)])
async def list_documents():
    """Uploaded documents, newest first. The first entry is the default query scope."""
    return {"documents": get_store().list_documents()}


def _pipeline_error(exc: Exception) -> JSONResponse:
    """Structured failure for /query: which pipeline node failed and a
    provider-neutral error kind (raw provider messages stay in the server
    log). LLM capacity errors are 503 (retry later); anything else is 500."""
    # LangGraph notes the failing node on the exception:
    # "During task with name 'generate' and id '...'".
    notes = " ".join(getattr(exc, "__notes__", []))
    match = re.search(r"During task with name '(\w+)'", notes)
    stage = match.group(1) if match else "pipeline"
    kind = classify_error(exc)
    logger.exception("Pipeline failed at %s (%s)", stage, kind or "internal")
    return JSONResponse(
        status_code=503 if kind else 500,
        content={"detail": "Pipeline failed.", "stage": stage, "error": kind or "internal"},
    )


@app.post("/query", dependencies=[Depends(require_api_key)])
async def query(request: QueryRequest):
    """Non-streaming variant — used by the eval harness and handy for curl."""
    start = time.perf_counter()
    try:
        state = await pipeline.ainvoke(_initial_state(request))
    except Exception as exc:
        return _pipeline_error(exc)
    total_ms = round((time.perf_counter() - start) * 1000, 1)

    query_id = await asyncio.to_thread(
        db.log_query, _db_record(request, state, total_ms)
    )

    if state.get("outcome") == "rejected_input":
        raise HTTPException(status_code=422, detail=state["rejection"])

    return {
        "query_id": query_id,
        "answer": state.get("answer", ""),
        "outcome": state.get("outcome", "error"),
        "route": state.get("route"),
        "doc_ids": state.get("scoped_doc_ids", []),
        "citations": state.get("citations", []),
        # Full chunk texts are returned so the eval harness can score
        # context precision without re-querying the store.
        "chunks": [
            {
                "chunk_id": c["chunk_id"],
                "page_number": c["page_number"],
                "text": c["text"],
                "score": round(c["score"], 5),
                "similarity": c["similarity"],
            }
            for c in state.get("chunks", [])
        ],
        "timings": {**state.get("timings", {}), "total_ms": total_ms},
    }


@app.post("/query/stream", dependencies=[Depends(require_api_key)])
async def query_stream(request: QueryRequest):
    async def event_stream():
        start = time.perf_counter()
        final: dict = {}
        try:
            async for update in pipeline.astream(
                _initial_state(request), stream_mode="updates"
            ):
                for node, delta in update.items():
                    _merge_update(final, delta)
                    yield _sse(
                        {
                            "type": "stage",
                            "stage": node,
                            "label": _STAGE_LABELS.get(node, node),
                            "detail": _stage_detail(node, delta or {}),
                        }
                    )

            total_ms = round((time.perf_counter() - start) * 1000, 1)
            answer = final.get("answer") or final.get("rejection") or ""

            # The answer is already verified; emit it in small chunks so the
            # frontend renders progressively.
            words = answer.split(" ")
            for i in range(0, len(words), 8):
                yield _sse({"type": "token", "text": " ".join(words[i : i + 8]) + (" " if i + 8 < len(words) else "")})

            yield _sse({"type": "citations", "citations": final.get("citations", [])})

            # Log BEFORE the done event so it can carry the query_id the
            # feedback buttons need to reference this exchange.
            query_id = await asyncio.to_thread(
                db.log_query, _db_record(request, final, total_ms)
            )
            yield _sse(
                {
                    "type": "done",
                    "query_id": query_id,
                    "outcome": final.get("outcome", "error"),
                    "timings": {**final.get("timings", {}), "total_ms": total_ms},
                }
            )
        except Exception:
            logger.exception("Pipeline failed")
            yield _sse(
                {"type": "error", "message": "Something went wrong running the pipeline."}
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/feedback", dependencies=[Depends(require_api_key)])
async def submit_feedback(request: FeedbackRequest):
    """Thumbs up/down (plus optional comment) for a logged query.

    Upserts: one feedback row per query, so re-clicking or adding a comment
    updates the verdict instead of double-counting it in /admin/stats.
    """
    status = await asyncio.to_thread(
        db.log_feedback, request.query_id, request.thumbs_up, request.comment
    )
    if status == "unknown_query":
        raise HTTPException(status_code=404, detail=f"No query with id {request.query_id}.")
    if status == "disabled":
        raise HTTPException(
            status_code=503, detail="Feedback requires query logging (DATABASE_URL) to be enabled."
        )
    if status == "error":
        raise HTTPException(status_code=500, detail="Could not store feedback.")
    return {"status": "ok"}


@app.get("/admin/logs", dependencies=[Depends(require_api_key)])
async def admin_logs(limit: int = 20):
    return {"logs": await asyncio.to_thread(db.fetch_logs, min(limit, 100))}


@app.get("/admin/stats", dependencies=[Depends(require_api_key)])
async def admin_stats():
    return await asyncio.to_thread(db.fetch_stats)


@app.get("/health")
async def health():
    return {"status": "ok", "chunks_indexed": get_store().count()}
