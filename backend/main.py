"""DocuMind API.

Endpoints:
    POST /upload        — validate + ingest a PDF/DOCX/TXT (editors/admins; the
                          form sets its clearance and optional team)
    GET  /documents     — uploaded documents, newest first
    POST /query         — run the full pipeline, return one JSON response
    POST /query/stream  — same pipeline over SSE: stage events, then the
                          verified answer, citations, and outcome
    PATCH /documents/{doc_id}/access — change a document's team/clearance
                          (editors/admins), with every chunk at once
    GET  /admin/logs    — recent query logs of the admin's org (admins only)
    GET  /admin/stats   — the same org's outcome counts + latency percentiles
    GET  /me            — who the token belongs to and whether they're set up
    GET  /health        — liveness + index size

Every endpoint except /me and /health requires a signed-in, set-up user
(auth.get_principal: 401 without a valid token, 403 if not set up); /admin/*
also checks with Firebase that the token wasn't revoked.

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
from functools import partial

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from auth import Principal, get_admin_principal, get_me, get_principal
from config import get_settings
from db.access import WRITER_ROLES
from db import session as db
import deps
from deps import get_store
from graph.pipeline import pipeline
from ingestion.chunking import chunk_pages
from ingestion.extract import IngestionError, validate_and_extract
from llm import classify_error, track_usage

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("documind")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.migration_database_url:
        # Owner credentials belong to the one-shot migrate step (migrate.py),
        # never to the serving process: with them, one bug could undo RLS.
        raise RuntimeError("MIGRATION_DATABASE_URL is set in the backend's environment. "
                           "Run migrations with `python migrate.py` (the migrate service) and "
                           "give the backend only DATABASE_URL (the documind_app role).")
    # Fail fast, with what to fix, rather than a 500 on the first sign-in.
    # Checked before the database: a config error shouldn't wait on Postgres.
    deps.get_token_checker().check_ready()
    logger.info("Firebase token checks ready.")
    db.init_db(settings)
    # The store fails fast with a clear error if the chunks table is missing
    # or its vector column doesn't match the embedding model. With
    # EMBEDDING_PROVIDER=local this also loads the model.
    get_store()
    logger.info("Store ready.")
    yield


app = FastAPI(title="DocuMind API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    # Documents to query. Omitted/empty -> the single most recent upload
    # (deliberately NOT the whole corpus, to avoid cross-document bleed).
    doc_ids: list[str] | None = None


class AccessChange(BaseModel):
    clearance: int = Field(ge=0)
    team_id: str | None = None  # None = org-wide


class FeedbackRequest(BaseModel):
    query_id: int
    thumbs_up: bool
    comment: str | None = Field(default=None, max_length=2000)


# --- helpers -------------------------------------------------------------------


def _initial_state(request: QueryRequest, principal: Principal) -> dict:
    return {"question": request.question, "doc_ids": request.doc_ids, "access": principal,
            "timings": {}}


def _merge_update(final: dict, delta: dict) -> None:
    """Fold one node's partial update into our running copy of the state."""
    for key, value in (delta or {}).items():
        if key == "timings":
            final.setdefault("timings", {}).update(value)
        else:
            final[key] = value


def _db_record(request: QueryRequest, state: dict, total_ms: float,
               principal: Principal) -> dict:
    timings = state.get("timings", {})
    redaction_categories = state.get("redaction_categories", [])
    return {
        "user_id": principal.user_id,
        "org_id": principal.org_id,
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


@app.post("/upload")
async def upload_document(file: UploadFile = File(...),
                          # Required: who may read it is always an explicit choice.
                          clearance: int = Form(..., ge=0),
                          # Empty = org-wide; otherwise one of the uploader's teams.
                          team_id: str | None = Form(None),
                          principal: Principal = Depends(get_principal)):
    # Checked before any work. RLS enforces the team and clearance rules
    # again in the database; these give a clear 403 instead of a 500.
    team_id = team_id or None
    _check_can_grant(principal, team_id, clearance)
    settings = get_settings()
    file_bytes = await file.read()
    filename = file.filename or ""

    try:
        pages, warnings = validate_and_extract(file_bytes, filename, settings)
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chunks = chunk_pages(pages, filename, settings)
    try:
        await get_store().scoped(principal).add_document(
            chunks, team_id=team_id, clearance=clearance, uploaded_by=principal.user_id)
    except Exception as exc:
        logger.exception("Embedding/indexing failed during ingestion")
        kind = classify_error(exc)
        if kind:
            # Embedding provider capacity (quota, overload, timeout): retry later.
            raise HTTPException(
                status_code=503, detail=f"Embedding provider unavailable ({kind}); try again later."
            ) from exc
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
        "team_id": team_id,
        "clearance": clearance,
        "warnings": warnings,
    }


def _check_can_grant(principal: Principal, team_id: str | None, clearance: int) -> None:
    """Uploads and access changes: editors and admins only, never into a team
    they're not in or above their own clearance. RLS enforces the team and
    clearance rules again in the database, and the store the role; these
    give a clear 403 first."""
    if principal.role not in WRITER_ROLES:
        raise HTTPException(status_code=403, detail="Only editors and admins can do that.")
    if team_id is not None and team_id not in principal.team_ids:
        raise HTTPException(status_code=403, detail=f"You're not in team '{team_id}'.")
    if clearance > principal.clearance:
        raise HTTPException(status_code=403,
                            detail=f"Clearance {clearance} is above your own ({principal.clearance}).")


@app.get("/documents")
async def list_documents(principal: Principal = Depends(get_principal)):
    """Documents the caller can see, newest first. The first entry is the
    default query scope."""
    return {"documents": get_store().scoped(principal).list_documents()}


@app.patch("/documents/{doc_id}/access")
async def change_document_access(doc_id: str, change: AccessChange,
                                 principal: Principal = Depends(get_principal)):
    """Move a document to another team (or org-wide) and/or clearance; its
    chunks follow in the same statement. 404 if the caller can't see it."""
    team_id = change.team_id or None
    _check_can_grant(principal, team_id, change.clearance)
    try:
        await asyncio.to_thread(get_store().scoped(principal).set_document_access,
                                doc_id, team_id=team_id, clearance=change.clearance)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=f"No document {doc_id}.") from exc
    return {"doc_id": doc_id, "team_id": team_id, "clearance": change.clearance}


def _pipeline_error(exc: Exception, llm_usage: dict, diagnostics: bool) -> JSONResponse:
    """Structured failure for /query: which pipeline node failed and a
    provider-neutral error kind (raw provider messages stay in the server
    log). For admins (`diagnostics`) it also carries the full error text and
    every LLM attempt the request made, so the eval report can record quota
    ids and limits. LLM capacity errors are 503 (retry later); anything else
    is 500."""
    # LangGraph notes the failing node on the exception:
    # "During task with name 'generate' and id '...'".
    notes = " ".join(getattr(exc, "__notes__", []))
    match = re.search(r"During task with name '(\w+)'", notes)
    stage = match.group(1) if match else "pipeline"
    kind = classify_error(exc)
    logger.exception("Pipeline failed at %s (%s)", stage, kind or "internal")
    content = {"detail": "Pipeline failed.", "stage": stage, "error": kind or "internal"}
    if diagnostics:
        content |= {"message": f"{type(exc).__name__}: {exc}", "llm_usage": llm_usage}
    return JSONResponse(status_code=503 if kind else 500, content=content)


@app.post("/query")
async def query(request: QueryRequest, principal: Principal = Depends(get_principal)):
    """Non-streaming variant — used by the eval harness and handy for curl."""
    start = time.perf_counter()
    with track_usage() as llm_usage:
        try:
            state = await pipeline.ainvoke(_initial_state(request, principal))
        except Exception as exc:
            return _pipeline_error(exc, llm_usage, diagnostics=principal.role == "admin")
    total_ms = round((time.perf_counter() - start) * 1000, 1)

    query_id = await asyncio.to_thread(
        partial(db.log_query, _db_record(request, state, total_ms, principal), access=principal)
    )

    if state.get("outcome") == "rejected_input":
        raise HTTPException(status_code=422, detail=state["rejection"])

    response = {
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
    if principal.role == "admin":
        # Attempts per model and the full text of every failed attempt
        # (retried ones included), for the eval report.
        response["llm_usage"] = llm_usage
    return response


@app.post("/query/stream")
async def query_stream(request: QueryRequest, principal: Principal = Depends(get_principal)):
    async def event_stream():
        start = time.perf_counter()
        final: dict = {}
        try:
            async for update in pipeline.astream(
                _initial_state(request, principal), stream_mode="updates"
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
                partial(db.log_query, _db_record(request, final, total_ms, principal),
                        access=principal)
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


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest, principal: Principal = Depends(get_principal)):
    """Thumbs up/down (plus optional comment) for a logged query.

    Upserts: one feedback row per query, so re-clicking or adding a comment
    updates the verdict instead of double-counting it in /admin/stats.
    """
    status = await asyncio.to_thread(
        partial(db.log_feedback, request.query_id, request.thumbs_up, request.comment,
                access=principal, user_id=principal.user_id)
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


@app.get("/admin/logs")
async def admin_logs(limit: int = 20, principal: Principal = Depends(get_admin_principal)):
    """The admin's own org's logs: RLS hides every other org's rows."""
    return {"logs": await asyncio.to_thread(partial(db.fetch_logs, min(limit, 100),
                                                    access=principal))}


@app.get("/admin/stats")
async def admin_stats(principal: Principal = Depends(get_admin_principal)):
    return await asyncio.to_thread(partial(db.fetch_stats, access=principal))


@app.get("/me")
async def me(body: dict = Depends(get_me)):
    return body


@app.get("/health")
async def health():
    # No chunk count: /health is unauthenticated, and under RLS a count is
    # per caller anyway.
    return {"status": "ok"}
