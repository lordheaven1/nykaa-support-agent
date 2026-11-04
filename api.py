"""
api.py
FastAPI production application exposing /ask, /add-document, and /health endpoints.
Implements Pydantic request/response contracts, single-instance graph execution,
and mandatory JSONL request auditing.
"""

from contextlib import asynccontextmanager
import time
from typing import Any, Dict, List, Literal, Optional
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agent import run_agent
from chunking import fixed_size_chunks, sentence_chunks
from logging_utils import log_request
import rag_core
from schema import validate_response


# ==========================================
# Pydantic Schemas
# ==========================================

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Customer inquiry")
    conversation_id: Optional[str] = Field(None, description="Thread identifier for multi-turn memory")


class AskResponse(BaseModel):
    trace_id: str
    route: Literal["rag", "order_status", "refused"]
    answer: str
    sources: List[str]
    grounded: bool
    guardrails_triggered: List[str]
    data: Optional[Dict[str, Any]] = None


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(..., pattern=r"^[a-z0-9_]+$", description="Snake_case unique document identifier")
    text: str = Field(..., min_length=10, description="Policy text content to index")


class AddDocumentResponse(BaseModel):
    doc_id: str
    chunks_added: int
    collection: str


# ==========================================
# FastAPI Lifespan & App Setup
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly initialize singletons at startup
    rag_core.get_embedding_model()
    rag_core.get_chroma_client()
    yield

app = FastAPI(
    title="Nykaa Support Agent API",
    version="1.0.0",
    description="Domain-specific customer support API with LangGraph orchestration and grounded retrieval",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Logs 422 validation errors through the audit logger before returning response."""
    trace_id = f"tr-{uuid.uuid4().hex[:12]}"
    log_request(
        endpoint=request.url.path,
        status_code=422,
        duration_ms=0.5,
        trace_id=trace_id,
        raw_request_text=str(exc.body) if hasattr(exc, "body") else "",
        error_message=str(exc.errors()),
    )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


# ==========================================
# Endpoints
# ==========================================

@app.get("/health")
def health() -> Dict[str, str]:
    t0 = time.perf_counter()
    status = {"status": "ok"}
    dur = (time.perf_counter() - t0) * 1000.0
    log_request(endpoint="/health", status_code=200, duration_ms=dur, trace_id=f"tr-{uuid.uuid4().hex[:12]}")
    return status


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    t0 = time.perf_counter()
    trace_id = f"tr-{uuid.uuid4().hex[:12]}"
    conv_id = payload.conversation_id or f"conv-{uuid.uuid4().hex[:8]}"

    try:
        agent_out = run_agent(
            query=payload.query,
            conversation_id=conv_id,
            trace_id=trace_id,
        )
        validate_response(agent_out)

        dur = (time.perf_counter() - t0) * 1000.0
        log_request(
            endpoint="/ask",
            status_code=200,
            duration_ms=dur,
            trace_id=trace_id,
            conversation_id=conv_id,
            raw_request_text=payload.query,
            route=agent_out["route"],
            guardrails_triggered=agent_out.get("guardrails_triggered", []),
            grounded=agent_out.get("grounded"),
            answer_length=len(agent_out.get("answer", "")),
        )
        return AskResponse(**agent_out)

    except Exception as exc:
        dur = (time.perf_counter() - t0) * 1000.0
        log_request(
            endpoint="/ask",
            status_code=500,
            duration_ms=dur,
            trace_id=trace_id,
            conversation_id=conv_id,
            raw_request_text=payload.query,
            error_message=str(exc),
        )
        raise HTTPException(status_code=500, detail=f"Internal agent execution error: {str(exc)}")


@app.post("/add-document", response_model=AddDocumentResponse)
def add_document(payload: AddDocumentRequest) -> AddDocumentResponse:
    t0 = time.perf_counter()
    trace_id = f"tr-{uuid.uuid4().hex[:12]}"
    client = rag_core.get_chroma_client()
    embedder = rag_core.get_embedding_model()

    coll_sentence = client.get_collection(name="kb_sentence")
    coll_fixed = client.get_collection(name="kb_fixed")

    # Safety rule: check if doc_id already exists in metadata
    existing = coll_sentence.get(where={"doc_id": payload.doc_id})
    if existing and existing["ids"]:
        dur = (time.perf_counter() - t0) * 1000.0
        log_request(
            endpoint="/add-document",
            status_code=409,
            duration_ms=dur,
            trace_id=trace_id,
            raw_request_text=f"doc_id={payload.doc_id}",
            error_message="Document already exists",
        )
        raise HTTPException(status_code=409, detail=f"Document '{payload.doc_id}' already exists in KB collections.")

    try:
        # 1. Add to kb_sentence
        s_chunks = sentence_chunks(payload.text)
        s_ids = [f"{payload.doc_id}::sentence::{i}" for i in range(len(s_chunks))]
        s_meta = [{"doc_id": payload.doc_id, "chunk_index": i, "strategy": "sentence"} for i in range(len(s_chunks))]
        s_embeddings = embedder.encode(s_chunks).tolist()
        coll_sentence.add(ids=s_ids, documents=s_chunks, metadatas=s_meta, embeddings=s_embeddings)

        # 2. Add to kb_fixed to keep collections in sync
        f_chunks = fixed_size_chunks(payload.text, chunk_size=200, overlap=40)
        f_ids = [f"{payload.doc_id}::fixed::{i}" for i in range(len(f_chunks))]
        f_meta = [{"doc_id": payload.doc_id, "chunk_index": i, "strategy": "fixed"} for i in range(len(f_chunks))]
        f_embeddings = embedder.encode(f_chunks).tolist()
        coll_fixed.add(ids=f_ids, documents=f_chunks, metadatas=f_meta, embeddings=f_embeddings)

        dur = (time.perf_counter() - t0) * 1000.0
        log_request(
            endpoint="/add-document",
            status_code=200,
            duration_ms=dur,
            trace_id=trace_id,
            raw_request_text=f"doc_id={payload.doc_id}",
            answer_length=len(s_chunks),
        )

        return AddDocumentResponse(
            doc_id=payload.doc_id,
            chunks_added=len(s_chunks),
            collection="kb_sentence",
        )

    except Exception as exc:
        dur = (time.perf_counter() - t0) * 1000.0
        log_request(
            endpoint="/add-document",
            status_code=500,
            duration_ms=dur,
            trace_id=trace_id,
            raw_request_text=f"doc_id={payload.doc_id}",
            error_message=str(exc),
        )
        raise HTTPException(status_code=500, detail=str(exc))
