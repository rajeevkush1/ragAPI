"""
api.py – FastAPI wrapper with SSE token streaming and MemorySaver persistence.

Architecture
─────────────────────────────────────────────────────────────────────
  POST /query          →  SSE stream OR JSON (stream=false)
  GET  /conversations/{thread_id}  →  full conversation history
  DELETE /conversations/{thread_id} →  clear a thread's memory
  GET  /threads        →  list all active thread IDs
  POST /ingest         →  upload + ingest a PDF
  GET  /collections    →  Qdrant collection info
  GET  /health         →  liveness + model availability
  GET  /docs           →  Swagger UI (built-in)

SSE event types (data: JSON\n\n)
─────────────────────────────────────────────────────────────────────
  {"type": "start",         "thread_id": ..., "turn": N}
  {"type": "node_start",    "node": "query_analyzer"}
  {"type": "token",         "content": "Flash", "node": "generator"}
  {"type": "node_complete", "node": "generator", "summary": {...}}
  {"type": "routing",       "from": "relevance_grader", "to": "record_failed_query"}
  {"type": "done",          "final_answer": "...", "metadata": {...}}
  {"type": "error",         "message": "..."}

Multi-turn memory
─────────────────────────────────────────────────────────────────────
  Each thread_id gets its own MemorySaver slot.
  After each successful run, the Q/A pair is appended to
  conversation_history in the checkpoint.
  On subsequent queries with the same thread_id, query_analyzer
  sees the history and can resolve follow-up references.

Run:
  python api.py                             # dev server on :8000
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except (AttributeError, TypeError):
    pass

import asyncio
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import AsyncIterator, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field

import config
from graph import build_graph, make_initial_state

# ── MemorySaver checkpointer (in-process, no external DB) ─────────────────────
import logging
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

_checkpointer = MemorySaver()
_graph        = build_graph(checkpointer=_checkpointer)

# ── Conversation history cap (prevents unbounded RAM growth) ──────────────────
MAX_HISTORY_TURNS = config.MAX_HISTORY_TURNS

# ── Upload size limit ─────────────────────────────────────────────────────────
MAX_PDF_MB = config.MAX_PDF_MB

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Advanced RAG API",
    description=(
        "LangGraph RAG pipeline · SSE token streaming · "
        "MemorySaver conversation memory · Qdrant + BGE-M3 + Ollama"
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

# ── Node names we care about for event filtering ───────────────────────────────
_PIPELINE_NODES = {
    "query_analyzer", "vector_retriever", "relevance_grader",
    "generator", "hallucination_checker", "direct_answer",
    "no_context", "record_failed_query", "prepare_strict_gen",
}

# ── Request / Response models ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question:  str  = Field(..., min_length=1, max_length=4000,
                            example="What is Flash Attention?")
    thread_id: str  = Field(default_factory=lambda: str(uuid.uuid4()),
                            description="Reuse to continue a conversation")
    stream:    bool = Field(True, description="Stream via SSE or return JSON")


class ConversationMessage(BaseModel):
    role:    str
    content: str


class QueryResponse(BaseModel):
    thread_id:  str
    question:   str
    answer:     str
    sources:    list[dict]
    grounded:   bool
    confidence: float
    latency_ms: Optional[float] = None


class ChunkInfo(BaseModel):
    chunk_id: str
    text:     str
    h1:       str
    h2:       str


class IngestResponse(BaseModel):
    filename:   str
    chunks:     int
    status:     str
    embed_secs: float
    chunk_list: list[ChunkInfo] = []


# ═══════════════════════════════════════════════════════════════════════════════
# SSE STREAMING CORE
# ═══════════════════════════════════════════════════════════════════════════════

def _sse(data: dict) -> str:
    """Format a dict as a Server-Sent Event line."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _node_summary(node: str, output: dict) -> dict:
    """Extract a concise, serialisable summary from a node's output dict."""
    if not isinstance(output, dict):
        return {}
    match node:
        case "query_analyzer":
            return {
                "rewritten_query": output.get("rewritten_query", ""),
                "query_type":      output.get("query_type", ""),
                "needs_context":   output.get("needs_context", True),
                "key_terms":       output.get("key_terms", []),
            }
        case "vector_retriever":
            return {"docs_retrieved": len(output.get("retrieved_docs", []))}
        case "relevance_grader":
            return {"docs_graded": len(output.get("graded_docs", []))}
        case "generator":
            ans = output.get("answer", "")
            return {"answer_chars": len(ans), "preview": ans[:120]}
        case "hallucination_checker":
            return {
                "is_grounded":        bool(output.get("is_grounded", False)),
                "grounding_score":    round(float(output.get("grounding_score", 0)), 3),
                "unsupported_claims": len(output.get("unsupported_claims", [])),
            }
        case "record_failed_query":
            return {
                "retrieval_retry": output.get("retrieval_retry_count", 0),
                "failed_queries":  output.get("failed_queries", []),
            }
        case "prepare_strict_gen":
            return {
                "generation_retry": output.get("generation_retry_count", 0),
                "hint_preview":     output.get("generation_hint", "")[:80],
            }
        case _:
            return {}


async def _stream_graph(question: str, thread_id: str) -> AsyncIterator[str]:
    """
    Async generator that runs the LangGraph pipeline and yields SSE events.

    Event sequence:
      start → node_start* → token* → node_complete* → done
                    ↑_______ (retry loops) ___________|
    """
    config_dict = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 60,       # safety net above our 3+3 retry caps
    }

    # ── Load existing conversation history from checkpoint ────────────────────
    try:
        snapshot      = _graph.get_state(config_dict)
        saved_values  = snapshot.values if snapshot else {}
        conv_history  = list(saved_values.get("conversation_history", []))
    except Exception:
        conv_history  = []

    turn = len(conv_history) // 2 + 1   # 1-indexed turn number
    yield _sse({"type": "start", "thread_id": thread_id, "turn": turn})

    # ── Build per-turn state (reset ephemeral fields, carry forward history) ──
    state = make_initial_state(question)
    state["conversation_history"] = conv_history

    # ── Track which pipeline node is currently active ─────────────────────────
    current_node: str = ""
    final_answer: str = ""
    pipeline_state: dict = {}

    try:
        async for event in _graph.astream_events(
            state,
            config=config_dict,
            version="v2",
            include_names=list(_PIPELINE_NODES),   # filter: only our nodes
        ):
            kind = event["event"]
            name = event.get("name", "")
            meta = event.get("metadata", {})
            # LangGraph annotates each event with the active node name
            lg_node = meta.get("langgraph_node", "")

            # ── Node lifecycle ─────────────────────────────────────────────────
            if kind == "on_chain_start" and name in _PIPELINE_NODES:
                current_node = name
                yield _sse({"type": "node_start", "node": name})

            elif kind == "on_chain_end" and name in _PIPELINE_NODES:
                out     = event.get("data", {}).get("output") or {}
                summary = _node_summary(name, out)
                yield _sse({"type": "node_complete", "node": name, "summary": summary})
                if isinstance(out, dict):
                    pipeline_state.update(out)
                    if out.get("final_answer"):
                        final_answer = out["final_answer"]

            # ── Token-level streaming from LLM ─────────────────────────────────
            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    # Only forward tokens that come from the generator or
                    # direct_answer node — not from query_analyzer or grader
                    # (those are classification/scoring, not worth streaming)
                    emitting_node = lg_node or current_node
                    if emitting_node in ("generator", "direct_answer", ""):
                        yield _sse({
                            "type":    "token",
                            "content": chunk.content,
                            "node":    emitting_node or "generator",
                        })

            # ── Routing events (conditional edge taken) ────────────────────────
            elif kind == "on_chain_end" and name == "LangGraph":
                # Top-level graph end — the whole run completed
                out = event.get("data", {}).get("output") or {}
                if isinstance(out, dict) and out.get("final_answer"):
                    final_answer = out["final_answer"]
                    pipeline_state.update(out)

    except asyncio.CancelledError:
        yield _sse({"type": "cancelled", "thread_id": thread_id})
        return
    except Exception as exc:
        yield _sse({"type": "error", "message": str(exc), "thread_id": thread_id})
        return

    # ── Grab authoritative final state from checkpoint ────────────────────────
    try:
        snapshot = _graph.get_state(config_dict)
        final    = snapshot.values if snapshot else pipeline_state
    except Exception:
        final = pipeline_state

    if not final_answer:
        final_answer = final.get("final_answer", "")

    # ── Append this turn to conversation history ──────────────────────────────
    new_history = list(final.get("conversation_history", conv_history))
    new_history.append({"role": "user",      "content": question})
    new_history.append({"role": "assistant", "content": final_answer})
    # Cap history to prevent unbounded RAM growth in long sessions
    if len(new_history) > MAX_HISTORY_TURNS * 2:
        new_history = new_history[-(MAX_HISTORY_TURNS * 2):]

    try:
        _graph.update_state(config_dict, {"conversation_history": new_history})
    except Exception:
        pass  # non-fatal

    # Format sources for UI
    sources = [
        {
            "source": d.get("source"),
            "chunk_id": d.get("chunk_id"),
            "title": f"{d.get('h1', '')} > {d.get('h2', '')}".strip(" >"),
            "relevance_score": round(d.get("relevance_score", 0.0), 3),
            "text": d.get("text", "")
        }
        for d in final.get("graded_docs", [])
    ]

    # ── Final "done" event with full metadata ─────────────────────────────────
    yield _sse({
        "type":         "done",
        "final_answer": final_answer,
        "sources":      sources,
        "grounded":     bool(final.get("is_grounded", False)),
        "confidence":   round(float(final.get("grounding_score", 0.0)), 3),
        "metadata": {
            "thread_id":          thread_id,
            "turn":               turn,
            "query_type":         final.get("query_type", "unknown"),
            "rewritten_query":    final.get("rewritten_query", ""),
            "retrieval_retries":  final.get("retrieval_retry_count", 0),
            "generation_retries": final.get("generation_retry_count", 0),
            "failed_queries":     final.get("failed_queries", []),
            "docs_retrieved":     len(final.get("retrieved_docs", [])),
            "docs_graded":        len(final.get("graded_docs", [])),
            "conversation_turns": len(new_history) // 2,
        },
    })



# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    """Liveness check — verifies Qdrant and reports model config."""
    from qdrant_client import QdrantClient
    try:
        QdrantClient(url=config.QDRANT_URL, timeout=3).get_collections()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False

    from nodes.query_analyzer import _ANALYZER_MODEL
    from nodes.relevance_grader import _GRADER_MODEL
    from nodes.generator import _GENERATOR_MODEL
    from nodes.hallucination_checker import _CHECKER_MODEL

    return {
        "status":         "ok" if qdrant_ok else "degraded",
        "qdrant":         "up" if qdrant_ok else "down",
        "collection":     config.COLLECTION_NAME,
        "embed_model":    config.EMBED_MODEL,
        "memory_backend": "MemorySaver (in-process)",
        "models": {
            "query_analyzer":        _ANALYZER_MODEL,
            "relevance_grader":      _GRADER_MODEL,
            "generator":             _GENERATOR_MODEL,
            "hallucination_checker": _CHECKER_MODEL,
        },
    }


# ── Collections ───────────────────────────────────────────────────────────────
@app.get("/collections", tags=["System"])
async def list_collections():
    """List Qdrant collections and their point counts."""
    from qdrant_client import QdrantClient
    try:
        client = QdrantClient(url=config.QDRANT_URL)
        cols   = client.get_collections().collections
        result = []
        for col in cols:
            info = client.get_collection(col.name)
            result.append({
                "name":         col.name,
                "points_count": info.points_count,
                "vector_size":  info.config.params.vectors.size,
            })
        return {"collections": result}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Qdrant unreachable: {exc}")


# ── UI Route ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, tags=["UI"])
def get_ui():
    """Serves the premium React chat UI at the root address."""
    html_path = Path(__file__).parent / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ── Query (SSE streaming or JSON) ──────────────────────────────────────────────
@app.get("/query", tags=["RAG"])
async def query_endpoint_get(
    question: str,
    thread_id: Optional[str] = None,
    stream: bool = True
):
    """
    Run the RAG pipeline using GET (useful for native browser EventSource API).
    """
    if not thread_id:
        thread_id = str(uuid.uuid4())
    if stream:
        return StreamingResponse(
            _stream_graph(question, thread_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control":   "no-cache",
                "X-Accel-Buffering": "no",
                "Connection":      "keep-alive",
            },
        )
    else:
        req = QueryRequest(question=question, thread_id=thread_id, stream=False)
        return await query_endpoint(req)


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_endpoint(req: QueryRequest):
    """
    Run the RAG pipeline.

    - **stream=true** (default): returns `text/event-stream` SSE.
      Each line is `data: <JSON>\\n\\n`. Listen for `type="done"` to know when finished.
    - **stream=false**: blocks and returns a single JSON response.

    Pass the same **thread_id** across calls to continue a conversation.
    Omit it (or generate a new UUID) to start a fresh thread.
    """
    if req.stream:
        return StreamingResponse(
            _stream_graph(req.question, req.thread_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control":   "no-cache",
                "X-Accel-Buffering": "no",       # disable Nginx buffering
                "Connection":      "keep-alive",
            },
        )

    # Non-streaming path: run synchronously via thread
    cfg = {"configurable": {"thread_id": req.thread_id}, "recursion_limit": 60}
    t0 = time.perf_counter()
    try:
        # Load history first
        try:
            snap = _graph.get_state(cfg)
            hist = list(snap.values.get("conversation_history", [])) if snap else []
        except Exception:
            hist = []

        state = make_initial_state(req.question)
        state["conversation_history"] = hist

        result = await asyncio.to_thread(_graph.invoke, state, cfg)

        # Persist history (capped to avoid unbounded RAM growth)
        new_hist = list(result.get("conversation_history", hist))
        new_hist.append({"role": "user",      "content": req.question})
        new_hist.append({"role": "assistant",  "content": result.get("final_answer", "")})
        if len(new_hist) > MAX_HISTORY_TURNS * 2:
            new_hist = new_hist[-(MAX_HISTORY_TURNS * 2):]
        try:
            _graph.update_state(cfg, {"conversation_history": new_hist})
        except Exception:
            pass

        sources = [
            {
                "source": d.get("source"),
                "chunk_id": d.get("chunk_id"),
                "title": f"{d.get('h1', '')} > {d.get('h2', '')}".strip(" >"),
                "relevance_score": round(d.get("relevance_score", 0.0), 3),
                "text": d.get("text", "")
            }
            for d in result.get("graded_docs", [])
        ]

        latency = (time.perf_counter() - t0) * 1000

        return QueryResponse(
            thread_id=req.thread_id,
            question=req.question,
            answer=result.get("final_answer", ""),
            sources=sources,
            grounded=bool(result.get("is_grounded", False)),
            confidence=round(float(result.get("grounding_score", 0.0)), 3),
            latency_ms=round(latency, 1)
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))



# ── Conversation memory ───────────────────────────────────────────────────────
@app.get("/conversations/{thread_id}", tags=["Memory"])
def get_conversation(thread_id: str):
    """Return the full conversation history for a thread."""
    cfg = {"configurable": {"thread_id": thread_id}}
    try:
        snap = _graph.get_state(cfg)
        if not snap or not snap.values:
            raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found")
        vals = snap.values
        return {
            "thread_id":            thread_id,
            "conversation_turns":   len(vals.get("conversation_history", [])) // 2,
            "conversation_history": vals.get("conversation_history", []),
            "last_query":           vals.get("original_query", ""),
            "last_answer":          vals.get("final_answer", ""),
            "last_grounding_score": vals.get("grounding_score", 0.0),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/conversations/{thread_id}", tags=["Memory"])
def delete_conversation(thread_id: str):
    """Clear all memory for a thread (start fresh with same thread_id)."""
    cfg = {"configurable": {"thread_id": thread_id}}
    try:
        _graph.update_state(cfg, {"conversation_history": []})
        return {"status": "cleared", "thread_id": thread_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/threads", tags=["Memory"])
def list_threads():
    """List all thread IDs that have an active checkpoint in memory."""
    try:
        # MemorySaver.storage is an internal implementation detail — not a public API.
        # A LangGraph version bump may silently break this; the except branch handles it.
        thread_ids = list({
            ns[0]
            for ns in _checkpointer.storage.keys()
            if isinstance(ns, tuple) and len(ns) >= 1
        })
        return {"threads": thread_ids, "count": len(thread_ids)}
    except Exception as exc:
        logger.warning(
            "Cannot introspect MemorySaver.storage — LangGraph API may have changed: %s", exc
        )
        return {"threads": [], "count": 0, "note": "Cannot introspect checkpointer storage"}


# ── Documents ─────────────────────────────────────────────────────────────────
@app.get("/documents", tags=["System"])
def list_documents():
    """List all ingested PDFs and their chunk counts in Qdrant."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    try:
        client = QdrantClient(url=config.QDRANT_URL)
        existing = {c.name for c in client.get_collections().collections}
        if config.COLLECTION_NAME not in existing:
            return {"documents": [], "count": 0}

        scroll_res = client.scroll(
            collection_name=config.COLLECTION_NAME,
            limit=1000,
            with_payload=["source"],
            with_vectors=False
        )
        points = scroll_res[0]
        sources = set()
        for p in points:
            if p.payload and "source" in p.payload:
                sources.add(p.payload["source"])

        results = []
        for src in sorted(sources):
            count = client.count(
                collection_name=config.COLLECTION_NAME,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=src)
                        )
                    ]
                )
            ).count
            results.append({
                "filename": src,
                "chunks": count
            })

        return {"documents": results, "count": len(results)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/documents/{filename}", tags=["System"])
def delete_document(filename: str):
    """Delete all chunks for a specific document from Qdrant and delete its local files."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    try:
        client = QdrantClient(url=config.QDRANT_URL)
        # 1. Delete points matching the filename source from Qdrant
        client.delete(
            collection_name=config.COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=filename)
                    )
                ]
            )
        )
        # 2. Delete local PDF file
        pdf_path = config.PDF_DIR / filename
        pdf_path.unlink(missing_ok=True)
        # 3. Delete parsed markdown cache file
        parsed_path = config.PARSED_DIR / f"{filename}.mmd"
        parsed_path.unlink(missing_ok=True)
        return {"status": "deleted", "filename": filename}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/documents", tags=["System"])
def delete_all_documents():
    """Clear all documents in the Qdrant collection and delete all local PDFs/caches."""
    from qdrant_client import QdrantClient
    try:
        client = QdrantClient(url=config.QDRANT_URL)
        existing = {c.name for c in client.get_collections().collections}
        if config.COLLECTION_NAME in existing:
            client.delete_collection(config.COLLECTION_NAME)
        # Clear local files
        for f in config.PDF_DIR.glob("*.pdf"):
            f.unlink(missing_ok=True)
        for f in config.PARSED_DIR.glob("*.mmd"):
            f.unlink(missing_ok=True)
        return {"status": "cleared_all"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))



@app.get("/documents/{filename}/chunks", tags=["System"])
def list_document_chunks(filename: str):
    """Retrieve all raw chunks for a specific ingested document."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    try:
        client = QdrantClient(url=config.QDRANT_URL)
        scroll_res = client.scroll(
            collection_name=config.COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=filename)
                    )
                ]
            ),
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        points = scroll_res[0]
        
        chunks = []
        for p in points:
            pl = p.payload or {}
            chunks.append({
                "chunk_id": pl.get("chunk_id", str(p.id)),
                "text": pl.get("text", ""),
                "h1": pl.get("h1", ""),
                "h2": pl.get("h2", ""),
                "chunk_index": pl.get("chunk_index", 0)
            })
        
        chunks.sort(key=lambda c: c["chunk_index"])
        return {
            "filename": filename,
            "chunks_count": len(chunks),
            "chunks": chunks
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── PDF Ingestion ─────────────────────────────────────────────────────────────
@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload and ingest a PDF into Qdrant.

    Parses with pymupdf4llm, splits on Markdown headers, embeds with BGE-M3,
    and upserts vectors into Qdrant. Runs synchronously (returns when done).
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    from ingest import ingest_pdf

    content = await file.read()
    if len(content) > MAX_PDF_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_PDF_MB} MB.",
        )

    # Sanitise filename — strips any directory components to prevent path traversal
    safe_name = Path(file.filename).name
    pdf_path  = config.PDF_DIR / safe_name

    pdf_path.write_bytes(content)

    t0 = time.perf_counter()
    try:
        stats = await asyncio.to_thread(ingest_pdf, pdf_path)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        # Clean up the file if ingestion fails so we don't leave corrupt/failed PDFs
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    return IngestResponse(
        filename=file.filename,
        chunks=stats.get("chunks", 0),
        status="ingested",
        embed_secs=round(time.perf_counter() - t0, 2),
        chunk_list=stats.get("chunk_list", []),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DEV SERVER
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,          # reload=True breaks shared state (MemorySaver)
        log_level="info",
    )
