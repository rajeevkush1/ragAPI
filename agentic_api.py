"""
agentic_api.py – FastAPI server for the Agentic RAG graph with SSE streaming on port 8001.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
import re
from pathlib import Path
from typing import AsyncIterator, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

import config
from agentic_graph import build_agent_graph, get_llm, get_main_llm, get_fallback_llm

app = FastAPI(
    title="Agentic RAG API",
    description="Agentic LangGraph RAG pipeline using Pydantic AgentState and Dynamic Model Routing.",
    version="1.0.0",
    docs_url="/docs",
)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    config.logger.info("Pre-warming FastEmbed model at startup...")
    try:
        from ingest import get_embed_model
        await asyncio.to_thread(get_embed_model, config.EMBED_MODEL)
        config.logger.info("FastEmbed model pre-warmed successfully!")
    except Exception as exc:
        config.logger.warning(f"FastEmbed pre-warm notice: {exc}")

_checkpointer = MemorySaver()
_graph = build_agent_graph(checkpointer=_checkpointer)

class QueryRequest(BaseModel):
    question: str = Field(..., json_schema_extra={"example": "What is Flash Attention?"})
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    stream: bool = Field(True)
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    retrieval_strategy: str = Field(default="hybrid")
    top_k: int = Field(default=5)
    context_limit: int = Field(default=4000)
    temperature: float = Field(default=0.2)
    provider: Optional[str] = Field(default=None)
    api_key: Optional[str] = Field(default=None)
    model: Optional[str] = Field(default=None)

class QueryResponse(BaseModel):
    thread_id: str
    question: str
    answer: str
    latency_ms: float
    sources: list[dict] = []
    grounded: bool = True
    confidence: float = 0.95

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

def parse_citations_and_diagnostics(messages: list) -> tuple[list[dict], bool, float, str]:
    """Parse messages list to extract rich source citations and judge diagnostics."""
    sources = []
    grounded = True
    confidence = 0.95
    judge_reason = "No evaluation metadata available."
    
    # 1. Parse ToolMessages for document chunks
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "retrieve_research_papers":
            content = msg.content
            # Split chunks by the doc separator
            docs_raw = content.split("\n\n---\n\n")
            for doc_raw in docs_raw:
                if doc_raw.startswith("[Doc"):
                    lines = doc_raw.split("\n", 1)
                    if len(lines) == 2:
                        header, body = lines[0], lines[1]
                        source_match = re.search(r"Source:\s*([^|]+)", header)
                        title_match = re.search(r"Title:\s*([^|]+)", header)
                        authors_match = re.search(r"Authors:\s*([^|]+)", header)
                        venue_match = re.search(r"Venue:\s*([^|]+)", header)
                        rel_match = re.search(r"Relevance:\s*([0-9.]+)", header)
                        
                        source_name = source_match.group(1).strip() if source_match else "unknown"
                        title_name = title_match.group(1).strip() if title_match else ""
                        authors_name = authors_match.group(1).strip() if authors_match else ""
                        venue_name = venue_match.group(1).strip() if venue_match else ""
                        rel_score = float(rel_match.group(1).strip()) if rel_match else 0.95
                        
                        if body.startswith("Content: "):
                            body = body[9:]
                            
                        # Avoid duplicates
                        if not any(s["text"] == body for s in sources):
                            sources.append({
                                "source": f"{source_name} ({title_name})" if title_name else source_name,
                                "raw_source": source_name,
                                "title": title_name or source_name,
                                "authors": authors_name or "Research Authors",
                                "venue": venue_name or "arXiv preprint",
                                "text": body,
                                "relevance_score": rel_score
                            })
                            
    # 2. Parse final AIMessage diagnostics (attached by judge_node)
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            diag = msg.additional_kwargs.get("diagnostics", {})
            if diag:
                grounded = diag.get("grounded", True)
                confidence = diag.get("confidence", 0.95)
                judge_reason = diag.get("judge_reason", "Evaluation complete.")
            break
            
    return sources, grounded, confidence, judge_reason

async def _stream_graph(
    question: str, 
    thread_id: str, 
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    retrieval_strategy: str = "hybrid",
    top_k: int = 5,
    context_limit: int = 4000,
    temperature: float = 0.2,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> AsyncIterator[str]:
    """Async generator running the Agentic RAG graph and yielding SSE events."""
    config_dict = {
        "configurable": {
            "thread_id": thread_id,
            "embedding_model": embedding_model,
            "retrieval_strategy": retrieval_strategy,
            "top_k": top_k,
            "context_limit": context_limit,
            "temperature": temperature,
            "provider": provider,
            "api_key": api_key,
            "model": model,
        },
        "recursion_limit": 50,
    }
    
    start_time = time.time()
    config.logger.info(f"[SSE Stream] Question: '{question}' | thread_id: '{thread_id}'")
    yield _sse({"type": "start", "thread_id": thread_id, "turn": "agentic"})
    
    # We initialize the state with a single HumanMessage containing the user's question
    state = {"messages": [HumanMessage(content=question)]}
    
    current_node = ""
    final_answer = ""
    
    try:
        # Stream events from LangGraph
        async for event in _graph.astream_events(
            state,
            config=config_dict,
            version="v2",
        ):
            kind = event["event"]
            name = event.get("name", "")
            meta = event.get("metadata", {})
            lg_node = meta.get("langgraph_node", "")
            
            # Node start/complete events
            if kind == "on_chain_start" and name in ("agent", "tools"):
                current_node = name
                yield _sse({"type": "node_start", "node": name})
                
            elif kind == "on_chain_end" and name in ("agent", "tools"):
                yield _sse({"type": "node_complete", "node": name, "summary": {}})
                
            # Token streaming from chat model
            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    emitting_node = lg_node or current_node
                    if emitting_node == "agent":
                        yield _sse({
                            "type": "token",
                            "content": chunk.content,
                            "node": "agent"
                        })
                        
    except asyncio.CancelledError:
        config.logger.warning(f"[SSE Stream] Cancelled for thread_id: '{thread_id}'")
        yield _sse({"type": "cancelled", "thread_id": thread_id})
        return
    except Exception as exc:
        config.logger.error(f"[SSE Stream] Exception for thread_id '{thread_id}': {exc}")
        err_str = str(exc)
        err_lower = err_str.lower()
        if any(k in err_lower for k in ["401", "invalid_api_key", "invalid api key", "unauthorized", "user not found", "authentication"]):
            friendly_msg = (
                "⚠️ **LLM API Key Authentication Error (401 - Invalid API Key)**\n\n"
                "The configured API key was rejected by the LLM provider.\n\n"
                "### 🔑 How to resolve:\n"
                "1. Tap the **⚙️ Settings** icon in the upper-right corner of this chat.\n"
                "2. Select your provider (**OpenRouter**, **Groq**, or **Google Gemini**) and paste your valid API Key.\n"
                "3. Or add `OPENROUTER_API_KEY=your_key` (or `GROQ_API_KEY`, `GEMINI_API_KEY`) to your `.env` file.\n"
                "4. Or run local Ollama (`ollama pull llama3.2:3b`) for offline model fallback.\n\n"
                "*Document ingestion and vector search indexed in your session remain completely active!*"
            )
            yield _sse({"type": "token", "content": friendly_msg, "node": "agent"})
            yield _sse({
                "type": "done",
                "final_answer": friendly_msg,
                "sources": [],
                "grounded": False,
                "confidence": 0.0,
                "latency": round(time.time() - start_time, 2)
            })
        else:
            yield _sse({"type": "error", "message": err_str, "thread_id": thread_id})
        return
        
    # Retrieve final execution output from graph state
    sources = []
    grounded = True
    confidence = 0.95
    judge_reason = ""
    try:
        snapshot = _graph.get_state(config_dict)
        messages = snapshot.values.get("messages", [])
        if messages:
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    final_answer = msg.content
                    break
            sources, grounded, confidence, judge_reason = parse_citations_and_diagnostics(messages)
    except Exception:
        pass

    elapsed = round(time.time() - start_time, 2)
    config.logger.info(f"[SSE Stream Completed] thread_id: '{thread_id}' | latency: {elapsed}s | grounded: {grounded} | sources: {len(sources)}")
        
    yield _sse({
        "type": "done",
        "final_answer": final_answer,
        "sources": sources,
        "grounded": grounded,
        "confidence": confidence,
        "latency": elapsed,
        "used_chunks_count": len(sources),
        "metadata": {
            "thread_id": thread_id,
            "agent_node": "agent",
            "query_type": "vector",
            "judge_reason": judge_reason,
            "retrieval_strategy": retrieval_strategy,
            "top_k": top_k,
            "context_limit": context_limit,
            "temperature": temperature,
        }
    })

@app.get("/health", tags=["System"])
async def health():
    """System health check and dynamic model status (Nemotron Main LLM + Ollama Fallback)."""
    main_llm_inst = get_main_llm()
    fallback_llm_inst = get_fallback_llm()

    main_model_name = getattr(main_llm_inst, "model_name", getattr(main_llm_inst, "model", "none")) if main_llm_inst else "none (Ollama active)"
    fallback_model_name = getattr(fallback_llm_inst, "model", getattr(fallback_llm_inst, "model_name", config.OLLAMA_MODEL))
    
    import os
    active_port = int(os.getenv("PORT", "8000"))
    return {
        "status": "ok",
        "main_llm": {
            "model": main_model_name,
            "provider": "NVIDIA / OpenRouter (Nemotron)" if main_llm_inst else "disabled",
            "active": main_llm_inst is not None
        },
        "fallback_llm": {
            "model": fallback_model_name,
            "provider": "Ollama (local)",
            "active": True
        },
        "port": active_port
    }

@app.get("/query", tags=["RAG"])
async def query_endpoint_get(
    question: str,
    thread_id: Optional[str] = None,
    stream: bool = True,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    retrieval_strategy: str = "hybrid",
    top_k: int = 5,
    context_limit: int = 4000,
    temperature: float = 0.2,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
):
    """Run the Agentic RAG pipeline using GET (useful for browser EventSource API)."""
    if not thread_id:
        thread_id = str(uuid.uuid4())
    if stream:
        return StreamingResponse(
            _stream_graph(
                question, 
                thread_id, 
                embedding_model,
                retrieval_strategy,
                top_k,
                context_limit,
                temperature,
                provider,
                api_key,
                model
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control":   "no-cache",
                "X-Accel-Buffering": "no",
                "Connection":      "keep-alive",
            },
        )
    else:
        req = QueryRequest(
            question=question, 
            thread_id=thread_id, 
            stream=False, 
            embedding_model=embedding_model,
            retrieval_strategy=retrieval_strategy,
            top_k=top_k,
            context_limit=context_limit,
            temperature=temperature,
            provider=provider,
            api_key=api_key,
            model=model
        )
        return await query_endpoint(req)

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_endpoint(req: QueryRequest):
    """Query the Agentic RAG chain. Supports SSE streaming or standard JSON response."""
    if req.stream:
        return StreamingResponse(
            _stream_graph(
                req.question, 
                req.thread_id, 
                req.embedding_model,
                req.retrieval_strategy,
                req.top_k,
                req.context_limit,
                req.temperature,
                req.provider,
                req.api_key,
                req.model
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )
        
    # Non-streaming path
    cfg = {
        "configurable": {
            "thread_id": req.thread_id,
            "embedding_model": req.embedding_model,
        },
        "recursion_limit": 50,
    }
    t0 = time.perf_counter()
    try:
        state = {"messages": [HumanMessage(content=req.question)]}
        result = await asyncio.to_thread(_graph.invoke, state, cfg)
        
        messages = result.get("messages", [])
        answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                answer = msg.content
                break
                
        sources, grounded, confidence, _ = parse_citations_and_diagnostics(messages)
        latency = (time.perf_counter() - t0) * 1000
        return QueryResponse(
            thread_id=req.thread_id,
            question=req.question,
            answer=answer,
            latency_ms=round(latency, 1),
            sources=sources,
            grounded=grounded,
            confidence=confidence
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/conversations/{thread_id}", tags=["Memory"])
def get_conversation(thread_id: str):
    """Retrieve full message history for a thread."""
    cfg = {"configurable": {"thread_id": thread_id}}
    try:
        snap = _graph.get_state(cfg)
        if not snap or not snap.values:
            raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found")
        messages = snap.values.get("messages", [])
        
        history = []
        for msg in messages:
            role = "unknown"
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, ToolMessage):
                role = "tool"
                
            history.append({
                "role": role,
                "content": msg.content,
                "additional_kwargs": getattr(msg, "additional_kwargs", {})
            })
            
        return {
            "thread_id": thread_id,
            "conversation_history": history,
            "turns_count": len(history)
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.delete("/conversations/{thread_id}", tags=["Memory"])
def delete_conversation(thread_id: str):
    """Clear memory and purge temporary session vectors for a specific thread_id."""
    from ingest import delete_session_data
    cfg = {"configurable": {"thread_id": thread_id}}
    try:
        _graph.update_state(cfg, {"messages": []})
        delete_session_data(thread_id)
        return {"status": "cleared", "thread_id": thread_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.delete("/sessions/{session_id}", tags=["Memory"])
def delete_session(session_id: str):
    """Purge all vectors and temporary files associated with session_id."""
    from ingest import delete_session_data
    try:
        delete_session_data(session_id)
        return {"status": "purged", "session_id": session_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── UI Route ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, tags=["UI"])
def get_ui():
    """Serves the premium React chat UI at the root address."""
    html_path = Path(__file__).parent / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ── Threads List ──────────────────────────────────────────────────────────────
@app.get("/threads", tags=["Memory"])
def list_threads():
    """List all thread IDs that have an active checkpoint in memory."""
    try:
        thread_ids = list({
            ns[0]
            for ns in _checkpointer.storage.keys()
            if isinstance(ns, tuple) and len(ns) >= 1
        })
        return {"threads": thread_ids, "count": len(thread_ids)}
    except Exception as exc:
        return {"threads": [], "count": 0, "note": f"Cannot introspect checkpointer: {exc}"}


# ── Documents Management ──────────────────────────────────────────────────────
@app.get("/documents", tags=["System"])
def list_documents(session_id: Optional[str] = None, thread_id: Optional[str] = None):
    """List all ingested PDFs and their chunk counts in Qdrant (filtered by session if specified)."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    try:
        client = QdrantClient(url=config.QDRANT_URL)
        existing = {c.name for c in client.get_collections().collections}
        if config.COLLECTION_NAME not in existing:
            return {"documents": [], "count": 0}

        active_session = session_id or thread_id
        scroll_filter = None
        if active_session and active_session != "global":
            scroll_filter = Filter(
                should=[
                    FieldCondition(key="session_id", match=MatchValue(value=active_session)),
                    FieldCondition(key="session_id", match=MatchValue(value="global")),
                ]
            )

        scroll_res = client.scroll(
            collection_name=config.COLLECTION_NAME,
            scroll_filter=scroll_filter,
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
            must_conditions = [FieldCondition(key="source", match=MatchValue(value=src))]
            if active_session and active_session != "global":
                must_conditions.append(
                    Filter(
                        should=[
                            FieldCondition(key="session_id", match=MatchValue(value=active_session)),
                            FieldCondition(key="session_id", match=MatchValue(value="global")),
                        ]
                    )
                )
            count = client.count(
                collection_name=config.COLLECTION_NAME,
                count_filter=Filter(must=must_conditions)
            ).count
            results.append({
                "filename": src,
                "chunks": count
            })

        return {"documents": results, "count": len(results), "session_id": active_session or "global"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/documents/{filename}", tags=["System"])
def delete_document(filename: str):
    """Delete all chunks for a specific document from Qdrant and delete its local files."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    try:
        client = QdrantClient(url=config.QDRANT_URL)
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
        pdf_path = config.PDF_DIR / filename
        pdf_path.unlink(missing_ok=True)
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


# ── Collections ───────────────────────────────────────────────────────────────
@app.get("/collections", tags=["System"])
def list_collections():
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


# ── PDF Ingestion ─────────────────────────────────────────────────────────────
class FileIngestStats(BaseModel):
    filename:   str
    chunks:     int
    status:     str
    embed_secs: float
    error:      Optional[str] = None

class BatchIngestResponse(BaseModel):
    status:     str
    results:    list[FileIngestStats]

@app.post("/ingest", response_model=BatchIngestResponse, tags=["Ingestion"])
async def ingest_endpoint(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    session_id: Optional[str] = None,
    thread_id: Optional[str] = None,
):
    """Upload and ingest multiple PDFs into Qdrant in a single batch tagged with session_id."""
    from ingest import ingest_pdf, unload_embed_model

    active_session = session_id or thread_id or "global"

    results = []
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            results.append(FileIngestStats(
                filename=file.filename or "unknown",
                chunks=0,
                status="failed",
                embed_secs=0.0,
                error="Only PDF files are supported"
            ))
            continue

        try:
            content = await file.read()
            if len(content) > config.MAX_PDF_MB * 1024 * 1024:
                results.append(FileIngestStats(
                    filename=file.filename,
                    chunks=0,
                    status="failed",
                    embed_secs=0.0,
                    error=f"File exceeds max size of {config.MAX_PDF_MB} MB"
                ))
                continue

            safe_name = Path(file.filename).name
            pdf_path  = config.PDF_DIR / safe_name
            pdf_path.write_bytes(content)

            t0 = time.perf_counter()
            try:
                stats = await asyncio.to_thread(ingest_pdf, pdf_path, parser="auto", embedding_model=embedding_model, session_id=active_session)
                results.append(FileIngestStats(
                    filename=file.filename,
                    chunks=stats.get("chunks", 0),
                    status="ingested",
                    embed_secs=round(time.perf_counter() - t0, 2)
                ))
            except Exception as exc:
                pdf_path.unlink(missing_ok=True)
                results.append(FileIngestStats(
                    filename=file.filename,
                    chunks=0,
                    status="failed",
                    embed_secs=0.0,
                    error=str(exc)
                ))
        except Exception as exc:
            results.append(FileIngestStats(
                filename=file.filename,
                chunks=0,
                status="failed",
                embed_secs=0.0,
                error=f"Upload read error: {exc}"
            ))

    # Free memory at the end of the batch
    try:
        unload_embed_model()
    except Exception:
        pass

    return BatchIngestResponse(
        status="completed",
        results=results
    )


if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run("agentic_api:app", host="0.0.0.0", port=port, reload=False)
