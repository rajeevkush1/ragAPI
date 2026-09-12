"""
agentic_tools.py – Custom tools for the Agentic RAG pipeline using local FastEmbed.
"""
from __future__ import annotations

import re
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from fastembed import TextEmbedding
import config

# Cache local embedding models to avoid reloading them on every query
_local_embed_models: dict[str, TextEmbedding] = {}

def get_local_embed_model(model_name: str) -> TextEmbedding:
    """Get or load a local FastEmbed embedding model (caches loaded models)."""
    global _local_embed_models
    target_model = model_name
    
    # Map BAAI/bge-tiny-en-v1.5 to sentence-transformers/all-MiniLM-L6-v2
    if "tiny" in target_model.lower():
        target_model = "sentence-transformers/all-MiniLM-L6-v2"
        
    if target_model not in _local_embed_models:
        _local_embed_models[target_model] = TextEmbedding(model_name=target_model)
    return _local_embed_models[target_model]

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lower-cased."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())

def _rrf_score(dense_rank: int, bm25_rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score calculation."""
    dense_contrib = 1.0 / (k + dense_rank)
    bm25_contrib  = 1.0 / (k + bm25_rank) if bm25_rank >= 0 else 0.0
    return dense_contrib + bm25_contrib

def local_hybrid_retrieve(
    query: str, 
    embedding_model: str, 
    top_k: int = 6, 
    retrieval_strategy: str = "hybrid",
    session_id: str | None = None
) -> list[dict]:
    """
    Perform local retrieval using FastEmbed + Qdrant (dense) and/or BM25 (sparse).
    Filters candidates by session_id (session-specific + global documents).
    """
    # 1. Fetch dense query vector via local FastEmbed
    embed = get_local_embed_model(embedding_model)
    q_vec = list(embed.embed([query]))[0].tolist()

    # 2. Build Qdrant query filter for session isolation
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    query_filter = None
    if session_id and session_id != "global":
        query_filter = Filter(
            should=[
                FieldCondition(key="session_id", match=MatchValue(value=session_id)),
                FieldCondition(key="session_id", match=MatchValue(value="global")),
            ]
        )

    # 3. Query local Qdrant instance for dense candidates
    try:
        qdrant = QdrantClient(url=config.QDRANT_URL, timeout=5.0)
        candidate_k = max(top_k * 3, 15)
        response = qdrant.query_points(
            collection_name=config.COLLECTION_NAME,
            query=q_vec,
            query_filter=query_filter,
            limit=candidate_k,
            score_threshold=config.SCORE_THRESH,
            with_payload=True
        )
        results = response.points
    except Exception as exc:
        config.logger.warning(f"Qdrant query notice: {exc}")
        return []
        
    if not results and query_filter is not None:
        try:
            response = qdrant.query_points(
                collection_name=config.COLLECTION_NAME,
                query=q_vec,
                limit=candidate_k,
                score_threshold=config.SCORE_THRESH,
                with_payload=True
            )
            results = response.points
        except Exception:
            pass

    if not results:
        try:
            scroll_res = qdrant.scroll(
                collection_name=config.COLLECTION_NAME,
                limit=candidate_k,
                with_payload=True
            )
            results = scroll_res[0]
        except Exception:
            return []

    if not results:
        return []

    # Build candidates list
    candidates = []
    for rank, r in enumerate(results):
        payload = r.payload or {}
        candidates.append({
            "text":        payload.get("text", ""),
            "source":      payload.get("source", "unknown"),
            "h1":          payload.get("h1", ""),
            "h2":          payload.get("h2", ""),
            "authors":     payload.get("authors", ""),
            "venue":       payload.get("venue", ""),
            "year":        payload.get("year", ""),
            "chunk_id":    payload.get("chunk_id", ""),
            "dense_score": float(r.score),
            "dense_rank":  rank,
            "bm25_rank":   -1,
            "bm25_score":  0.0,
        })

    # 3. Local BM25 re-scoring
    corpus_tokens = [_tokenize(c["text"]) for c in candidates]
    bm25 = BM25Okapi(corpus_tokens)
    bm25_query_tokens = _tokenize(query)
    bm25_scores = bm25.get_scores(bm25_query_tokens)

    bm25_ranked = sorted(
        range(len(candidates)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )
    for bm25_rank, candidate_idx in enumerate(bm25_ranked):
        candidates[candidate_idx]["bm25_rank"]  = bm25_rank
        candidates[candidate_idx]["bm25_score"] = float(bm25_scores[candidate_idx])

    # 4. Strategy ranking & RRF merge
    strat = (retrieval_strategy or "hybrid").lower()
    for c in candidates:
        c["rrf_score"] = _rrf_score(c["dense_rank"], c["bm25_rank"])

    if "dense" in strat:
        candidates.sort(key=lambda c: c["dense_score"], reverse=True)
        # Normalize dense score to 0..1 range
        max_s = max(c["dense_score"] for c in candidates) or 1.0
        for c in candidates:
            c["relevance_score"] = min(0.99, max(0.50, round(c["dense_score"] / max_s, 2)))
    elif "sparse" in strat or "bm25" in strat:
        candidates.sort(key=lambda c: c["bm25_score"], reverse=True)
        max_b = max(c["bm25_score"] for c in candidates) or 1.0
        for c in candidates:
            c["relevance_score"] = min(0.99, max(0.50, round(c["bm25_score"] / max_b, 2)))
    else:  # hybrid
        candidates.sort(key=lambda c: c["rrf_score"], reverse=True)
        max_r = max(c["rrf_score"] for c in candidates) or 1.0
        for c in candidates:
            c["relevance_score"] = min(0.99, max(0.50, round(c["rrf_score"] / max_r, 2)))

    return candidates[:top_k]

@tool
def retrieve_research_papers(query: str, config_run: RunnableConfig) -> str:
    """
    Search and retrieve relevant text chunks from the ingested research papers vector store.
    Use this whenever you need factual context to answer questions about AI papers, machine learning,
    system architecture, or mathematical formulas contained in the loaded documents.
    """
    configurable = config_run.get("configurable", {})
    emb_model = configurable.get("embedding_model", "BAAI/bge-small-en-v1.5")
    top_k = configurable.get("top_k", config.TOP_K)
    strategy = configurable.get("retrieval_strategy", "hybrid")
    session_id = configurable.get("thread_id") or configurable.get("session_id")
    
    try:
        docs = local_hybrid_retrieve(
            query=query, 
            embedding_model=emb_model, 
            top_k=top_k, 
            retrieval_strategy=strategy,
            session_id=session_id
        )
        if not docs:
            return "No relevant context found in the research papers. The user might need to ingest more PDFs."

        formatted_docs = []
        for idx, doc in enumerate(docs, 1):
            source_info = doc.get("source", "paper.pdf")
            authors = doc.get("authors", "")
            venue = doc.get("venue", "")
            year = doc.get("year", "")
            meta_str = f"Source: {source_info} | Title: {doc.get('h1', '')} > {doc.get('h2', '')}"
            if authors:
                meta_str += f" | Authors: {authors}"
            if venue or year:
                meta_str += f" | Venue: {venue} ({year})"
            meta_str += f" | Relevance: {doc.get('relevance_score', 0.95)}"

            formatted_docs.append(
                f"[Doc {idx}] {meta_str}\n"
                f"Content: {doc.get('text')}"
            )
        return "\n\n---\n\n".join(formatted_docs)
    except Exception as exc:
        return f"Error during retrieval: {exc}"
