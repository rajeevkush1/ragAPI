"""
nodes/vector_retriever.py
━━━━━━━━━━━━━━━━━━━━━━━━
Node 2 of 5 – Hybrid Vector Retriever
Model  : BGE-M3 (dense) + BM25 (sparse)
Merge  : Reciprocal Rank Fusion (RRF)
Purpose: Retrieve semantically relevant AND keyword-matching chunks.

Hybrid strategy:
  1. Dense retrieval  – BGE-M3 cosine search in Qdrant (TOP_K × 2 candidates)
  2. BM25 re-scoring  – rank_bm25 over the dense candidate set
  3. RRF merge        – combine dense_rank + bm25_rank into a single score
  4. Return top TOP_K docs sorted by RRF score

Standalone test:
    python nodes/vector_retriever.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import re

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi                  # pip install rank-bm25

import config
from state import RAGState, RetrievedDoc

# ── Singletons ────────────────────────────────────────────────────────────────
_embed:  TextEmbedding | None = None
_qdrant: QdrantClient  | None = None


def _get_embed() -> TextEmbedding:
    global _embed
    if _embed is None:
        _embed = TextEmbedding(model_name=config.EMBED_MODEL)
    return _embed


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=config.QDRANT_URL)
    return _qdrant


# ── Tokenizer for BM25 ────────────────────────────────────────────────────────
def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lower-cased."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


# ── RRF scoring ───────────────────────────────────────────────────────────────
def _rrf_score(dense_rank: int, bm25_rank: int, k: int = 60) -> float:
    """
    Reciprocal Rank Fusion.
    score = 1/(k + dense_rank) + 1/(k + bm25_rank)
    bm25_rank = -1 means the doc was not found by BM25 → contributes 0.
    """
    dense_contrib = 1.0 / (k + dense_rank)
    bm25_contrib  = 1.0 / (k + bm25_rank) if bm25_rank >= 0 else 0.0
    return dense_contrib + bm25_contrib


# ── Core function (standalone-testable) ───────────────────────────────────────
def hybrid_retrieve(
    query: str,
    key_terms: list[str] | None = None,
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> list[RetrievedDoc]:
    """
    Hybrid BM25 + dense retrieval with RRF fusion.

    Args:
        query:           Natural language query (rewritten preferred)
        key_terms:       Extra BM25 boost terms from query_analyzer
        top_k:           Number of final docs to return (default: config.TOP_K)
        score_threshold: Min cosine similarity for dense stage

    Returns:
        List of RetrievedDoc sorted by RRF score (best first)
    """
    top_k     = top_k or config.TOP_K
    threshold = score_threshold or config.SCORE_THRESH
    candidate_k = top_k * 3          # fetch 3× for BM25 reranking headroom

    embed  = _get_embed()
    qdrant = _get_qdrant()

    # ── Stage 1: Dense retrieval ──────────────────────────────────────────────
    q_vec   = list(embed.embed([query]))[0].tolist()
    response = qdrant.query_points(
        collection_name=config.COLLECTION_NAME,
        query=q_vec,
        limit=candidate_k,
        score_threshold=threshold,
        with_payload=True,
    )
    results = response.points

    if not results:
        return []

    # Build candidate pool with dense ranks
    candidates: list[dict] = []
    for rank, r in enumerate(results):
        candidates.append({
            "text":       r.payload.get("text", ""),
            "source":     r.payload.get("source", "unknown"),
            "h1":         r.payload.get("h1", ""),
            "h2":         r.payload.get("h2", ""),
            "chunk_id":   r.payload.get("chunk_id", ""),
            "dense_score": r.score,
            "dense_rank":  rank,
            "bm25_rank":   -1,         # filled in next stage
            "bm25_score":  0.0,
        })

    # ── Stage 2: BM25 re-scoring over dense candidates ────────────────────────
    corpus_tokens = [_tokenize(c["text"]) for c in candidates]
    bm25          = BM25Okapi(corpus_tokens)

    # Combine rewritten query with key_terms for BM25 query
    bm25_query_text = query
    if key_terms:
        bm25_query_text += " " + " ".join(key_terms)
    bm25_query_tokens = _tokenize(bm25_query_text)

    bm25_scores = bm25.get_scores(bm25_query_tokens)   # numpy array

    # Rank by BM25 score (higher = better)
    bm25_ranked = sorted(
        range(len(candidates)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )
    for bm25_rank, candidate_idx in enumerate(bm25_ranked):
        candidates[candidate_idx]["bm25_rank"]  = bm25_rank
        candidates[candidate_idx]["bm25_score"] = float(bm25_scores[candidate_idx])

    # ── Stage 3: RRF merge ────────────────────────────────────────────────────
    for c in candidates:
        c["rrf_score"] = _rrf_score(c["dense_rank"], c["bm25_rank"])

    candidates.sort(key=lambda c: c["rrf_score"], reverse=True)

    # ── Build output ──────────────────────────────────────────────────────────
    docs: list[RetrievedDoc] = []
    for c in candidates[:top_k]:
        docs.append(
            RetrievedDoc(
                text=c["text"],
                source=c["source"],
                score=round(c["rrf_score"], 6),
                dense_rank=c["dense_rank"],
                bm25_rank=c["bm25_rank"],
                h1=c["h1"],
                h2=c["h2"],
                chunk_id=c["chunk_id"],
            )
        )
    return docs


# ── LangGraph node ────────────────────────────────────────────────────────────
def vector_retriever(state: RAGState) -> dict:
    """LangGraph node — reads rewritten_query + key_terms, writes retrieved_docs."""
    query     = state.get("rewritten_query") or state["original_query"]
    key_terms = state.get("key_terms", [])

    docs = hybrid_retrieve(query=query, key_terms=key_terms)
    return {"retrieved_docs": docs}


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 70)
    print("  NODE TEST: vector_retriever  (BGE-M3 dense + BM25 hybrid + RRF)")
    print("═" * 70)

    test_query = "Flash Attention memory efficient attention mechanism"
    key_terms  = ["flash attention", "memory", "HBM", "tiling"]

    print(f"\n  Query     : {test_query}")
    print(f"  Key terms : {key_terms}")
    print()

    try:
        docs = hybrid_retrieve(test_query, key_terms=key_terms, top_k=4)

        if not docs:
            print("  [!] No results — make sure you have ingested PDFs first.")
            print("      Run: python ingest.py your_paper.pdf")
        else:
            for i, d in enumerate(docs, 1):
                print(f"  [{i}] source={d['source']}  rrf={d['score']:.5f}"
                      f"  dense_rank={d['dense_rank']}  bm25_rank={d['bm25_rank']}")
                print(f"      section: {d['h1']} > {d['h2']}")
                print(f"      text[:120]: {d['text'][:120].replace(chr(10), ' ')}")

        print(f"\n  ✓ Retrieved {len(docs)} docs  (hybrid BM25+dense, RRF merged)\n")

    except Exception as e:
        print(f"  [ERROR] {e}")
        print("  Make sure Qdrant is running: docker compose up -d")
