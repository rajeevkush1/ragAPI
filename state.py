"""
state.py – Shared state schema for the Advanced RAG LangGraph pipeline.

Every node reads from and writes to this TypedDict.
Fields are ordered by pipeline stage for readability.
"""
from __future__ import annotations

from typing import TypedDict, Optional


# ── Per-document types ────────────────────────────────────────────────────────

class RetrievedDoc(TypedDict):
    """A single document returned by the vector retriever."""
    text:        str
    source:      str          # original PDF filename
    score:       float        # retrieval score (RRF-merged)
    dense_rank:  int          # rank in dense retrieval
    bm25_rank:   int          # rank in BM25 retrieval (-1 if not found)
    h1:          str          # Markdown section h1
    h2:          str          # Markdown section h2
    chunk_id:    str          # MD5 hash chunk identifier


class GradedDoc(TypedDict):
    """A RetrievedDoc enriched with an LLM relevance score."""
    text:            str
    source:          str
    score:           float        # original retrieval score
    relevance_score: float        # 0.0–1.0 from relevance grader
    relevance_reason: str         # grader's one-line explanation
    h1:              str
    h2:              str
    chunk_id:        str


# ── Graph state ───────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    # ── Stage 0: Input ────────────────────────────────────────────────────────
    original_query: str

    # ── Stage 1: query_analyzer ──────────────────────────────────────────────
    rewritten_query:  str
    query_type:       str          # factual | analytical | comparative | procedural
    key_terms:        list[str]    # extracted for BM25 boosting
    needs_context:    bool         # False → answer directly without retrieval

    # ── Stage 2: vector_retriever ─────────────────────────────────────────────
    retrieved_docs:  list[RetrievedDoc]

    # ── Stage 3: relevance_grader ─────────────────────────────────────────────
    graded_docs:     list[GradedDoc]

    # ── Stage 4: generator ────────────────────────────────────────────────────
    answer:          str

    # ── Stage 5: hallucination_checker ────────────────────────────────────────
    is_grounded:         bool
    grounding_score:     float          # 0.0–1.0
    unsupported_claims:  list[str]      # sentences not backed by sources
    final_answer:        str

    # ── Multi-turn conversation memory (persisted by MemorySaver) ────────────────
    conversation_history: list[dict]   # [{"role": "user"|"assistant", "content": str}]

    # ── Loop A: retrieval retry (Edge A: grader → query_analyzer) ──────────────
    retrieval_retry_count: int        # how many times retrieval loop has fired
    max_retrieval_retries: int        # cap (default 3)
    failed_queries:        list[str]  # rewritten queries that yielded no results

    # ── Loop B: generation retry (Edge B: checker → generator) ───────────────
    generation_retry_count: int       # how many times generation loop has fired
    max_generation_retries: int       # cap (default 3)
    generation_hint:        str       # injected instruction for strict re-generation
                                      # e.g. "be more conservative, do not infer"
