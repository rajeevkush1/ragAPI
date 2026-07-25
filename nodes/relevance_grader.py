"""
nodes/relevance_grader.py
━━━━━━━━━━━━━━━━━━━━━━━━
Node 3 of 5 – Relevance Grader
Model  : Qwen 3.5:0.8b  (tiny, fast, strong at scoring)
Purpose: Score each retrieved chunk 0.0–1.0 for relevance to the query.
         Chunks below RELEVANCE_THRESHOLD are dropped before generation.

Standalone test:
    python nodes/relevance_grader.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import concurrent.futures

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config
from state import RAGState, RetrievedDoc, GradedDoc

# ── Config ────────────────────────────────────────────────────────────────────
_GRADER_MODEL       = config.resolve_model(
    ["gemma4:31b-cloud", "nemotron-3-super:cloud"], "llama3.2:1b"
)
RELEVANCE_THRESHOLD = config.RELEVANCE_THRESHOLD
MAX_WORKERS         = config.MAX_WORKERS

_PROMPT = ChatPromptTemplate.from_template(
    """You are a strict relevance grader for an AI research paper RAG system.

Evaluate whether the document chunk is relevant to answering the user query.

Return a JSON object with exactly these keys:
{{
  "score": <float 0.0 to 1.0>,
  "reason": "<one sentence explaining the score>"
}}

Scoring guide:
- 0.9–1.0 : Directly answers the query with specific details
- 0.7–0.9 : Highly relevant, discusses the same topic/method
- 0.5–0.7 : Partially relevant, tangentially related
- 0.2–0.5 : Slightly related but unlikely to help answer the query
- 0.0–0.2 : Not relevant at all

Return ONLY valid JSON, no markdown, no preamble.

Query: {query}
Chunk: {text}

JSON:"""
)

# ── Core function (standalone-testable) ───────────────────────────────────────
def grade_doc(
    query: str,
    doc: RetrievedDoc,
    llm: ChatOllama | None = None,
) -> GradedDoc:
    """
    Grade a single retrieved document for relevance.

    Returns a GradedDoc with relevance_score and relevance_reason fields added.
    """
    if llm is None:
        llm = ChatOllama(
            model=_GRADER_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.0,
            num_predict=128,
            client_kwargs={"timeout": config.OLLAMA_TIMEOUT},
        )

    chain = _PROMPT | llm | StrOutputParser()

    # Truncate chunk for grader (saves tokens; grader only needs gist)
    chunk_preview = doc["text"][:800]

    raw = chain.invoke({"query": query, "text": chunk_preview})
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        parsed = json.loads(raw)
        score  = max(0.0, min(1.0, float(parsed.get("score", 0.0))))
        reason = str(parsed.get("reason", ""))
    except (json.JSONDecodeError, ValueError):
        score  = 0.0
        reason = "parse error"

    return GradedDoc(
        text=doc["text"],
        source=doc["source"],
        score=doc["score"],
        relevance_score=round(score, 4),
        relevance_reason=reason,
        h1=doc["h1"],
        h2=doc["h2"],
        chunk_id=doc["chunk_id"],
    )


def grade_all_docs(
    query: str,
    docs: list[RetrievedDoc],
    llm: ChatOllama | None = None,
    threshold: float = RELEVANCE_THRESHOLD,
    parallel: bool = True,
) -> list[GradedDoc]:
    """
    Grade all documents, filter by threshold, sort by relevance_score descending.

    Args:
        query:     The (rewritten) query
        docs:      Retrieved docs from vector_retriever
        llm:       Optional shared LLM instance
        threshold: Drop docs with relevance_score below this
        parallel:  Grade docs concurrently (faster for large doc sets)

    Returns:
        Filtered + sorted list of GradedDoc
    """
    if llm is None:
        llm = ChatOllama(
            model=_GRADER_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.0,
            num_predict=128,
            client_kwargs={"timeout": config.OLLAMA_TIMEOUT},
        )

    if parallel and len(docs) > 1:
        # Grade docs concurrently — each call gets its own chain (stateless)
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(grade_doc, query, doc, None) for doc in docs]
            graded  = [f.result() for f in futures]  # preserve submission order
    else:
        graded = [grade_doc(query, doc, llm) for doc in docs]

    # Filter + sort
    relevant = [g for g in graded if g["relevance_score"] >= threshold]
    relevant.sort(key=lambda g: g["relevance_score"], reverse=True)
    return relevant


# ── LangGraph node ────────────────────────────────────────────────────────────
def relevance_grader(state: RAGState) -> dict:
    """LangGraph node — reads rewritten_query + retrieved_docs, writes graded_docs."""
    query = state.get("rewritten_query") or state["original_query"]
    docs  = state.get("retrieved_docs", [])

    graded = grade_all_docs(query=query, docs=docs)
    return {"graded_docs": graded}


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 70)
    print("  NODE TEST: relevance_grader  (qwen3.5:0.8b)")
    print("═" * 70)

    # Mock docs for testing without needing Qdrant
    mock_docs: list[RetrievedDoc] = [
        RetrievedDoc(
            text=(
                "Flash Attention is an IO-aware exact attention algorithm that "
                "uses tiling to reduce the number of memory reads/writes between "
                "GPU high bandwidth memory (HBM) and GPU on-chip SRAM. "
                "Flash Attention is 2-4× faster than standard attention and uses "
                "up to 10-20× less memory."
            ),
            source="flash_attention.pdf",
            score=0.92,
            dense_rank=0,
            bm25_rank=0,
            h1="Introduction",
            h2="Method",
            chunk_id="abc123",
        ),
        RetrievedDoc(
            text=(
                "The Adam optimizer uses adaptive learning rates and is commonly "
                "used for training neural networks. It maintains moving averages "
                "of gradients and squared gradients."
            ),
            source="adam_paper.pdf",
            score=0.45,
            dense_rank=3,
            bm25_rank=5,
            h1="Methods",
            h2="Optimization",
            chunk_id="def456",
        ),
        RetrievedDoc(
            text=(
                "Multi-head attention allows the model to jointly attend to "
                "information from different representation subspaces at different "
                "positions. With a single attention head, averaging inhibits this."
            ),
            source="attention_paper.pdf",
            score=0.78,
            dense_rank=1,
            bm25_rank=2,
            h1="Model Architecture",
            h2="Attention",
            chunk_id="ghi789",
        ),
    ]

    query = "How does Flash Attention reduce GPU memory usage?"
    print(f"\n  Query : {query}")
    print(f"  Docs  : {len(mock_docs)} retrieved\n")

    graded = grade_all_docs(query, mock_docs, threshold=0.4)

    for g in graded:
        bar = "█" * int(g["relevance_score"] * 10) + "░" * (10 - int(g["relevance_score"] * 10))
        print(f"  [{bar}] {g['relevance_score']:.2f}  {g['source']}")
        print(f"         {g['relevance_reason']}")

    print(f"\n  ✓ Graded {len(mock_docs)} docs → {len(graded)} passed threshold {RELEVANCE_THRESHOLD}\n")
