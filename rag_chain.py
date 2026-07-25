"""
rag_chain.py – LangGraph ReAct RAG chain backed by Ollama + Qdrant.

Graph nodes:
  retrieve  → semantic search in Qdrant (BGE-M3 vectors)
  grade     → filter irrelevant docs (LLM self-check)
  generate  → answer from graded context (Ollama LLM)
  fallback  → polite "I don't know" if no context survives grading

Usage:
  from rag_chain import build_chain
  chain = build_chain()
  result = chain.invoke({"question": "What is Flash Attention?"})
  print(result["answer"])
"""
from __future__ import annotations

from typing import TypedDict, Annotated, Sequence
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

from langgraph.graph import StateGraph, END

import config

# ── State schema ──────────────────────────────────────────────────────────────
class RAGState(TypedDict):
    question:  str
    documents: list[dict]          # raw Qdrant payloads
    graded:    list[dict]          # filtered docs
    answer:    str


# ── Shared singletons (lazy-init) ─────────────────────────────────────────────
_embed: TextEmbedding | None = None
_qdrant: QdrantClient | None = None
_llm: ChatOllama | None = None


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


def _get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.1,
            client_kwargs={"timeout": config.OLLAMA_TIMEOUT},
        )
    return _llm


# ── Node: retrieve ─────────────────────────────────────────────────────────────
def retrieve(state: RAGState) -> dict:
    """Embed the question and retrieve TOP_K chunks from Qdrant."""
    question = state["question"]
    embed    = _get_embed()
    qdrant   = _get_qdrant()

    q_vec    = list(embed.embed([question]))[0].tolist()
    response = qdrant.query_points(
        collection_name=config.COLLECTION_NAME,
        query=q_vec,
        limit=config.TOP_K,
        score_threshold=config.SCORE_THRESH,
        with_payload=True,
    )
    results = response.points

    docs = [
        {
            "text":   r.payload.get("text", ""),
            "score":  r.score,
            "source": r.payload.get("source", "unknown"),
            "h1":     r.payload.get("h1", ""),
            "h2":     r.payload.get("h2", ""),
        }
        for r in results
    ]
    return {"documents": docs}


# ── Node: grade ────────────────────────────────────────────────────────────────
_GRADE_PROMPT = ChatPromptTemplate.from_template(
    """You are a relevance grader. Given the user question and a document chunk,
output only 'yes' if the chunk is relevant to answering the question, or 'no' if it is not.

Question: {question}
Document: {text}

Relevant (yes/no):"""
)

def grade(state: RAGState) -> dict:
    """LLM-based relevance filter — keeps only genuinely useful chunks."""
    llm     = _get_llm()
    chain   = _GRADE_PROMPT | llm | StrOutputParser()
    graded  = []
    for doc in state["documents"]:
        verdict = chain.invoke({"question": state["question"], "text": doc["text"]})
        if verdict.strip().lower().startswith("yes"):
            graded.append(doc)
    return {"graded": graded}


# ── Node: generate ─────────────────────────────────────────────────────────────
_GEN_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert research assistant specializing in AI and machine learning papers.
Answer the question using ONLY the provided context. Cite sections when possible.
If the answer is not in the context, say "I don't have enough information in the loaded papers."

Context:
{context}

Question: {question}

Answer (be precise and technical):"""
)

def generate(state: RAGState) -> dict:
    """Generate an answer from graded context chunks."""
    llm  = _get_llm()
    chain = _GEN_PROMPT | llm | StrOutputParser()

    context = "\n\n---\n\n".join(
        f"[{d['source']} | {d['h1']} > {d['h2']}]\n{d['text']}"
        for d in state["graded"]
    )
    answer = chain.invoke({"context": context, "question": state["question"]})
    return {"answer": answer}


# ── Node: fallback ─────────────────────────────────────────────────────────────
def fallback(state: RAGState) -> dict:
    return {
        "answer": (
            "I couldn't find relevant information in the loaded research papers. "
            "Try ingesting more PDFs or rephrasing your question."
        )
    }


# ── Edge condition ─────────────────────────────────────────────────────────────
def route_after_grade(state: RAGState) -> str:
    return "generate" if state.get("graded") else "fallback"


# ── Graph builder ──────────────────────────────────────────────────────────────
def build_chain() -> StateGraph:
    """Build and compile the LangGraph RAG chain."""
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("grade",    grade)
    graph.add_node("generate", generate)
    graph.add_node("fallback", fallback)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "fallback": "fallback"},
    )
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)

    return graph.compile()


# ── Quick CLI smoke-test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) or "What is attention mechanism in transformers?"
    print(f"\nQuestion: {question}\n")
    chain  = build_chain()
    result = chain.invoke({"question": question})
    print(f"Answer:\n{result['answer']}")
