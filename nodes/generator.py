"""
nodes/generator.py
━━━━━━━━━━━━━━━━━━
Node 4 of 5 – Answer Generator
Model  : Qwen 3.5:0.8b  (good instruction following at tiny size)
Purpose: Synthesize a precise, cited answer from graded context chunks.
         Adapts prompt style to query_type for better output quality.

Standalone test:
    python nodes/generator.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config
from state import RAGState, GradedDoc

# ── Model ─────────────────────────────────────────────────────────────────────
_GENERATOR_MODEL = config.resolve_model(
    ["gemma4:31b-cloud", "nemotron-3-super:cloud"], "llama3.2:1b"
)

# ── Prompt templates per query type ──────────────────────────────────────────
_BASE_SYSTEM = (
    "You are an expert AI research assistant. "
    "Answer the question using ONLY the provided context. "
    "Cite sources inline as [source_name] after each claim. "
    "If the answer is not in the context, say so explicitly—do NOT fabricate. "
    "Be highly detailed, exhaustive, and specific in your explanation. "
    "Do not summarize or omit exact technical names, version numbers, operating system sub-distributions "
    "(such as specific Linux distributions like Ubuntu), or exact categories of forensic data and memory states "
    "(such as backpropagation and gradient memory states, or optimizer names like Adam). "
    "Always list all concrete details, categories, parameters, and systems mentioned in the source context."
)

_PROMPTS: dict[str, ChatPromptTemplate] = {
    "factual": ChatPromptTemplate.from_template(
        _BASE_SYSTEM + "\n\n"
        "Provide a direct, concise answer. Lead with the key fact.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    "analytical": ChatPromptTemplate.from_template(
        _BASE_SYSTEM + "\n\n"
        "Provide a structured analytical answer. Use bullet points or numbered steps "
        "where helpful. Explain the 'why' behind each point.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Analysis:"
    ),
    "comparative": ChatPromptTemplate.from_template(
        _BASE_SYSTEM + "\n\n"
        "Structure your answer as a clear comparison. Use a table if comparing "
        "3+ attributes, otherwise use parallel prose sections.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Comparison:"
    ),
    "procedural": ChatPromptTemplate.from_template(
        _BASE_SYSTEM + "\n\n"
        "Provide step-by-step instructions. Number each step. "
        "Include code snippets from the context if present.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Steps:"
    ),
}

# Fallback prompt
_DEFAULT_PROMPT = _PROMPTS["analytical"]


# ── Context builder ────────────────────────────────────────────────────────────
def _build_context(graded_docs: list[GradedDoc], max_chars: int = 4000) -> str:
    """
    Build a context string from graded docs, with citation labels.
    Respects a character budget to avoid overflowing the model's context window.
    """
    parts:  list[str] = []
    total = 0

    for doc in graded_docs:
        section = f"{doc['h1']} > {doc['h2']}".strip(" >")
        label   = doc["source"]
        header  = f"[{label} | {section}] (relevance={doc['relevance_score']:.2f})"
        block   = f"{header}\n{doc['text']}\n"

        if total + len(block) > max_chars:
            break

        parts.append(block)
        total += len(block)

    return "\n---\n".join(parts) if parts else "No context available."


# ── Core function (standalone-testable) ───────────────────────────────────────
def generate_answer(
    question:    str,
    graded_docs: list[GradedDoc],
    query_type:  str = "analytical",
    llm:         ChatOllama | None = None,
    strict:      bool = False,
    generation_hint: str = "",
) -> str:
    """
    Generate an answer from graded context.

    Args:
        question:         The (rewritten) user question
        graded_docs:      Filtered, scored context docs
        query_type:       Controls prompt style
        llm:              Optional pre-built ChatOllama
        strict:           Adds conservative grounding preamble
        generation_hint:  Explicit instruction injected by Edge B on retry
                          e.g. "Do not infer. Only state what sources say directly."

    Returns:
        Answer string with inline [source] citations
    """
    if not graded_docs:
        return (
            "I couldn't find relevant information in the loaded research papers. "
            "Try ingesting more PDFs or rephrasing your question."
        )

    if llm is None:
        llm = ChatOllama(
            model=_GENERATOR_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.15,
            num_predict=1024,
            client_kwargs={"timeout": config.OLLAMA_TIMEOUT},
        )

    context = _build_context(graded_docs)

    # Layer 1: strict mode preamble (retry_count > 0)
    if strict:
        context = (
            "⚠ IMPORTANT: Only cite claims that are EXPLICITLY stated in the "
            "context below. If unsure, say 'the context does not specify this.'\n\n"
            + context
        )

    # Layer 2: generation_hint from Edge B (specific unsupported claims to avoid)
    if generation_hint:
        context = (
            f"🔴 CORRECTION REQUIRED: {generation_hint}\n\n"
            + context
        )

    prompt   = _PROMPTS.get(query_type, _DEFAULT_PROMPT)
    chain    = prompt | llm | StrOutputParser()
    answer   = chain.invoke({"context": context, "question": question})

    return answer.strip()


# ── LangGraph node ────────────────────────────────────────────────────────────
def generator(state: RAGState) -> dict:
    """LangGraph node — reads graded_docs, rewritten_query, query_type, generation_hint."""
    question         = state.get("rewritten_query") or state["original_query"]
    graded_docs      = state.get("graded_docs", [])
    query_type       = state.get("query_type", "analytical")
    generation_retry = state.get("generation_retry_count", 0)
    generation_hint  = state.get("generation_hint", "")

    # strict = True from the first generation retry onward
    strict = generation_retry > 0

    answer = generate_answer(
        question=question,
        graded_docs=graded_docs,
        query_type=query_type,
        strict=strict,
        generation_hint=generation_hint,
    )
    return {"answer": answer}


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from state import GradedDoc

    print("\n" + "═" * 70)
    print("  NODE TEST: generator  (qwen3.5:0.8b)")
    print("═" * 70)

    mock_graded: list[GradedDoc] = [
        GradedDoc(
            text=(
                "Flash Attention achieves memory efficiency through tiling. "
                "Instead of materializing the full N×N attention matrix in HBM, "
                "it computes attention in blocks that fit in fast SRAM. "
                "This reduces HBM reads/writes from O(N²) to O(N), enabling "
                "sequences up to 64K tokens on an A100 GPU."
            ),
            source="flash_attention.pdf",
            score=0.93,
            relevance_score=0.95,
            relevance_reason="Directly explains the memory mechanism of Flash Attention",
            h1="Method",
            h2="Tiling Algorithm",
            chunk_id="abc123",
        ),
        GradedDoc(
            text=(
                "Flash Attention-2 further improves parallelism by partitioning "
                "work across thread blocks along the sequence length dimension, "
                "achieving 2× speedup over Flash Attention on A100 GPUs."
            ),
            source="flash_attention2.pdf",
            score=0.88,
            relevance_score=0.87,
            relevance_reason="Provides quantitative speedup data for Flash Attention-2",
            h1="Flash Attention-2",
            h2="Parallelism",
            chunk_id="def456",
        ),
    ]

    for qt in ["factual", "analytical", "comparative"]:
        print(f"\n  ── query_type={qt} " + "─" * 40)
        answer = generate_answer(
            question="How does Flash Attention reduce GPU memory usage?",
            graded_docs=mock_graded,
            query_type=qt,
        )
        print(f"  {answer[:400]}...")

    print("\n  ✓ generator test complete\n")
