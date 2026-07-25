"""
nodes/hallucination_checker.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Node 5 of 5 – Hallucination Checker
Model  : Llama 3.2:1b  (fast, binary classification)
Purpose: Verify each sentence in the generated answer is grounded in
         the source documents. Flag unsupported claims and return a
         grounding score. Low scores trigger a regeneration retry.

Standalone test:
    python nodes/hallucination_checker.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config
from state import RAGState, GradedDoc

# ── Config ────────────────────────────────────────────────────────────────────
_CHECKER_MODEL      = config.resolve_model(
    ["gemma4:31b-cloud", "nemotron-3-super:cloud"], "llama3.2:1b"
)
GROUNDING_THRESHOLD = config.GROUNDING_THRESHOLD
MAX_SOURCE_CHARS    = config.MAX_SOURCE_CHARS

_PROMPT = ChatPromptTemplate.from_template(
    """You are a hallucination detector for a RAG system.

Your task: Check whether the generated answer is fully grounded in the source documents.

Instructions:
1. Read the source documents carefully.
2. For each factual claim in the answer, determine if it's explicitly supported.
3. Identify any claims NOT found in the sources (hallucinations or speculation).

Return a JSON object with exactly these keys:
{{
  "is_grounded": <true if ALL key claims are supported, false otherwise>,
  "grounding_score": <float 0.0–1.0, fraction of claims that ARE grounded>,
  "unsupported_claims": ["<exact sentence 1>", "<exact sentence 2>", ...]
}}

Rules:
- grounding_score = 1.0 means perfectly grounded, 0.0 means completely hallucinated
- unsupported_claims should be the verbatim sentences from the answer that are NOT in sources
- An empty unsupported_claims list with grounding_score >= 0.8 means is_grounded = true
- Return ONLY valid JSON, no markdown, no explanation

Source Documents:
{sources}

Generated Answer:
{answer}

JSON:"""
)


# ── Source formatter ───────────────────────────────────────────────────────────
def _format_sources(graded_docs: list[GradedDoc], max_chars: int = MAX_SOURCE_CHARS) -> str:
    """Compact source representation for the checker's context."""
    parts  = []
    total  = 0

    for doc in graded_docs:
        entry = f"[{doc['source']}] {doc['text']}"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)

    return "\n\n".join(parts) if parts else "No sources available."


# ── Core function (standalone-testable) ───────────────────────────────────────
def check_hallucination(
    answer:      str,
    graded_docs: list[GradedDoc],
    llm:         ChatOllama | None = None,
) -> dict:
    """
    Check if the answer is grounded in the provided source docs.

    Returns:
        dict with keys:
            is_grounded        (bool)
            grounding_score    (float 0.0–1.0)
            unsupported_claims (list[str])
    """
    if not graded_docs or not answer:
        return {
            "is_grounded":        False,
            "grounding_score":    0.0,
            "unsupported_claims": ["No source documents to verify against."],
        }

    if llm is None:
        llm = ChatOllama(
            model=_CHECKER_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.0,
            num_predict=512,
            client_kwargs={"timeout": config.OLLAMA_TIMEOUT},
        )

    sources = _format_sources(graded_docs)
    chain   = _PROMPT | llm | StrOutputParser()
    raw     = chain.invoke({"sources": sources, "answer": answer})
    raw     = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        parsed = json.loads(raw)
        score  = max(0.0, min(1.0, float(parsed.get("grounding_score", 0.0))))
        return {
            "is_grounded":        bool(parsed.get("is_grounded", score >= GROUNDING_THRESHOLD)),
            "grounding_score":    round(score, 4),
            "unsupported_claims": list(parsed.get("unsupported_claims", [])),
        }
    except (json.JSONDecodeError, ValueError):
        # Can't parse → fail closed: treat as unverified so Edge B retry fires
        return {
            "is_grounded":        False,
            "grounding_score":    0.0,
            "unsupported_claims": ["[checker parse error — could not verify grounding]"],
        }


# ── LangGraph node ────────────────────────────────────────────────────────────
def hallucination_checker(state: RAGState) -> dict:
    """
    LangGraph node — reads answer + graded_docs, writes hallucination check results.
    Also sets final_answer (either the answer itself or a qualified version).
    """
    answer      = state.get("answer", "")
    graded_docs = state.get("graded_docs", [])

    result = check_hallucination(answer=answer, graded_docs=graded_docs)

    # Build final_answer — annotate if hallucinations detected
    if result["is_grounded"]:
        final_answer = answer
    else:
        unsupported = result["unsupported_claims"]
        if unsupported:
            caveat = (
                "\n\n---\n⚠ *Note: The following claims could not be verified "
                f"against the source documents and may be inaccurate:*\n"
                + "\n".join(f"- {c}" for c in unsupported)
            )
            final_answer = answer + caveat
        else:
            final_answer = answer

    return {
        "is_grounded":        result["is_grounded"],
        "grounding_score":    result["grounding_score"],
        "unsupported_claims": result["unsupported_claims"],
        "final_answer":       final_answer,
    }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from state import GradedDoc

    print("\n" + "═" * 70)
    print("  NODE TEST: hallucination_checker  (llama3.2:1b)")
    print("═" * 70)

    mock_sources: list[GradedDoc] = [
        GradedDoc(
            text=(
                "Flash Attention achieves O(N) memory complexity by using "
                "tiling to avoid materializing the full NxN attention matrix. "
                "It is 2-4x faster than standard attention."
            ),
            source="flash_attention.pdf",
            score=0.93,
            relevance_score=0.95,
            relevance_reason="Direct match",
            h1="Method",
            h2="Algorithm",
            chunk_id="abc",
        ),
    ]

    grounded_answer = (
        "Flash Attention reduces memory from O(N²) to O(N) using tiling, "
        "avoiding materializing the full attention matrix. "
        "It is 2-4× faster than standard attention [flash_attention.pdf]."
    )

    hallucinated_answer = (
        "Flash Attention reduces memory by using sparse attention patterns "
        "and achieves 10× speedup over transformers. "
        "It was invented at Stanford in 2019 and uses a novel XYZ algorithm."
    )

    for label, ans in [
        ("✓ Grounded answer",      grounded_answer),
        ("✗ Hallucinated answer",  hallucinated_answer),
    ]:
        print(f"\n  {label}")
        print(f"  Answer: {ans[:120]}...")
        result = check_hallucination(ans, mock_sources)
        score  = result["grounding_score"]
        bar    = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        print(f"  [{bar}] score={score:.2f}  grounded={result['is_grounded']}")
        if result["unsupported_claims"]:
            print("  Unsupported claims:")
            for claim in result["unsupported_claims"]:
                print(f"    - {claim}")

    print("\n  ✓ hallucination_checker test complete\n")
