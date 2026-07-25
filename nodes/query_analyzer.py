"""
nodes/query_analyzer.py
━━━━━━━━━━━━━━━━━━━━━━
Node 1 of 5 – Query Analyzer
Model  : Llama 3.2:1b  (fast, low-latency classifier)
Purpose: Classify the query type, extract key terms, and rewrite
         the query for better retrieval precision.

Standalone test:
    python nodes/query_analyzer.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Allow running as a script from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config
from state import RAGState

# ── Model ─────────────────────────────────────────────────────────────────────
_ANALYZER_MODEL = config.resolve_model(
    ["gemma4:31b-cloud", "nemotron-3-super:cloud"], "llama3.2:1b"
)

_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert query analyzer for an AI research paper retrieval system.

Analyze the user query and return a JSON object with exactly these keys:

{{
  "rewritten_query": "<improved, more specific query optimized for semantic search>",
  "query_type": "<one of: factual | analytical | comparative | procedural>",
  "key_terms": ["<term1>", "<term2>", ...],
  "needs_context": <true if context from papers is needed, false if it can be answered directly>
}}

Query type definitions:
- factual      → asking for a specific fact, number, name, or definition
- analytical   → asking to explain, analyze, or reason about a concept
- comparative  → asking to compare/contrast multiple approaches or methods
- procedural   → asking how to implement or do something step by step

Rules:
- rewritten_query must be concise and retrieval-optimized (under 80 words)
- key_terms must be the 3-7 most important technical terms for BM25 keyword search
- needs_context = false only for simple conversational queries ("hi", "thanks", etc.)
- Return ONLY valid JSON, no markdown fences, no explanation

User query: {query}

JSON:"""
)

# ── Retry prompt (Edge A: used when previous retrieval returned 0 results) ────
_RETRY_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert query analyzer for an AI research paper retrieval system.

Previous retrieval attempts failed — the following query rewrites returned NO relevant results:
{failed_queries}

Your task: Rewrite the query using a DIFFERENT strategy to improve recall:
  - Use broader/more general terminology
  - Try synonyms, acronym expansions, or related concepts
  - Remove overly specific constraints
  - Think about what section headers in an academic paper would cover this topic

Return a JSON object with exactly these keys:

{{
  "rewritten_query": "<broader, alternative query that avoids the failed phrasings above>",
  "query_type": "<one of: factual | analytical | comparative | procedural>",
  "key_terms": ["<term1>", "<term2>", ...],
  "needs_context": true
}}

Rules:
- rewritten_query MUST differ substantially from the failed queries above
- key_terms should use synonyms/related terms not in the failed queries
- Return ONLY valid JSON, no markdown fences, no explanation

Original user query: {query}
Attempt number: {attempt}

JSON:"""
)

# ── Conversational follow-up prompt (used when history exists) ────────────────
_CONTEXTUAL_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert query analyzer for an AI research paper retrieval system.

Conversation history (last {n_turns} turns):
{history}

The user just asked: "{query}"

This may be a follow-up to the conversation above. If it references previous topics
(e.g. "it", "that", "this", "the paper", "compared to"), resolve those references
before rewriting the query.

Return a JSON object with exactly these keys:

{{
  "rewritten_query": "<self-contained, retrieval-optimized query with all references resolved>",
  "query_type": "<one of: factual | analytical | comparative | procedural>",
  "key_terms": ["<term1>", "<term2>", ...],
  "needs_context": <true unless this is a greeting or trivial conversational message>
}}

Rules:
- rewritten_query must be fully self-contained (no pronouns referring to the history)
- key_terms must be the 3-7 most important technical terms for BM25 keyword search
- Return ONLY valid JSON, no markdown fences, no explanation

JSON:"""
)

# ── Core function (standalone-testable) ───────────────────────────────────────
def analyze_query(
    query: str,
    llm: ChatOllama | None = None,
    failed_queries: list[str] | None = None,
    attempt: int = 0,
    conversation_history: list[dict] | None = None,
) -> dict:
    """
    Analyze and rewrite a query.

    Args:
        query:                Raw user question
        llm:                  Optional pre-built ChatOllama (reuse across calls)
        failed_queries:       Previous rewrites that returned no results (Edge A retry)
        attempt:              Which retry attempt this is (0 = first run)
        conversation_history: Previous Q/A turns for follow-up resolution

    Returns:
        dict with keys: rewritten_query, query_type, key_terms, needs_context
    """
    if llm is None:
        llm = ChatOllama(
            model=_ANALYZER_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.0,
            num_predict=256,
            client_kwargs={"timeout": config.OLLAMA_TIMEOUT},
        )

    # Priority 1: retrieval retry — use broadening prompt
    if failed_queries and attempt > 0:
        retry_llm = ChatOllama(
            model=_ANALYZER_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=min(0.3 + attempt * 0.1, 0.7),  # gradually more creative
            num_predict=256,
            client_kwargs={"timeout": config.OLLAMA_TIMEOUT},
        )
        failed_str = "\n".join(f"  - {q}" for q in failed_queries)
        chain = _RETRY_PROMPT | retry_llm | StrOutputParser()
        raw   = chain.invoke({
            "query":          query,
            "failed_queries": failed_str,
            "attempt":        attempt,
        })

    # Priority 2: conversation history exists — resolve follow-up references
    elif conversation_history and len(conversation_history) >= 2:
        # Format the last 3 Q/A pairs (6 messages) as readable history
        recent = conversation_history[-6:]
        history_lines = []
        for msg in recent:
            role   = "User" if msg.get("role") == "user" else "Assistant"
            # Truncate very long messages so we don't blow up the context window
            content = msg.get("content", "")[:300]
            if len(msg.get("content", "")) > 300:
                content += "..."
            history_lines.append(f"  {role}: {content}")
        history_str = "\n".join(history_lines)
        n_turns     = len(recent) // 2
        chain = _CONTEXTUAL_PROMPT | llm | StrOutputParser()
        raw   = chain.invoke({
            "query":   query,
            "history": history_str,
            "n_turns": n_turns,
        })

    # Priority 3: fresh query — standard prompt
    else:
        chain = _PROMPT | llm | StrOutputParser()
        raw   = chain.invoke({"query": query})

    # Strip any accidental markdown fences
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Graceful fallback
        result = {
            "rewritten_query": query,
            "query_type":      "factual",
            "key_terms":       query.split()[:5],
            "needs_context":   True,
        }

    # Force needs_context to True unless this is a simple conversational greeting
    conversational_keywords = ["hi", "hello", "hey", "thanks", "thank", "bye", "good morning", "good afternoon", "welcome"]
    q_clean = query.strip().lower()
    is_greeting = len(q_clean.split()) <= 3 and any(w in q_clean for w in conversational_keywords)

    if attempt > 0 or not is_greeting:
        needs_context = True
    else:
        needs_context = False


    return {
        "rewritten_query": str(result.get("rewritten_query", query)),
        "query_type":      str(result.get("query_type", "factual")),
        "key_terms":       list(result.get("key_terms", [])),
        "needs_context":   needs_context,
    }


# ── LangGraph node ────────────────────────────────────────────────────────────
def query_analyzer(state: RAGState) -> dict:
    """LangGraph node — reads original_query + retry context + conversation_history."""
    result = analyze_query(
        query=state["original_query"],
        failed_queries=state.get("failed_queries", []),
        attempt=state.get("retrieval_retry_count", 0),
        conversation_history=state.get("conversation_history", []),
    )
    return {
        "rewritten_query": result["rewritten_query"],
        "query_type":      result["query_type"],
        "key_terms":       result["key_terms"],
        "needs_context":   result["needs_context"],
    }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "What is Flash Attention and how does it reduce memory usage?",
        "Compare LoRA vs QLoRA for fine-tuning large language models",
        "How do I implement rotary positional embeddings in PyTorch?",
        "What BLEU score did GPT-4 achieve on WMT23?",
        "Hello there!",
    ]

    print("\n" + "═" * 70)
    print("  NODE TEST: query_analyzer  (llama3.2:1b)")
    print("═" * 70)

    llm = ChatOllama(
            model=_ANALYZER_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.0,
            num_predict=256,
            client_kwargs={"timeout": config.OLLAMA_TIMEOUT},
        )

    for q in test_queries:
        print(f"\n  Query : {q}")
        result = analyze_query(q, llm=llm)
        print(f"  ├─ type     : {result['query_type']}")
        print(f"  ├─ rewritten: {result['rewritten_query']}")
        print(f"  ├─ key_terms: {result['key_terms']}")
        print(f"  └─ needs_ctx: {result['needs_context']}")

    print("\n✓ query_analyzer test passed\n")
