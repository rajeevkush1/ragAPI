"""
graph.py – LangGraph StateGraph with two smart retry loops.

════════════════════════════════════════════════════════════════════════
  FLOW DIAGRAM
════════════════════════════════════════════════════════════════════════

  query_analyzer ──[needs_context=False]──────────────────► direct_answer → END
       │
       └─[needs_context=True]──► vector_retriever
                                        │
                                relevance_grader
                                        │
           ┌────────────────────────────┼────────────────────────────┐
     EDGE A │                           │                             │
  [no docs AND              [no docs AND                     [docs found]
   retries < 3]          retries exhausted]                       │
           │                            │                       generator ◄──────┐
  record_failed_query            no_context → END                  │             │
           │                                              hallucination_checker   │
   query_analyzer ◄─────────────────────┘                         │             │
   (broadening prompt)               ┌────────────────────────────┼───────────┐ │
                               EDGE B │                            │           │ │
                        [hallucinated AND              [grounded OR          [hallucinated
                          retries < 3]              retries exhausted]    AND retries < 3]
                                      │                   final_answer        │
                            prepare_strict_gen ──────────────────────────────┘
                            (injects generation_hint,
                             lists unsupported claims)
                                      │
                               generator (strict)

════════════════════════════════════════════════════════════════════════

Usage:
    from graph import build_graph, make_initial_state
    app    = build_graph()
    result = app.invoke(make_initial_state("What is Flash Attention?"))
    print(result["final_answer"])
"""
from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except (AttributeError, TypeError):
    pass

import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from langgraph.graph import StateGraph, END

from state import RAGState
from nodes import (
    query_analyzer,
    vector_retriever,
    relevance_grader,
    generator,
    hallucination_checker,
)
import config

# ── Retry caps (independent for each loop) ────────────────────────────────────
MAX_RETRIEVAL_RETRIES  = 3   # Edge A: grader → query_analyzer
MAX_GENERATION_RETRIES = 3   # Edge B: checker → generator


# ═══════════════════════════════════════════════════════════════════════════════
# TERMINAL NODES  (no LLM, instant)
# ═══════════════════════════════════════════════════════════════════════════════

def _word_match(text: str, words: list[str]) -> bool:
    """True if any of `words` appears as a whole word in `text`."""
    pattern = r"\b(" + "|".join(re.escape(w) for w in words) + r")\b"
    return bool(re.search(pattern, text))


def direct_answer(state: RAGState) -> dict:
    """Shortcut for conversational queries that need no retrieval."""
    q = state["original_query"].strip().lower()
    if _word_match(q, ["hi", "hello", "hey"]):
        msg = (
            "Hello! I'm an AI research paper assistant. "
            "Ask me anything about papers you've ingested — for example: "
            "'What is Flash Attention?' or 'Compare LoRA and QLoRA.'"
        )
    elif _word_match(q, ["thanks", "thank", "bye"]):
        msg = "You're welcome! Let me know if you have more questions."
    else:
        msg = "I can answer questions about your ingested research papers. Please ask a specific question."
    return {
        "final_answer":    msg,
        "answer":          msg,
        "is_grounded":     True,
        "grounding_score": 1.0,
    }


def no_context(state: RAGState) -> dict:
    """Final fallback when all retrieval retries are exhausted."""
    retries = state.get("retrieval_retry_count", 0)
    failed  = state.get("failed_queries", [])

    tried_msg = ""
    if failed:
        tried_msg = (
            "\n\nQuery rewrites attempted:\n"
            + "\n".join(f"  {i+1}. {q}" for i, q in enumerate(failed))
        )

    msg = (
        f"After {retries + 1} retrieval attempt(s), no relevant context was found "
        f"in the loaded research papers for your question.{tried_msg}\n\n"
        "Suggestions:\n"
        "1. Ingest more PDFs: `python ingest.py ./pdfs/`\n"
        "2. Check Qdrant is running: `docker compose up -d`\n"
        "3. Try rephrasing with different terminology"
    )
    return {
        "final_answer":    msg,
        "answer":          msg,
        "is_grounded":     True,
        "grounding_score": 1.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LOOP A NODES  (Edge A: relevance_grader → query_analyzer)
# ═══════════════════════════════════════════════════════════════════════════════

def record_failed_query(state: RAGState) -> dict:
    """
    Edge A transition node.
    Records the current rewritten_query as failed, increments retrieval_retry_count.
    The updated failed_queries list is passed to query_analyzer on the next iteration,
    which uses the _RETRY_PROMPT to generate a broadened alternative.
    """
    failed        = list(state.get("failed_queries", []))
    current_query = state.get("rewritten_query") or state.get("original_query", "")

    if current_query and current_query not in failed:
        failed.append(current_query)

    new_count = state.get("retrieval_retry_count", 0) + 1

    return {
        "failed_queries":        failed,
        "retrieval_retry_count": new_count,
        # Reset downstream state so re-retrieval starts clean
        "retrieved_docs": [],
        "graded_docs":    [],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LOOP B NODES  (Edge B: hallucination_checker → generator)
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_strict_gen(state: RAGState) -> dict:
    """
    Edge B transition node.
    Builds a specific generation_hint from the unsupported claims detected by
    the hallucination_checker, then increments generation_retry_count.

    The hint is injected into the generator context window as a 🔴 CORRECTION block,
    telling the model exactly which sentences to avoid or rephrase.
    """
    unsupported = state.get("unsupported_claims", [])
    score       = state.get("grounding_score", 0.0)
    count       = state.get("generation_retry_count", 0)

    if unsupported:
        claims_str = " | ".join(f'"{c[:80]}"' for c in unsupported[:3])
        hint = (
            f"The previous answer contained {len(unsupported)} unsupported claim(s) "
            f"(grounding score={score:.2f}). "
            f"Do NOT repeat these unverified statements: {claims_str}. "
            "Only assert what is explicitly present in the source documents. "
            "Use hedging phrases ('according to [source]', 'the paper states') "
            "rather than absolute statements."
        )
    else:
        hint = (
            f"The previous answer was not well-grounded (score={score:.2f}). "
            "Be more conservative: only state what the source documents explicitly say. "
            "Use direct quotes where possible."
        )

    return {
        "generation_hint":        hint,
        "generation_retry_count": count + 1,
        # Clear old answer so generator produces a fresh one
        "answer": "",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTING FUNCTIONS (conditional edge logic)
# ═══════════════════════════════════════════════════════════════════════════════

def route_after_analyzer(state: RAGState) -> str:
    """Entry branch: skip retrieval for conversational queries."""
    return "direct_answer" if not state.get("needs_context", True) else "vector_retriever"


def route_after_grader(state: RAGState) -> str:
    """
    EDGE A decision point.

    ┌─ docs found                     → generator
    ├─ no docs + retries remaining    → record_failed_query  (→ query_analyzer)
    └─ no docs + retries exhausted    → no_context
    """
    if state.get("graded_docs"):
        return "generator"

    retries = state.get("retrieval_retry_count", 0)
    if retries < MAX_RETRIEVAL_RETRIES:
        return "record_failed_query"

    return "no_context"


def route_after_checker(state: RAGState) -> str:
    """
    EDGE B decision point.

    ┌─ grounded                                → END
    ├─ not grounded + retries remaining        → prepare_strict_gen (→ generator)
    └─ not grounded + retries exhausted        → END  (with caveat already in final_answer)
    """
    is_grounded = state.get("is_grounded", True)
    retries     = state.get("generation_retry_count", 0)

    if not is_grounded and retries < MAX_GENERATION_RETRIES:
        return "prepare_strict_gen"

    return END


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph(checkpointer=None) -> StateGraph:
    """
    Build and compile the full RAG StateGraph with both retry loops.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. MemorySaver) for
                      persisting state between invocations on the same thread_id.
                      Pass None (default) for stateless single-shot use.

    Returns a compiled LangGraph app ready for .invoke(), .astream(), or .astream_events().
    """
    g = StateGraph(RAGState)

    # ── Core pipeline nodes ────────────────────────────────────────────────────
    g.add_node("query_analyzer",        query_analyzer)
    g.add_node("vector_retriever",      vector_retriever)
    g.add_node("relevance_grader",      relevance_grader)
    g.add_node("generator",             generator)
    g.add_node("hallucination_checker", hallucination_checker)

    # ── Terminal nodes ─────────────────────────────────────────────────────────
    g.add_node("direct_answer", direct_answer)
    g.add_node("no_context",    no_context)

    # ── Loop A: retrieval retry nodes ──────────────────────────────────────────
    g.add_node("record_failed_query", record_failed_query)

    # ── Loop B: generation retry nodes ────────────────────────────────────────
    g.add_node("prepare_strict_gen", prepare_strict_gen)

    # ── Entry point ────────────────────────────────────────────────────────────
    g.set_entry_point("query_analyzer")

    # ── Edge: query_analyzer → branch ─────────────────────────────────────────
    g.add_conditional_edges(
        "query_analyzer",
        route_after_analyzer,
        {
            "direct_answer":    "direct_answer",
            "vector_retriever": "vector_retriever",
        },
    )

    # ── Edge: retriever → grader ───────────────────────────────────────────────
    g.add_edge("vector_retriever", "relevance_grader")

    # ── EDGE A: grader → branch ────────────────────────────────────────────────
    g.add_conditional_edges(
        "relevance_grader",
        route_after_grader,
        {
            "generator":          "generator",
            "record_failed_query": "record_failed_query",
            "no_context":         "no_context",
        },
    )

    # ── Loop A wiring: record_failed_query → query_analyzer → retriever ────────
    #    (query_analyzer picks up failed_queries and uses _RETRY_PROMPT)
    g.add_edge("record_failed_query", "query_analyzer")

    # ── Edge: generator → checker ─────────────────────────────────────────────
    g.add_edge("generator", "hallucination_checker")

    # ── EDGE B: checker → branch ───────────────────────────────────────────────
    g.add_conditional_edges(
        "hallucination_checker",
        route_after_checker,
        {
            "prepare_strict_gen": "prepare_strict_gen",
            END:                  END,
        },
    )

    # ── Loop B wiring: prepare_strict_gen → generator ─────────────────────────
    #    (generator reads generation_hint and generation_retry_count)
    g.add_edge("prepare_strict_gen", "generator")

    # ── Terminal edges ─────────────────────────────────────────────────────────
    g.add_edge("direct_answer", END)
    g.add_edge("no_context",    END)

    return g.compile(checkpointer=checkpointer)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def make_initial_state(question: str, **overrides) -> RAGState:
    """Construct a fully-initialised RAGState for a fresh query turn."""
    base: RAGState = {
        # Input
        "original_query":       question,
        # Stage 1 (filled by query_analyzer)
        "rewritten_query":      "",
        "query_type":           "factual",
        "key_terms":            [],
        "needs_context":        True,
        # Stage 2
        "retrieved_docs":       [],
        # Stage 3
        "graded_docs":          [],
        # Stage 4
        "answer":               "",
        # Stage 5
        "is_grounded":          False,
        "grounding_score":      0.0,
        "unsupported_claims":   [],
        "final_answer":         "",
        # Conversation memory (preserved across turns by MemorySaver)
        "conversation_history": [],
        # Loop A counters
        "retrieval_retry_count":  0,
        "max_retrieval_retries":  MAX_RETRIEVAL_RETRIES,
        "failed_queries":         [],
        # Loop B counters
        "generation_retry_count": 0,
        "max_generation_retries": MAX_GENERATION_RETRIES,
        "generation_hint":        "",
    }
    base.update(overrides)
    return base


# ═══════════════════════════════════════════════════════════════════════════════
# CLI RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from rich.console import Console
    from rich.panel   import Panel
    from rich.text    import Text

    console  = Console()
    question = " ".join(sys.argv[1:]) or "What is Flash Attention?"

    console.print(f"\n[bold cyan]Building RAG graph…[/bold cyan]")
    app = build_graph()
    console.print(f"[bold green]✓ Graph compiled[/bold green]  "
                  f"(Edge A cap={MAX_RETRIEVAL_RETRIES}, "
                  f"Edge B cap={MAX_GENERATION_RETRIES})\n")
    console.print(Panel(question, title="Question", border_style="blue"))

    state = make_initial_state(question)

    # ── Stream with live per-node logging ─────────────────────────────────────
    for step in app.stream(state):
        node_name = list(step.keys())[0]
        node_out  = step[node_name]

        # Choose a colour per node type
        colour = {
            "query_analyzer":        "cyan",
            "vector_retriever":      "blue",
            "relevance_grader":      "magenta",
            "generator":             "yellow",
            "hallucination_checker": "red",
            "record_failed_query":   "bright_red",
            "prepare_strict_gen":    "bright_red",
            "direct_answer":         "green",
            "no_context":            "dark_orange",
        }.get(node_name, "white")

        label = f"[{colour}]▶ {node_name}[/{colour}]"

        if "rewritten_query" in node_out and node_out["rewritten_query"]:
            rq = node_out["rewritten_query"]
            rt = node_out.get("retrieval_retry_count", state.get("retrieval_retry_count", 0))
            console.print(f"  {label}  →  \"{rq[:65]}\"  "
                          f"[dim](retrieval_retry={rt})[/dim]")
        elif "retrieved_docs" in node_out:
            n = len(node_out["retrieved_docs"])
            console.print(f"  {label}  →  {n} docs retrieved")
        elif "graded_docs" in node_out:
            n = len(node_out["graded_docs"])
            console.print(f"  {label}  →  {n} docs passed relevance grading")
        elif "answer" in node_out and node_out["answer"]:
            gr = node_out.get("generation_retry_count", state.get("generation_retry_count", 0))
            console.print(f"  {label}  →  {len(node_out['answer'])} chars  "
                          f"[dim](generation_retry={gr})[/dim]")
        elif "grounding_score" in node_out:
            s    = node_out["grounding_score"]
            ok   = node_out.get("is_grounded", False)
            icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
            unsup = node_out.get("unsupported_claims", [])
            console.print(f"  {label}  →  {icon} grounding={s:.2f}  "
                          f"unsupported_claims={len(unsup)}")
        elif "generation_hint" in node_out:
            hint = node_out["generation_hint"][:80]
            console.print(f"  {label}  →  hint=\"{hint}…\"")
        elif "failed_queries" in node_out:
            fq = node_out["failed_queries"]
            rc = node_out.get("retrieval_retry_count", "?")
            console.print(f"  {label}  →  {len(fq)} failed queries, retry #{rc}")
        elif "final_answer" in node_out:
            console.print(f"  {label}  →  {len(node_out['final_answer'])} chars")
        else:
            console.print(f"  {label}")

        # Keep state in sync for log annotations
        state.update(node_out)

    # ── Final answer panel ─────────────────────────────────────────────────────
    console.print("\n" + "═" * 70)
    console.print(Panel(
        state.get("final_answer", "[red]No answer generated.[/red]"),
        title=(
            f"[bold]Answer[/bold]  "
            f"type={state.get('query_type','?')}  "
            f"grounding={state.get('grounding_score', 0):.2f}  "
            f"retrieval_retries={state.get('retrieval_retry_count', 0)}  "
            f"generation_retries={state.get('generation_retry_count', 0)}"
        ),
        border_style="green",
    ))
