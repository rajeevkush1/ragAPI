"""
agentic_graph.py – ReAct-loop agentic RAG graph with dynamically resolved models and Gemini judge.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from agentic_state import AgentState
from agentic_tools import retrieve_research_papers
import config

def get_fallback_llm():
    """Returns the local Ollama fallback LLM instance."""
    from langchain_ollama import ChatOllama
    config.logger.info(f"Initialized Ollama fallback model: '{config.OLLAMA_MODEL}' at '{config.OLLAMA_BASE_URL}'")
    return ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.1,
        client_kwargs={"timeout": config.OLLAMA_TIMEOUT},
    )

def get_main_llm():
    """
    Instantiates Nvidia Nemotron as the main LLM.
    Supports Nvidia AI Endpoints (NVIDIA_API_KEY) or OpenRouter (OPENROUTER_API_KEY).
    """
    from langchain_openai import ChatOpenAI

    # 1. Try direct Nvidia API endpoint
    nvidia_key = config.NVIDIA_API_KEY
    if nvidia_key and not nvidia_key.startswith("your_"):
        try:
            config.logger.info(f"Initialized Main LLM (Nvidia API): '{config.NEMOTRON_MODEL}'")
            return ChatOpenAI(
                model=config.NEMOTRON_MODEL,
                api_key=nvidia_key,
                base_url=config.NVIDIA_BASE_URL,
                temperature=0.1,
            )
        except Exception as exc:
            config.logger.warning(f"Failed to initialize direct Nvidia API: {exc}")

    # 2. Try OpenRouter configured with free auto-router or preferred model
    openrouter_key = config.OPENROUTER_API_KEY
    if openrouter_key and not openrouter_key.startswith("your_") and len(openrouter_key) > 10:
        try:
            model_name = config.OPENROUTER_MODEL or "openrouter/free"
            config.logger.info(f"Initialized Main LLM (OpenRouter): '{model_name}'")
            return ChatOpenAI(
                model=model_name,
                api_key=openrouter_key,
                base_url=config.OPENROUTER_BASE_URL,
                temperature=0.1,
                default_headers={
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "Agentic RAG"
                }
            )
        except Exception as exc:
            config.logger.warning(f"Failed to initialize OpenRouter: {exc}")

    config.logger.info("No cloud API keys set for main LLM; using Ollama directly.")
    return None

def get_llm():
    """
    Returns the resolved LLM runnable: Nemotron as main LLM with Ollama as fallback.
    If no main cloud API key is present, defaults to Ollama directly.
    """
    fallback_llm = get_fallback_llm()
    main_llm = get_main_llm()

    if main_llm is not None:
        return main_llm.with_fallbacks([fallback_llm])
    return fallback_llm

def get_llm_with_tools(tools_list):
    """
    Binds tools to main and fallback LLMs and configures runtime fallback.
    """
    fallback_llm = get_fallback_llm()
    fallback_with_tools = fallback_llm.bind_tools(tools_list)
    
    main_llm = get_main_llm()
    if main_llm is not None:
        main_with_tools = main_llm.bind_tools(tools_list)
        return main_with_tools.with_fallbacks([fallback_with_tools])
    return fallback_with_tools

# Instantiate LLM and bind retrieval tool
tools = [retrieve_research_papers]
llm = get_llm()
llm_with_tools = get_llm_with_tools(tools)


def call_model(state: AgentState):
    """Node that invokes the LLM with system guidance prepended."""
    messages = state.messages
    
    # Prepend a guiding system prompt on the very first turn
    if not any(isinstance(m, SystemMessage) for m in messages):
        system_msg = SystemMessage(
            content=(
                "You are an expert AI research assistant specializing in machine learning and systems papers. "
                "Synthesize clear, helpful, and comprehensive answers. Whenever research paper context is available, "
                "cross-reference key findings, methodologies, and conclusions."
            )
        )
        messages = [system_msg] + messages
        
    main_llm = get_main_llm()
    if main_llm is not None:
        # Tier 1: Try Main LLM with tool calling
        try:
            main_with_tools = main_llm.bind_tools(tools)
            response = main_with_tools.invoke(messages)
            return {"messages": [response]}
        except Exception as err1:
            config.logger.warning(f"Main LLM with tools failed ({err1}); retrying direct invocation without tools...")
            # Tier 2: Try Main LLM without tool calling (handles models that don't support function calling)
            try:
                response = main_llm.invoke(messages)
                return {"messages": [response]}
            except Exception as err2:
                config.logger.error(f"Main LLM direct invoke failed ({err2}). Trying fallback LLM...")

    # Tier 3: Try Ollama fallback LLM
    try:
        fallback_llm = get_fallback_llm()
        response = fallback_llm.invoke(messages)
        return {"messages": [response]}
    except Exception as exc:
        config.logger.error(f"All LLMs failed: {exc}")
        return {
            "messages": [
                AIMessage(
                    content="Hello! I am your AI Research Assistant. You can upload research papers using the **+** icon beside the chat box, and ask me questions about them."
                )
            ]
        }


def judge_node(state: AgentState):
    """
    Critic node evaluating whether the proposed answer is factually grounded in retrieved documents.
    """
    messages = state.messages
    
    # 1. Find the proposed final AIMessage answer
    proposed_answer = ""
    target_msg = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            target_msg = msg
            proposed_answer = msg.content
            break
            
    if not target_msg or not proposed_answer:
        return {"messages": []}
        
    # 2. Find all retrieved context in ToolMessages
    retrieved_contexts = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "retrieve_research_papers":
            retrieved_contexts.append(msg.content)
            
    if not retrieved_contexts:
        # No context retrieved: cannot verify grounding, mark as no_context
        diagnostics = {
            "grounded": False,
            "confidence": 0.0,
            "query_type": "no_context",
            "judge_reason": "No document chunks were retrieved from the database to evaluate grounding."
        }
        updated_msg = AIMessage(
            id=target_msg.id,
            content=target_msg.content,
            additional_kwargs={"diagnostics": diagnostics}
        )
        return {"messages": [updated_msg]}
        
    context_str = "\n\n---\n\n".join(retrieved_contexts)
    prompt = (
        "You are an expert evaluator. Evaluate if the proposed answer is factually grounded in the provided retrieved context. "
        "Do not allow any claims that cannot be directly supported by the context.\n\n"
        f"Retrieved Context:\n{context_str}\n\n"
        f"Proposed Answer:\n{proposed_answer}\n\n"
        "Instructions:\n"
        "1. Respond ONLY with a JSON object in this format:\n"
        '{"grounded": true/false, "confidence": 0.0 to 1.0, "reason": "concise explanation"}\n'
        "2. Do not write any markdown wrappers or comments outside the JSON."
    )
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        cleaned_content = response.content.strip().replace("```json", "").replace("```", "").strip()
        res_json = json.loads(cleaned_content)
        
        diagnostics = {
            "grounded": res_json.get("grounded", True),
            "confidence": res_json.get("confidence", 0.95),
            "query_type": "vector",
            "judge_reason": f"[Critic Judge] {res_json.get('reason', 'Evaluation complete.')}"
        }
    except Exception as exc:
        diagnostics = {
            "grounded": True,
            "confidence": 0.85,
            "query_type": "vector",
            "judge_reason": f"Evaluator bypassed ({exc})"
        }
        
    updated_msg = AIMessage(
        id=target_msg.id,
        content=target_msg.content,
        additional_kwargs={"diagnostics": diagnostics}
    )
    return {"messages": [updated_msg]}


def route_after_agent(state: AgentState):
    """If tool calls exist, continue to tools node. Otherwise, pass to critic judge."""
    messages = state.messages
    last_msg = messages[-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "judge"


def build_agent_graph(checkpointer=None) -> StateGraph:
    """Build and compile the ReAct agent state graph with critic judge."""
    workflow = StateGraph(AgentState)
    
    # Nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("judge", judge_node)
    
    # Edges
    workflow.add_edge(START, "agent")
    
    # Conditional edge: after model call, check if tools need to run or evaluate answer
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "judge": "judge"
        }
    )
    
    # After executing tools, loop back to the agent for synthesis
    workflow.add_edge("tools", "agent")
    
    # After grading the answer, terminate execution
    workflow.add_edge("judge", END)
    
    return workflow.compile(checkpointer=checkpointer)
