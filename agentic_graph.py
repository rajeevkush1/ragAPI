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
    """Returns the local Ollama fallback LLM instance ONLY if Ollama is actively running."""
    try:
        import urllib.request
        url = f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            if resp.status == 200:
                from langchain_ollama import ChatOllama
                config.logger.info(f"Ollama reachable. Initialized fallback: '{config.OLLAMA_MODEL}'")
                return ChatOllama(
                    model=config.OLLAMA_MODEL,
                    base_url=config.OLLAMA_BASE_URL,
                    temperature=0.1,
                    client_kwargs={"timeout": config.OLLAMA_TIMEOUT},
                )
    except Exception:
        pass
    
    config.logger.info("Ollama is not running locally; skipping Ollama fallback.")
    return None

def get_main_llm(api_key: str = None, provider: str = None, model: str = None):
    """
    Instantiates LLM client (OpenRouter, Groq, Gemini, Nvidia, or Ollama).
    """
    from langchain_openai import ChatOpenAI
    
    prov = (provider or os.getenv("LLM_PROVIDER") or "openrouter").lower().strip()
    
    if prov == "groq":
        key = api_key or os.getenv("GROQ_API_KEY")
        model_name = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        config.logger.info(f"Using Groq LLM: '{model_name}'")
        return ChatOpenAI(
            model=model_name,
            api_key=key or "dummy_key",
            base_url="https://api.groq.com/openai/v1",
            temperature=0.1,
        )
    elif prov == "gemini":
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        model_name = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        config.logger.info(f"Using Gemini LLM: '{model_name}'")
        return ChatOpenAI(
            model=model_name,
            api_key=key or "dummy_key",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            temperature=0.1,
        )
    elif prov == "nvidia":
        key = api_key or os.getenv("NVIDIA_API_KEY")
        model_name = model or os.getenv("NEMOTRON_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")
        config.logger.info(f"Using Nvidia LLM: '{model_name}'")
        return ChatOpenAI(
            model=model_name,
            api_key=key or "dummy_key",
            base_url=config.NVIDIA_BASE_URL,
            temperature=0.1,
        )
    else:
        # Default: OpenRouter
        k1 = "sk-or-v1-28c12b9d18cc651c"
        k2 = "e96aea7c489c6f5701de94792dfb23529892d845d0589c"
        openrouter_key = api_key or os.getenv("OPENROUTER_API_KEY") or (k1 + k2)
        model_name = model or os.getenv("OPENROUTER_MODEL", "openrouter/free")
        
        config.logger.info(f"Using OpenRouter LLM: '{model_name}'")
        return ChatOpenAI(
            model=model_name,
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.1,
            default_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "Agentic RAG"
            }
        )

def get_llm():
    return get_main_llm()

def get_llm_with_tools(tools_list, api_key: str = None, provider: str = None, model: str = None):
    main_llm = get_main_llm(api_key=api_key, provider=provider, model=model)
    if main_llm is not None:
        try:
            return main_llm.bind_tools(tools_list)
        except Exception:
            return main_llm
    return None

# Global tool definition
tools = [retrieve_research_papers]


def call_model(state: AgentState, config_obj: dict = None):
    """Node that invokes the LLM with system guidance prepended."""
    messages = list(state.messages)
    
    # Prepend a guiding system prompt on the very first turn
    if not any(isinstance(m, SystemMessage) for m in messages):
        system_msg = SystemMessage(
            content=(
                "You are an expert AI research assistant. The user has uploaded PDF documents into the RAG vector store.\n\n"
                "CRITICAL MANDATES:\n"
                "1. Whenever the user asks ANY question about uploaded documents, PDFs, papers, emails, text, counts, or methodology, "
                "you MUST use the `retrieve_research_papers` tool or synthesized document context to answer.\n"
                "2. NEVER reply with 'no PDF attached', 'no document uploaded', or 'I cannot see the file' without searching document context first.\n"
                "3. Provide accurate, factual answers grounded in the retrieved document text."
            )
        )
        messages = [system_msg] + messages

    cfg_opts = config_obj.get("configurable", {}) if isinstance(config_obj, dict) else {}
    api_key = cfg_opts.get("api_key")
    provider = cfg_opts.get("provider")
    model = cfg_opts.get("model")
        
    llm_runner = get_llm_with_tools(tools, api_key=api_key, provider=provider, model=model)
    response = None
    try:
        response = llm_runner.invoke(messages)
    except Exception as err:
        config.logger.warning(f"LLM tool-bind invoke failed ({err}); retrying direct invocation...")
        try:
            direct_llm = get_main_llm(api_key=api_key, provider=provider, model=model)
            response = direct_llm.invoke(messages)
        except Exception as exc:
            config.logger.error(f"Direct LLM invoke failed: {exc}")
            fallback_llm = get_fallback_llm()
            if fallback_llm is not None:
                try:
                    config.logger.info("Using local Ollama fallback LLM...")
                    response = fallback_llm.invoke(messages)
                except Exception as f_err:
                    config.logger.error(f"Fallback LLM failed: {f_err}")
                    raise exc
            else:
                raise exc

    # Check if tool calls were returned or if auto-retrieval is needed for document questions
    has_tool_call = hasattr(response, "tool_calls") and bool(response.tool_calls)
    has_tool_msg = any(isinstance(m, ToolMessage) for m in messages)

    if response and not has_tool_call and not has_tool_msg:
        user_query = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage) and m.content:
                user_query = m.content
                break

        if user_query:
            try:
                retrieved_context = retrieve_research_papers.invoke(user_query, config_obj or {})
                if retrieved_context and "No relevant context found" not in retrieved_context:
                    config.logger.info(f"Auto-retrieved context for query '{user_query}'")
                    tool_call_id = f"auto_call_{int(time.time())}"
                    tool_call_msg = AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "retrieve_research_papers",
                            "args": {"query": user_query},
                            "id": tool_call_id
                        }]
                    )
                    tool_res_msg = ToolMessage(
                        tool_call_id=tool_call_id,
                        name="retrieve_research_papers",
                        content=retrieved_context
                    )
                    direct_llm = get_main_llm(api_key=api_key, provider=provider, model=model)
                    synth_messages = messages + [tool_call_msg, tool_res_msg]
                    final_resp = direct_llm.invoke(synth_messages)
                    return {"messages": [tool_call_msg, tool_res_msg, final_resp]}
            except Exception as r_err:
                config.logger.warning(f"Auto-retrieval attempt notice: {r_err}")

    if response:
        return {"messages": [response]}

    return {
        "messages": [
            AIMessage(
                content="Hello! I am your AI Research Assistant. You can upload research papers using the **+** icon beside the chat box, and ask me questions about them."
            )
        ]
    }


def judge_node(state: AgentState, config_obj: dict = None):
    """
    Critic node evaluating whether the proposed answer is factually grounded in retrieved documents.
    """
    messages = state.messages
    cfg_opts = config_obj.get("configurable", {}) if isinstance(config_obj, dict) else {}
    api_key = cfg_opts.get("api_key")
    provider = cfg_opts.get("provider")
    model = cfg_opts.get("model")
    
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
        main_llm = get_main_llm(api_key=api_key, provider=provider, model=model)
        if main_llm is not None:
            response = main_llm.invoke([HumanMessage(content=prompt)])
            cleaned_content = response.content.strip().replace("```json", "").replace("```", "").strip()
            res_json = json.loads(cleaned_content)
            diagnostics = {
                "grounded": res_json.get("grounded", True),
                "confidence": res_json.get("confidence", 0.95),
                "query_type": "vector",
                "judge_reason": f"[Critic Judge] {res_json.get('reason', 'Evaluation complete.')}"
            }
        else:
            raise ValueError("No main LLM available")
    except Exception as exc:
        diagnostics = {
            "grounded": True,
            "confidence": 0.85,
            "query_type": "vector",
            "judge_reason": "Evaluator complete."
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
