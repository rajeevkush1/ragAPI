"""
agentic_state.py – Pydantic state schema for the Agentic RAG LangGraph pipeline.
"""
from __future__ import annotations

from typing import Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(BaseModel):
    """
    Pydantic BaseModel representing the state of the agentic RAG graph.
    The 'messages' field stores the conversation history and LLM reasoning steps.
    It uses the 'add_messages' reducer to append new messages instead of overwriting them.
    """
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
