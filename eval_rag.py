"""
eval_rag.py – RAG Evaluation Script using DeepEval and Dynamic LLM Resolver.
Evaluates the LangGraph Agentic RAG pipeline on Faithfulness and Answer Relevancy.
"""
from __future__ import annotations

import os
import asyncio
from dotenv import load_dotenv

# Load env variables from .env file
load_dotenv()

from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Import our compiled LangGraph graph and resolver
from agentic_graph import build_agent_graph, get_llm

# Initialize LangGraph Checkpointer-free graph for standalone evaluation runs
eval_graph = build_agent_graph(checkpointer=None)

class RAGDeepEvalLLM(DeepEvalBaseLLM):
    """DeepEval evaluator model wrapper around our dynamically resolved LLM (Nemotron main / Ollama fallback)."""
    def __init__(self):
        self.chat_model = get_llm()
        self.model_name = getattr(self.chat_model, "model", getattr(self.chat_model, "model_name", "unknown"))
        
    def load_model(self):
        return self.chat_model
        
    def generate(self, prompt: str) -> str:
        return self.chat_model.invoke(prompt).content
        
    async def a_generate(self, prompt: str) -> str:
        res = await self.chat_model.ainvoke(prompt)
        return res.content
        
    def get_model_name(self) -> str:
        return self.model_name


def run_rag_inference(question: str) -> tuple[str, list[str]]:
    """Runs a single question through the RAG graph and returns (answer, retrieval_contexts)."""
    print(f"\n[RAG Inference] Running query: '{question}'...")
    
    # We pass a fresh thread_id for each evaluation case to avoid message history carryover
    cfg = {
        "configurable": {
            "thread_id": f"eval-thread-{os.urandom(4).hex()}",
            "embedding_model": "BAAI/bge-small-en-v1.5"
        },
        "recursion_limit": 50
    }
    
    state = {"messages": [HumanMessage(content=question)]}
    result = eval_graph.invoke(state, cfg)
    messages = result.get("messages", [])
    
    # 1. Extract final answer from messages history
    answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            answer = msg.content
            break
            
    # 2. Extract retrieved context passages
    retrievals = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "retrieve_research_papers":
            # Extract raw chunks from tool message
            chunks = msg.content.split("\n\n---\n\n")
            for chunk in chunks:
                if chunk.startswith("[Doc"):
                    lines = chunk.split("\n", 1)
                    if len(lines) == 2:
                        body = lines[1]
                        if body.startswith("Content: "):
                            body = body[9:]
                        retrievals.append(body)
                        
    return answer, retrievals


def evaluate_pipeline():
    """Runs evaluations on the RAG pipeline using DeepEval metrics."""
    print("=" * 70)
    print("             AGENTIC RAG PIPELINE EVALUATION SUITE            ")
    print("=" * 70)
    
    # Initialize the dynamic evaluator LLM (resolves to ChatGroq based on your env)
    evaluator_llm = RAGDeepEvalLLM()
    print(f"[Evaluator] Using resolved LLM model: '{evaluator_llm.get_model_name()}'")
    
    # Define DeepEval metrics powered by our dynamic evaluator
    faithfulness_metric = FaithfulnessMetric(threshold=0.7, model=evaluator_llm)
    relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model=evaluator_llm)
    
    # Define test questions (preferably queries about ingested papers)
    test_queries = [
        "what is LLM and how is it used as investigative assistant in digital forensics?",
        "What are the main forensic artifacts associated with Ollama and llama.cpp?"
    ]
    
    for idx, question in enumerate(test_queries, 1):
        print(f"\n--- Test Case #{idx} ---")
        answer, context = run_rag_inference(question)
        
        if not context:
            print("[Warning] No context was retrieved from Qdrant. Check that documents are ingested.")
            
        print(f"[RAG Output] Answer:\n{answer}\n")
        
        # Package into DeepEval LLMTestCase
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=context if context else ["No context retrieved."]
        )
        
        # 1. Evaluate Faithfulness (how grounded is the answer in the context)
        print("[Evaluating] Measuring Faithfulness (Grounding)...")
        faithfulness_metric.measure(test_case)
        faith_score = faithfulness_metric.score
        faith_reason = faithfulness_metric.reason
        faith_passed = faithfulness_metric.is_successful()
        
        # 2. Evaluate Answer Relevancy (how well does the answer address the question)
        print("[Evaluating] Measuring Answer Relevancy...")
        relevancy_metric.measure(test_case)
        rel_score = relevancy_metric.score
        rel_reason = relevancy_metric.reason
        rel_passed = relevancy_metric.is_successful()
        
        print("\n" + "-" * 50)
        print(f"RESULTS FOR TEST CASE #{idx}:")
        print("-" * 50)
        print(f"Faithfulness Score: {faith_score:.2f} ({'PASSED' if faith_passed else 'FAILED'})")
        print(f"Reason: {faith_reason}\n")
        print(f"Relevancy Score:  {rel_score:.2f} ({'PASSED' if rel_passed else 'FAILED'})")
        print(f"Reason: {rel_reason}")
        print("-" * 50)

if __name__ == "__main__":
    evaluate_pipeline()
