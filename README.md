# Advanced Agentic RAG Pipeline

**LangGraph ReAct Loop + Qdrant + local FastEmbed + FastAPI UI + Nvidia Nemotron Main LLM + Ollama Fallback**

A production-ready Agentic RAG stack for AI research papers — featuring dynamic local embedding model selection, hybrid dense-sparse retrieval (BM25 + Dense + RRF), Nvidia Nemotron main LLM with local Ollama fallback, and a premium React chat interface.

### 🔗 Quick Links
* 🌐 **Live Web Application**: [http://127.0.0.1:8001/](http://127.0.0.1:8001/)
* ⚡ **FastAPI Swagger Docs**: [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

---

## Architecture Diagram

```
PDF INGESTION FLOW:
┌───────────┐      ┌─────────────┐      ┌─────────────────┐      ┌─────────────────────┐      ┌────────────┐
│ PDF Paper │ ───► │ pymupdf4llm │ ───► │ Markdown Header │ ───► │   Local FastEmbed   │ ───► │   Qdrant   │
└───────────┘      └─────────────┘      │  & Char Splitter│      │ (BGE-Small/MiniLM)  │      │ Vector DB  │
                                        └─────────────────┘      └─────────────────────┘      └────────────┘

QUERY AGENT LOOP (LangGraph ReAct):
                    ┌──────────────────────────────┐
                    │      Query / User Prompt     │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
         ┌──────────────────────────────────────────────────┐
         │      ReAct Agent LLM Router Decision Loop        │
         │ (Main: Nemotron ──► Fallback: Local Ollama)      │
         └───────────┬──────────────────────────▲───────────┘
                     │                          │
           Needs Context?                  Yield Answer
                     │                          │
                     ▼                          │
        ┌─────────────────────────┐             │
        │ retrieve_research_papers│             │
        │         (Tool)          │             │
        └────────────┬────────────┘             │
                     │                          │
        ┌────────────▼────────────┐             │
        │  Local Hybrid Retrieve: │             │
        │  1. FastEmbed (Dense)   │             │
        │  2. BM25 (Sparse)       │             │
        │  3. RRF (Merge Rank)    │             │
        └────────────┬────────────┘             │
                     │                          │
                     └──────────────────────────┘
```

---

## Key Features

1. **Dynamic Embedding Selection:** Toggle between `BAAI/bge-small-en-v1.5` (More Accurate) and `BAAI/bge-tiny-en-v1.5` (Faster, mapped to `sentence-transformers/all-MiniLM-L6-v2`) directly in the UI. Both map to a 384-dimensional vector space, preventing database conflicts.
2. **On-Device Embeddings:** Generating vectors locally via ONNX-backed FastEmbed — completely eliminating third-party embedding API costs and keys (`HF_TOKEN` is discarded).
3. **Adaptive LLM Routing & Fallback:** Nemotron main LLM with seamless Ollama fallback:
   * **Nvidia Nemotron** (`nvidia/llama-3.1-nemotron-70b-instruct`) $\rightarrow$ Primary main LLM (via `NVIDIA_API_KEY` or `OPENROUTER_API_KEY`).
   * **Ollama** (`llama3.2:3b`) $\rightarrow$ Local fallback LLM active when offline, missing cloud keys, or during runtime API errors.
4. **Hybrid Retrieval (RRF):** Merges semantic dense matches from Qdrant with lexical sparse matches from Rank-BM25 using Reciprocal Rank Fusion (RRF) for optimal context quality.
5. **SSE Streaming UI:** Real-time token streaming and step-by-step pipeline node tracking delivered directly to the custom React-based frontend.

---

## Quick Start

### 1. Start Qdrant Vector DB
Ensure you have Docker running, then start the Qdrant instance:
```bash
docker compose up -d
```

### 2. Configure Environment Keys
Copy `.env.example` to `.env` and configure your API keys:
```env
# Main LLM: Nvidia Nemotron (Nvidia AI Endpoints or OpenRouter)
NVIDIA_API_KEY=your_nvidia_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Fallback LLM: Local Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```
*(If no cloud keys are configured, it will seamlessly fall back to your local Ollama server).*

### 3. Install Dependencies
Initialize your virtual environment and install package dependencies:
```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Start the Agentic API Server
Start the FastAPI server on port `8001`:
```bash
.venv\Scripts\python agentic_api.py
```

### 5. Access the Chat Interface
Open your web browser and navigate to **[http://127.0.0.1:8001/](http://127.0.0.1:8001/)** to select your embedding model, upload PDFs, and ask questions!

---

## File Structure

| File | Purpose |
|------|---------|
| `agentic_api.py` | FastAPI server serving endpoints (Query, Ingest, Documents) and the React UI. |
| `agentic_graph.py` | LangGraph ReAct agent loop definition and LLM resolver. |
| `agentic_state.py` | Agent message list state utilizing Pydantic. |
| `agentic_tools.py` | Retrieval tool conducting local FastEmbed + Qdrant + BM25 + RRF query search. |
| `ingest.py` | PDF ingestion pipeline (Parsing $\rightarrow$ Splitting $\rightarrow$ Local FastEmbed $\rightarrow$ Qdrant). |
| `config.py` | Global settings, directories, and model dimensions. |
| `index.html` | Frontend React UI supporting chat, document management, and model selection. |
| `docker-compose.yml` | Sets up Qdrant container with persistent volume mapping. |
| `.env` | Holds active API keys and custom configuration overrides. |
