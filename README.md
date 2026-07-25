# Advanced RAG Pipeline

**LangGraph + Qdrant + BGE-M3 + Ollama + Nougat**  
A production-ready RAG stack for AI research papers — preserves math, tables, and structure.

---

## Architecture

```
PDF
 │
 ▼ Nougat OCR
Structured Markdown (.mmd)   ← preserves LaTeX equations, tables
 │
 ▼ MarkdownHeaderTextSplitter (Pass 1 — semantic boundaries)
 ▼ RecursiveCharacterTextSplitter (Pass 2 — size cap)
Chunks (≤1000 tokens, 200 overlap)
 │
 ▼ BGE-M3 (FastEmbed, 1024-dim, multilingual)
Dense Vectors
 │
 ▼ Qdrant (Docker, local)
Vector Store

QUERY
 │
 ▼ BGE-M3 embed question
 ▼ Qdrant cosine search (TOP_K=6)
 ▼ LLM relevance grading (Ollama)
 ▼ Ollama generate answer
ANSWER + sources
```

---

## Quick Start

### 1. Start Qdrant
```bash
docker compose up -d
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Pull your Ollama model
```bash
ollama pull llama3.2:3b
```

### 4. Ingest PDFs
```bash
# Single PDF
python ingest.py paper.pdf

# Entire folder
python ingest.py ./pdfs/
```

### 5. Start the API
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```
Then visit **http://localhost:8000/docs** for the Swagger UI.

### 6. Query via CLI
```bash
python rag_chain.py "What is the key insight of Flash Attention?"
```

### 7. Query via API
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Flash Attention?"}'
```

---

## File Structure

| File | Purpose |
|------|---------|
| `config.py` | Central config — all tuneable via env vars |
| `ingest.py` | PDF → Nougat → split → embed → Qdrant |
| `rag_chain.py` | LangGraph graph: retrieve → grade → generate |
| `api.py` | FastAPI REST server |
| `docker-compose.yml` | Qdrant with persistent volume |
| `.env.example` | Environment template |
| `pdfs/` | Drop your PDFs here |
| `parsed/` | Nougat .mmd output cache |

---

## Configuration

All settings in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `llama3.2:3b` | Any model from `ollama list` |
| `EMBED_MODEL` | `BAAI/bge-m3` | FastEmbed model name |
| `CHUNK_SIZE` | `1000` | Max chars per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `6` | Retrieved chunks per query |
| `SCORE_THRESH` | `0.30` | Min cosine similarity |
| `QDRANT_COLLECTION` | `ai_research` | Collection name |

---

## Why This Stack?

| Component | Why |
|-----------|-----|
| **Nougat** | Academic PDF parsing — preserves math (LaTeX), tables, figures |
| **MarkdownHeaderTextSplitter** | Respects section boundaries before size-splitting |
| **BGE-M3** | Multilingual, 1024-dim, SOTA on academic retrieval benchmarks |
| **Qdrant** | Fast HNSW cosine search, rich metadata filtering |
| **LangGraph** | Explicit graph control flow — easy to add re-ranking, routing |
| **Ollama** | Local LLM — no API costs, full privacy |
