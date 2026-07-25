"""
config.py – Central configuration for the Advanced RAG Pipeline.
All settings can be overridden via environment variables or a .env file.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
PDF_DIR     = BASE_DIR / "pdfs"
PARSED_DIR  = BASE_DIR / "parsed"

PDF_DIR.mkdir(exist_ok=True)
PARSED_DIR.mkdir(exist_ok=True)

# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_HOST       = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_GRPC_PORT  = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
QDRANT_URL        = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
COLLECTION_NAME   = os.getenv("QDRANT_COLLECTION", "ai_research")

# ── Embeddings ────────────────────────────────────────────────────────────────
# BGE-Small-en-v1.5: 384-dim, fast and efficient local embedding
EMBED_MODEL       = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_BATCH_SIZE  = int(os.getenv("EMBED_BATCH_SIZE", "32"))   # lower to reduce peak memory (prevent bad allocation)
_embed_parallel   = os.getenv("EMBED_PARALLEL")
EMBED_PARALLEL    = int(_embed_parallel) if _embed_parallel is not None else None

_EMBED_DIM_MAP = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5":  768,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-m3":            1024,
}
EMBED_DIM = _EMBED_DIM_MAP.get(EMBED_MODEL, 384)
if EMBED_DIM == 384 and EMBED_MODEL not in _EMBED_DIM_MAP:
    import warnings
    warnings.warn(
        f"Unknown EMBED_MODEL '{EMBED_MODEL}'; assuming EMBED_DIM=384. "
        "Set EMBED_DIM env var to override.",
        stacklevel=1,
    )

# ── Ollama / LLM ──────────────────────────────────────────────────────────────
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "llama3.2:3b")  # swap freely
OLLAMA_TIMEOUT    = float(os.getenv("OLLAMA_TIMEOUT", "300.0"))  # default 5 minutes timeout

# ── Chunking ──────────────────────────────────────────────────────────────────
# Headers that MarkdownHeaderTextSplitter will use as semantic boundaries
MARKDOWN_HEADERS = [
    ("#",   "h1"),
    ("##",  "h2"),
    ("###", "h3"),
]
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K               = int(os.getenv("TOP_K", "6"))
SCORE_THRESH        = float(os.getenv("SCORE_THRESH", "0.3"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.5"))
MAX_WORKERS         = int(os.getenv("MAX_WORKERS", "4"))

# ── Grounding / Validation ────────────────────────────────────────────────────
GROUNDING_THRESHOLD = float(os.getenv("GROUNDING_THRESHOLD", "0.6"))
MAX_SOURCE_CHARS   = int(os.getenv("MAX_SOURCE_CHARS", "4000"))

# ── API Limits / Constraints ──────────────────────────────────────────────────
MAX_HISTORY_TURNS   = int(os.getenv("MAX_HISTORY_TURNS", "20"))
MAX_PDF_MB          = int(os.getenv("MAX_PDF_MB", "50"))

# ── Dynamic Model Resolution (Cloud vs Local) ──────────────────────────────────
_AVAILABLE_MODELS: set[str] = set()
try:
    import ollama
    # Connect to local Ollama instance and fetch available models
    client = ollama.Client(host=OLLAMA_BASE_URL)
    _AVAILABLE_MODELS = {m["model"] for m in client.list().get("models", [])}
except Exception:
    pass

def resolve_model(preferred_models: list[str], fallback_model: str) -> str:
    """Returns the first preferred cloud model available, otherwise the fallback model."""
    for pm in preferred_models:
        if pm in _AVAILABLE_MODELS:
            return pm
    return fallback_model
