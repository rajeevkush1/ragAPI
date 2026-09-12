"""
config.py – Central configuration for the Advanced RAG Pipeline.
All settings can be overridden via environment variables or a .env file.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import logging
from logging.handlers import RotatingFileHandler

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
PDF_DIR     = BASE_DIR / "pdfs"
PARSED_DIR  = BASE_DIR / "parsed"
LOGS_DIR    = BASE_DIR / "logs"

PDF_DIR.mkdir(exist_ok=True)
PARSED_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

LOG_FILE = LOGS_DIR / "app.log"

# ── Logging Configuration ─────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("rag_api")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(file_formatter)
        logger.addHandler(console_handler)
    return logger

logger = setup_logging()

# ── Qdrant ────────────────────────────────────────────────────────────────────
def _resolve_qdrant_host() -> str:
    env_host = os.getenv("QDRANT_HOST")
    if env_host:
        return env_host
    try:
        import socket
        socket.gethostbyname("qdrant")
        return "qdrant"
    except Exception:
        return "localhost"

QDRANT_HOST       = _resolve_qdrant_host()
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_GRPC_PORT  = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
QDRANT_URL        = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
COLLECTION_NAME   = os.getenv("QDRANT_COLLECTION", "ai_research")

# Limit ONNX/OpenMP thread allocation to prevent peak RAM spikes/OOM on CPU VMs
os.environ["OMP_NUM_THREADS"] = os.getenv("OMP_NUM_THREADS", "2")
os.environ["MKL_NUM_THREADS"] = os.getenv("MKL_NUM_THREADS", "2")
os.environ["OPENBLAS_NUM_THREADS"] = os.getenv("OPENBLAS_NUM_THREADS", "2")

# ── Embeddings ────────────────────────────────────────────────────────────────
# BGE-Small-en-v1.5: 384-dim, fast and efficient local embedding
EMBED_MODEL       = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_BATCH_SIZE  = int(os.getenv("EMBED_BATCH_SIZE", "16"))   # lower to 16 to reduce peak memory
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

# ── LLM Configuration (Main: Nemotron | Fallback: Ollama) ─────────────────────
NEMOTRON_MODEL    = os.getenv("NEMOTRON_MODEL", os.getenv("NVIDIA_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct"))
NVIDIA_API_KEY    = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL   = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "openrouter/free")
OPENROUTER_BASE_URL= os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "llama3.2:3b")  # fallback model
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
def resolve_model(preferred_models: list[str], fallback_model: str) -> str:
    """Returns the first preferred cloud model available, otherwise the fallback model."""
    try:
        import ollama
        client = ollama.Client(host=OLLAMA_BASE_URL, timeout=1.0)
        available = {m["model"] for m in client.list().get("models", [])}
        for pm in preferred_models:
            if pm in available:
                return pm
    except Exception:
        pass
    return fallback_model
