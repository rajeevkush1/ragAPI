# ── Build/Run stage for Advanced RAG API ──────────────────────────────────────
FROM python:3.11-slim

# Install system dependencies (build-essential needed for some pyproject builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install dependencies (use --no-cache-dir to keep image slim)
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache FastEmbed model to bake it into the image
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

# Copy application source code
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Launch server
CMD ["python", "api.py"]
