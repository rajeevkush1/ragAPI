"""
ingest.py – Full PDF → Qdrant ingestion pipeline.

Pipeline:
  1. PDF parsing  → structured Markdown
       Primary  : pymupdf4llm  (Python 3.14-native, fast, LangChain-integrated)
       Fallback : marker-pdf   (high-fidelity for math/tables, requires torch)
  2. MarkdownHeaderTextSplitter → semantic header-aware chunks
  3. RecursiveCharacterTextSplitter → size-bounded sub-chunks
  4. BGE-M3 via FastEmbed → 1024-dim dense vectors
  5. Qdrant upsert with rich metadata payload

Usage:
  python ingest.py paper.pdf
  python ingest.py ./pdfs/              # batch-ingest entire folder
  python ingest.py paper.pdf --parser marker   # force marker-pdf
  python ingest.py --help
"""
from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except (AttributeError, TypeError):
    pass

import hashlib
import time
from enum import Enum
from pathlib import Path
from typing import Literal

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding

import config

# ── Parser availability ───────────────────────────────────────────────────────
try:
    import pymupdf4llm  # type: ignore
    _PYMUPDF_OK = True
except ImportError:
    _PYMUPDF_OK = False

try:
    from marker.convert import convert_single_pdf  # type: ignore
    from marker.models import load_all_models       # type: ignore
    _MARKER_OK = True
except ImportError:
    _MARKER_OK = False

# ── Globals ───────────────────────────────────────────────────────────────────
app     = typer.Typer(pretty_exceptions_enable=False)
console = Console()

# ── Retry helper ──────────────────────────────────────────────────────────────
def _retry(fn, *, retries: int = 3, base_delay: float = 1.0, label: str = ""):
    """
    Call `fn()` with exponential back-off.  Raises the last exception after
    `retries` attempts.  Back-off schedule: 1 s → 2 s → 4 s …
    """
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            console.log(
                f"[yellow]⚠ {label or 'operation'} failed "
                f"(attempt {attempt + 1}/{retries}): {exc}. "
                f"Retrying in {delay:.0f}s…[/yellow]"
            )
            time.sleep(delay)

_EMBED_MODEL: TextEmbedding | None = None

def get_embed_model() -> TextEmbedding:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        console.log(f"[cyan]Loading embedding model[/cyan] [bold]{config.EMBED_MODEL}[/bold]…")
        _EMBED_MODEL = TextEmbedding(model_name=config.EMBED_MODEL)
    return _EMBED_MODEL


def unload_embed_model() -> None:
    """Unloads the embedding model from memory and runs garbage collection."""
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        del _EMBED_MODEL
        _EMBED_MODEL = None
        import gc
        gc.collect()


def unload_embed_model() -> None:
    """Unloads the embedding model from memory and runs garbage collection."""
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        del _EMBED_MODEL
        _EMBED_MODEL = None
        import gc
        gc.collect()


# ── Step 1: PDF Parsing ───────────────────────────────────────────────────────
_marker_models = None  # loaded once on first marker call


def _parse_with_pymupdf(pdf_path: Path) -> str:
    """Fast, Python-3.14-compatible parser. Handles multi-column, tables, images."""
    md = pymupdf4llm.to_markdown(
        str(pdf_path),
        show_progress=False,
    )
    return md


def _parse_with_marker(pdf_path: Path) -> str:
    """High-fidelity parser for math-heavy papers (requires torch)."""
    global _marker_models
    if _marker_models is None:
        console.log("[cyan]Loading marker models (first run, takes ~30s)…[/cyan]")
        _marker_models = load_all_models()
    full_text, _metadata, _images = convert_single_pdf(
        str(pdf_path), _marker_models
    )
    return full_text


def parse_pdf(
    pdf_path: Path,
    parser: Literal["auto", "pymupdf", "marker"] = "auto",
) -> tuple[str, str]:
    """
    Parse a PDF to Markdown. Returns (markdown_text, parser_used).

    parser="auto"   → try pymupdf4llm first; fall back to marker-pdf
    parser="pymupdf" → force pymupdf4llm
    parser="marker"  → force marker-pdf (higher quality, needs torch)
    """
    cache_path = config.PARSED_DIR / (pdf_path.stem + ".md")

    if cache_path.exists():
        console.log(f"[yellow]Cache hit[/yellow] – skipping parse for {pdf_path.name}")
        return cache_path.read_text(encoding="utf-8"), "cache"

    if parser in ("auto", "pymupdf") and _PYMUPDF_OK:
        console.log(f"[cyan]pymupdf4llm[/cyan] parsing [bold]{pdf_path.name}[/bold]…")
        try:
            md = _parse_with_pymupdf(pdf_path)
            cache_path.write_text(md, encoding="utf-8")
            console.log(f"[green]✓ Parsed (pymupdf4llm)[/green] → {len(md):,} chars")
            return md, "pymupdf4llm"
        except Exception as e:
            console.log(f"[yellow]pymupdf4llm failed ({e}), trying marker…[/yellow]")

    if parser in ("auto", "marker") and _MARKER_OK:
        console.log(f"[cyan]marker-pdf[/cyan] parsing [bold]{pdf_path.name}[/bold]…")
        try:
            md = _parse_with_marker(pdf_path)
            cache_path.write_text(md, encoding="utf-8")
            console.log(f"[green]✓ Parsed (marker)[/green] → {len(md):,} chars")
            return md, "marker-pdf"
        except Exception as e:
            console.log(f"[red]marker-pdf also failed: {e}[/red]")
            raise

    raise RuntimeError(
        "No PDF parser available. Run: pip install pymupdf4llm\n"
        "(or: pip install marker-pdf  for math-heavy papers)"
    )


# ── Step 2: Markdown Splitting ────────────────────────────────────────────────
def split_markdown(text: str, source_metadata: dict) -> list[dict]:
    """
    Two-pass splitting strategy:
      Pass 1 – MarkdownHeaderTextSplitter respects semantic section boundaries.
      Pass 2 – RecursiveCharacterTextSplitter enforces a hard size ceiling.

    Returns a list of dicts: {"text": str, "metadata": dict}
    """
    # Pass 1 – header-aware semantic boundaries
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=config.MARKDOWN_HEADERS,
        strip_headers=False,  # keep headers in chunk for context
    )
    header_docs = header_splitter.split_text(text)

    # Pass 2 – size-bounded sub-chunks
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks: list[dict] = []
    for doc in header_docs:
        sub_chunks = char_splitter.split_text(doc.page_content)
        for idx, chunk_text in enumerate(sub_chunks):
            chunk_text = chunk_text.strip()
            if len(chunk_text) < 40:          # skip near-empty chunks
                continue
            metadata = {
                **source_metadata,
                **doc.metadata,               # h1/h2/h3 hierarchy from splitter
                "chunk_index": idx,
                "chunk_id": hashlib.md5(chunk_text.encode()).hexdigest(),
            }
            chunks.append({"text": chunk_text, "metadata": metadata})

    return chunks


# ── Step 3: Embed chunks with BGE-M3 ─────────────────────────────────────────
def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Add a 'vector' key to each chunk dict using BGE-M3."""
    model   = get_embed_model()
    texts   = [c["text"] for c in chunks]
    vectors = list(
        model.embed(
            texts,
            batch_size=config.EMBED_BATCH_SIZE,
            parallel=config.EMBED_PARALLEL,
        )
    )  # generator → list of np arrays

    for chunk, vec in zip(chunks, vectors):
        chunk["vector"] = vec.tolist()

    return chunks


# ── Step 4: Upsert to Qdrant ──────────────────────────────────────────
def get_qdrant_client() -> QdrantClient:
    """Return a QdrantClient, verifying connectivity with up to 3 retries."""
    def _connect():
        client = QdrantClient(url=config.QDRANT_URL, timeout=60)  # 60 s covers large batches
        client.get_collections()  # connectivity probe
        return client

    return _retry(_connect, retries=3, base_delay=2.0, label="Qdrant connection")


def ensure_collection(client: QdrantClient) -> None:
    """Create collection if it does not already exist (retries on transient errors)."""
    def _ensure():
        existing = {c.name for c in client.get_collections().collections}
        if config.COLLECTION_NAME not in existing:
            console.log(f"[cyan]Creating collection[/cyan] [bold]{config.COLLECTION_NAME}[/bold]…")
            client.create_collection(
                collection_name=config.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=config.EMBED_DIM,
                    distance=Distance.COSINE,
                ),
            )
            console.log(f"[green]✓ Collection created[/green]")
        else:
            console.log(f"[yellow]Collection exists[/yellow] – reusing [bold]{config.COLLECTION_NAME}[/bold]")

    _retry(_ensure, retries=3, base_delay=2.0, label="ensure_collection")


def upsert_chunks(client: QdrantClient, chunks: list[dict]) -> None:
    """Batch-upsert embedded chunks into Qdrant with per-batch retry."""
    points = [
        PointStruct(
            id=int(c["metadata"]["chunk_id"][:8], 16),  # deterministic uint32 from hash
            vector=c["vector"],
            payload={
                "text":       c["text"],
                **c["metadata"],
            },
        )
        for c in chunks
    ]

    BATCH = 64
    for i in range(0, len(points), BATCH):
        batch = points[i : i + BATCH]
        batch_label = f"upsert batch {i // BATCH + 1}/{(len(points) + BATCH - 1) // BATCH}"
        _retry(
            lambda b=batch: client.upsert(
                collection_name=config.COLLECTION_NAME,
                points=b,
                wait=True,
            ),
            retries=3,
            base_delay=1.0,
            label=batch_label,
        )


# ── Top-level ingest function ─────────────────────────────────────────────────
def ingest_pdf(
    pdf_path: Path,
    parser: Literal["auto", "pymupdf", "marker"] = "auto",
) -> dict:
    """Full pipeline for a single PDF. Returns stats dict."""
    pdf_path = pdf_path.resolve()
    console.rule(f"[bold blue]Ingesting[/bold blue] {pdf_path.name}")

    # 1. Parse
    markdown, parser_used = parse_pdf(pdf_path, parser=parser)

    source_metadata = {
        "source":    pdf_path.name,
        "parser":    parser_used,
        "file_size": pdf_path.stat().st_size,
    }

    # 2. Split
    chunks = split_markdown(markdown, source_metadata)
    console.log(f"[green]✓[/green] {len(chunks)} chunks produced")

    # 3. Embed
    console.log("[cyan]Embedding with BGE-M3…[/cyan]")
    t0     = time.perf_counter()
    chunks = embed_chunks(chunks)
    elapsed = time.perf_counter() - t0
    console.log(f"[green]✓ Embedded[/green] in {elapsed:.1f}s")
    
    # Free embedding model memory immediately
    unload_embed_model()
    
    # Free embedding model memory immediately
    unload_embed_model()

    # 4. Upsert
    try:
        client = get_qdrant_client()
        ensure_collection(client)
        console.log(f"[cyan]Upserting {len(chunks)} vectors to Qdrant…[/cyan]")
        upsert_chunks(client, chunks)
        console.log(f"[green]✓ Upserted successfully[/green]")
    except Exception as exc:
        raise RuntimeError(
            f"Qdrant ingestion failed after retries: {exc}\n"
            f"  Check that Qdrant is reachable at {config.QDRANT_URL}"
        ) from exc

    stats = {
        "file":       pdf_path.name,
        "chunks":     len(chunks),
        "embed_secs": round(elapsed, 2),
        "parser":     parser_used,
        "chunk_list": [
            {
                "chunk_id": c["metadata"]["chunk_id"],
                "text": c["text"],
                "h1": c["metadata"].get("h1", ""),
                "h2": c["metadata"].get("h2", ""),
            }
            for c in chunks
        ],
    }
    # Print clean summary without printing all chunks to terminal
    print_stats = stats.copy()
    print_stats["chunk_list"] = f"[{len(chunks)} chunks]"
    console.print(print_stats)
    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────
@app.command()
def main(
    path: Path = typer.Argument(..., help="PDF file or directory of PDFs"),
    parser: str = typer.Option(
        "auto",
        help="Parser: auto | pymupdf (fast, Python 3.14) | marker (high-quality, needs torch)",
    ),
):
    """
    Ingest one PDF or an entire directory of PDFs into Qdrant.

    Examples:\n
      python ingest.py paper.pdf\n
      python ingest.py ./pdfs/\n
      python ingest.py paper.pdf --parser marker\n
    """
    if path.is_dir():
        pdfs = sorted(path.glob("*.pdf"))
        if not pdfs:
            console.print("[red]No PDFs found in directory.[/red]")
            raise typer.Exit(1)

        all_stats = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Ingesting PDFs", total=len(pdfs))
            for pdf in pdfs:
                stats = ingest_pdf(pdf, parser=parser)  # type: ignore[arg-type]
                all_stats.append(stats)
                progress.advance(task)

        total_chunks = sum(s["chunks"] for s in all_stats)
        console.rule("[bold green]Done")
        console.print(f"Ingested [bold]{len(pdfs)}[/bold] PDFs → [bold]{total_chunks}[/bold] total chunks")

    elif path.is_file() and path.suffix.lower() == ".pdf":
        ingest_pdf(path, parser=parser)  # type: ignore[arg-type]

    else:
        console.print(f"[red]Error:[/red] {path} is not a PDF or directory.")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
