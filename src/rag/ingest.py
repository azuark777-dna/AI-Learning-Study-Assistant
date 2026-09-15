"""Ingestion: parse course materials into text and split into overlapping chunks.

Chunks are persisted per-student as a JSON "chunk store" so the retriever can
load them without re-parsing the source files on every query. Swap this for a
real embedding + vector-DB pipeline later without changing the public
functions (`ingest_file`, `load_chunks`).
"""
import json
from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader
from docx import Document as DocxDocument

from src.config import MATERIALS_DIR, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS


def _extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt" or suffix == ".md":
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        doc = DocxDocument(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError(f"Unsupported file type: {suffix}")


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def _chunk_store_path(student_id: str) -> Path:
    return MATERIALS_DIR / f"{student_id}.chunks.json"


def load_chunks(student_id: str) -> List[Dict]:
    """Load all previously ingested chunks for a student."""
    path = _chunk_store_path(student_id)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _save_chunks(student_id: str, chunks: List[Dict]) -> None:
    _chunk_store_path(student_id).write_text(
        json.dumps(chunks, indent=2), encoding="utf-8"
    )


def ingest_file(student_id: str, file_path: str) -> int:
    """Parse a course material file, chunk it, and append to the student's
    chunk store. Returns the number of chunks added.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {file_path}")

    text = _extract_text(path)
    raw_chunks = _chunk_text(text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)

    existing = load_chunks(student_id)
    next_id = len(existing)
    new_chunks = [
        {
            "id": next_id + i,
            "source": path.name,
            "chunk_index": i,
            "text": chunk,
        }
        for i, chunk in enumerate(raw_chunks)
    ]

    _save_chunks(student_id, existing + new_chunks)
    return len(new_chunks)
