"""Basic tests for the ingestion + retrieval pipeline (no API key required)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.ingest import ingest_file, load_chunks, _chunk_store_path
from src.rag.retriever import retrieve

TEST_STUDENT = "pytest_student"
SAMPLE_FILE = str(
    Path(__file__).resolve().parents[1] / "data/materials/sample_course_notes.txt"
)


def setup_function(_):
    """Ensure each test starts from a clean chunk store for TEST_STUDENT."""
    path = _chunk_store_path(TEST_STUDENT)
    if path.exists():
        path.unlink()


def test_ingest_creates_chunks():
    n = ingest_file(TEST_STUDENT, SAMPLE_FILE)
    assert n > 0
    chunks = load_chunks(TEST_STUDENT)
    assert len(chunks) == n
    assert all("text" in c and "source" in c for c in chunks)


def test_retrieve_finds_relevant_chunk():
    ingest_file(TEST_STUDENT, SAMPLE_FILE)
    results = retrieve(TEST_STUDENT, "difference between supervised and unsupervised learning")
    assert len(results) > 0
    # The top result should mention supervised/unsupervised learning
    top_text = results[0]["text"].lower()
    assert "supervised" in top_text


def test_retrieve_empty_for_unseeded_student():
    results = retrieve("nobody_has_ingested_for_this_id", "anything")
    assert results == []
