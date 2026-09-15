"""Retrieval over a student's ingested course materials.

Uses TF-IDF + cosine similarity so the project runs with no external
embedding API or vector database. The interface (`retrieve`) is the seam to
swap in real embeddings (e.g. OpenAI/Voyage embeddings + Chroma/pgvector)
later.
"""
from typing import List, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.rag.ingest import load_chunks
from src.config import TOP_K_RETRIEVAL


def retrieve(student_id: str, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict]:
    """Return the top_k most relevant chunks for `query` from the student's
    ingested materials, each annotated with a similarity score and source.
    """
    chunks = load_chunks(student_id)
    if not chunks:
        return []

    corpus = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus + [query])

    query_vec = matrix[-1]
    doc_vecs = matrix[:-1]
    scores = cosine_similarity(query_vec, doc_vecs).flatten()

    ranked_idx = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in ranked_idx:
        if scores[idx] <= 0:
            continue
        chunk = chunks[idx]
        results.append(
            {
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "score": float(scores[idx]),
            }
        )
    return results


def format_context(results: List[Dict]) -> str:
    """Render retrieved chunks into a citation-friendly context block for the
    LLM prompt.
    """
    if not results:
        return "(no relevant material found)"
    lines = []
    for r in results:
        lines.append(
            f"[Source: {r['source']}, chunk {r['chunk_index']}]\n{r['text']}"
        )
    return "\n\n".join(lines)
