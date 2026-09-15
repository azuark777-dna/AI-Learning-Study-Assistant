"""Configuration loading for the AI Learning & Study Assistant."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
MATERIALS_DIR = DATA_DIR / "materials"
MEMORY_DIR = DATA_DIR / "memory"

MATERIALS_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

# RAG settings
CHUNK_SIZE_WORDS = 220
CHUNK_OVERLAP_WORDS = 40
TOP_K_RETRIEVAL = 4

# Progress settings
WEAK_TOPIC_SCORE_THRESHOLD = 0.7  # quiz score below this flags a topic as weak
