"""LLM provider: a local Ollama model wrapped via LangChain's ChatOllama.

This is the single place the rest of the app gets its LLM from — swap the
model or provider here without touching graph/tool code.
"""
from langchain_ollama import ChatOllama

from src.config import OLLAMA_HOST, OLLAMA_MODEL


def get_llm(temperature: float = 0.2) -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_HOST,
        temperature=temperature,
    )
