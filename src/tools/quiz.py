"""Tool: generate_quiz

Generates a practice quiz on a topic, grounded in the student's ingested
course materials via RAG, using a local Ollama model for question generation.
"""
import json
import re
from typing import Dict, List

import ollama

from src.config import OLLAMA_HOST, OLLAMA_MODEL
from src.rag.retriever import retrieve, format_context

_client = ollama.Client(host=OLLAMA_HOST)

QUIZ_PROMPT = """You are a study-quiz generator. Using ONLY the course material \
context below, write {num_questions} quiz questions on the topic "{topic}". \
Mix multiple-choice and true/false questions. Base every question strictly on \
the provided context — do not invent facts that aren't supported by it.

Respond ONLY with valid JSON (no markdown fences, no preamble, no commentary) \
in this shape:
{{
  "topic": "{topic}",
  "questions": [
    {{
      "type": "multiple_choice",
      "question": "...",
      "options": ["A", "B", "C", "D"],
      "answer": "B",
      "explanation": "..."
    }},
    {{
      "type": "true_false",
      "question": "...",
      "answer": "true",
      "explanation": "..."
    }}
  ]
}}

Course material context:
{context}
"""


def _extract_json(text: str) -> str:
    """Local models sometimes wrap JSON in markdown fences or add stray
    commentary — strip that before parsing."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    # Fall back to the first {...} block in the text.
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    return brace.group(0) if brace else text


def generate_quiz(student_id: str, topic: str, num_questions: int = 5) -> Dict:
    """Generate a quiz grounded in the student's ingested materials.

    Falls back to a clear error message (rather than fabricating questions)
    if no relevant material has been ingested yet, or Ollama is unreachable.
    """
    results = retrieve(student_id, topic, top_k=6)
    if not results:
        return {
            "topic": topic,
            "questions": [],
            "error": (
                "No relevant course material found for this topic. "
                "Upload materials with /upload first."
            ),
        }

    context = format_context(results)
    prompt = QUIZ_PROMPT.format(
        num_questions=num_questions, topic=topic, context=context
    )

    try:
        response = _client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
    except Exception as e:
        return {
            "topic": topic,
            "questions": [],
            "error": f"Could not reach Ollama at {OLLAMA_HOST}: {e}",
        }

    text = response["message"]["content"].strip()

    try:
        quiz = json.loads(_extract_json(text))
    except json.JSONDecodeError:
        return {"topic": topic, "questions": [], "error": "Failed to parse quiz output."}

    quiz["sources"] = sorted({r["source"] for r in results})
    return quiz


def grade_quiz(quiz: Dict, student_answers: List[str]) -> Dict:
    """Grade a quiz given the student's submitted answers (same order as
    quiz['questions']). Returns score fraction and per-question feedback.
    """
    questions = quiz.get("questions", [])
    correct = 0
    feedback = []
    for q, given in zip(questions, student_answers):
        is_correct = str(given).strip().lower() == str(q["answer"]).strip().lower()
        correct += int(is_correct)
        feedback.append(
            {
                "question": q["question"],
                "given": given,
                "correct_answer": q["answer"],
                "is_correct": is_correct,
                "explanation": q.get("explanation", ""),
            }
        )
    score = correct / len(questions) if questions else 0.0
    return {"score": score, "correct": correct, "total": len(questions), "feedback": feedback}
