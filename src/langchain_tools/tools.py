"""LangChain tool wrappers.

These wrap the plain-Python business logic in src/tools/ (and the RAG
retriever) as LangChain Tool objects so a LangChain/LangGraph agent can call
them via standard tool-calling. Each call is bound to a specific student_id
via a factory function, since a single student's session should never see
another student's materials or memory.
"""
from typing import List, Optional

from langchain_core.tools import tool

from src.rag.retriever import retrieve, format_context
from src.tools.learning_plan import create_learning_plan as _create_learning_plan
from src.tools.quiz import generate_quiz as _generate_quiz
from src.tools.progress import update_progress as _update_progress


def build_tools(student_id: str) -> List:
    """Return the four agent tools, each bound to `student_id`."""

    @tool
    def search_materials(query: str) -> str:
        """Retrieve relevant passages from the student's uploaded course
        materials for a query. Always use this before answering a factual
        question about course content."""
        results = retrieve(student_id, query)
        if not results:
            return "No relevant material found. The student may need to /upload materials first."
        return format_context(results)

    @tool
    def create_learning_plan(
        topics: List[str], exam_date: str, hours_per_week: float
    ) -> str:
        """Create and save a personalized study plan.

        Args:
            topics: ordered list of topic names to study.
            exam_date: target exam date, format YYYY-MM-DD.
            hours_per_week: hours the student can study per week.
        """
        plan = _create_learning_plan(student_id, topics, exam_date, hours_per_week)
        return (
            f"Created a plan covering {len(plan['topics'])} topics over "
            f"{plan['days_remaining']} days ({plan['hours_per_topic']} hours/topic, "
            f"{plan['total_planned_hours']} total hours)."
        )

    @tool
    def generate_quiz(topic: str, num_questions: int = 5) -> str:
        """Generate a practice quiz on a topic, grounded in the student's
        uploaded materials.

        Args:
            topic: the topic to quiz on.
            num_questions: how many questions to generate (default 5).
        """
        quiz = _generate_quiz(student_id, topic, num_questions)
        if quiz.get("error"):
            return f"Could not generate quiz: {quiz['error']}"
        lines = [f"Quiz: {quiz['topic']}"]
        for i, q in enumerate(quiz["questions"], start=1):
            lines.append(f"\nQ{i}. {q['question']}")
            if q["type"] == "multiple_choice":
                lines.extend(f"   - {opt}" for opt in q["options"])
            lines.append(f"   Answer: {q['answer']}")
            if q.get("explanation"):
                lines.append(f"   Why: {q['explanation']}")
        return "\n".join(lines)

    @tool
    def update_progress(
        topic: Optional[str] = None,
        status: Optional[str] = None,
        quiz_score: Optional[float] = None,
        quiz_num_questions: Optional[int] = None,
    ) -> str:
        """Update a topic's status (upcoming/in_progress/completed) and/or
        record a quiz result in the student's memory.

        Args:
            topic: topic name to update.
            status: one of "upcoming", "in_progress", "completed".
            quiz_score: fraction correct (0.0-1.0) from a completed quiz.
            quiz_num_questions: number of questions in that quiz.
        """
        result = _update_progress(
            student_id,
            topic=topic,
            status=status,
            quiz_score=quiz_score,
            quiz_num_questions=quiz_num_questions,
        )
        weak = result.get("weak_topics") or []
        weak_note = f" Weak topics to revisit: {', '.join(weak)}." if weak else ""
        return f"Progress updated.{weak_note}"

    return [search_materials, create_learning_plan, generate_quiz, update_progress]
