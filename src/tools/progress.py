"""Tool: update_progress

Updates plan status and records quiz results in a student's persisted memory,
and surfaces weak topics for review.
"""
from datetime import date
from typing import Dict, Optional

from src.memory import StudentMemory
from src.config import WEAK_TOPIC_SCORE_THRESHOLD


def update_progress(
    student_id: str,
    topic: Optional[str] = None,
    status: Optional[str] = None,
    quiz_score: Optional[float] = None,
    quiz_num_questions: Optional[int] = None,
) -> Dict:
    """Update a topic's plan status and/or record a quiz result.

    Args:
        student_id: unique student identifier.
        topic: topic name to update (required if status or quiz_score given).
        status: one of "upcoming", "in_progress", "completed".
        quiz_score: fraction correct (0.0-1.0) from a completed quiz.
        quiz_num_questions: number of questions in that quiz.
    """
    memory = StudentMemory.load(student_id)

    updated_status = False
    if topic and status:
        updated_status = memory.update_topic_status(topic, status)

    if topic and quiz_score is not None:
        memory.record_quiz_result(
            topic=topic,
            score=quiz_score,
            num_questions=quiz_num_questions or 0,
            date=date.today().isoformat(),
        )

    return {
        "updated_status": updated_status,
        "weak_topics": memory.weak_topics(WEAK_TOPIC_SCORE_THRESHOLD),
        "plan": memory.plan,
        "quiz_history": memory.quiz_history,
    }
