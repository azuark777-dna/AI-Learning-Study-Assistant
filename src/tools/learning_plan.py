"""Tool: create_learning_plan

Builds an ordered study plan from a set of topics, an exam date, and the
student's available study hours per week, and persists it to memory.
"""
from datetime import datetime, date
from typing import List, Dict

from src.memory import StudentMemory


def create_learning_plan(
    student_id: str,
    topics: List[str],
    exam_date: str,
    hours_per_week: float,
) -> Dict:
    """Create and persist a study plan.

    Args:
        student_id: unique student identifier.
        topics: ordered list of topic names extracted from course materials.
        exam_date: ISO date string (YYYY-MM-DD) for the target exam.
        hours_per_week: how many hours per week the student can study.

    Returns:
        dict describing the generated plan.
    """
    memory = StudentMemory.load(student_id)

    today = date.today()
    try:
        target = datetime.strptime(exam_date, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError("exam_date must be in YYYY-MM-DD format")

    days_remaining = max((target - today).days, 1)
    weeks_remaining = max(days_remaining / 7, 1)
    total_hours = weeks_remaining * hours_per_week
    hours_per_topic = round(total_hours / max(len(topics), 1), 1)

    memory.set_plan(topics)
    memory.update_profile(exam_date=exam_date, hours_per_week=hours_per_week)

    plan = {
        "topics": topics,
        "exam_date": exam_date,
        "days_remaining": days_remaining,
        "hours_per_topic": hours_per_topic,
        "total_planned_hours": round(total_hours, 1),
    }
    return plan
