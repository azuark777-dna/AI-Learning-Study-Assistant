"""Long-term memory for a student: profile, learning plan state, and quiz
history, persisted as a per-student JSON file.

This is the single seam the rest of the app talks to for persistence — swap
the load/save internals for a real database later without touching callers.
"""
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional

from src.config import MEMORY_DIR


def _memory_path(student_id: str) -> Path:
    return MEMORY_DIR / f"{student_id}.json"


@dataclass
class StudentMemory:
    student_id: str
    profile: Dict = field(default_factory=dict)          # name, course, exam_date, hours_per_week
    plan: List[Dict] = field(default_factory=list)        # [{topic, status, order}]
    quiz_history: List[Dict] = field(default_factory=list)  # [{topic, score, date, num_questions}]
    session_log: List[str] = field(default_factory=list)  # short notes across sessions

    # ---------- persistence ----------
    @classmethod
    def load(cls, student_id: str) -> "StudentMemory":
        path = _memory_path(student_id)
        if not path.exists():
            return cls(student_id=student_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)

    def save(self) -> None:
        _memory_path(self.student_id).write_text(
            json.dumps(asdict(self), indent=2), encoding="utf-8"
        )

    # ---------- profile ----------
    def update_profile(self, **kwargs) -> None:
        self.profile.update({k: v for k, v in kwargs.items() if v is not None})
        self.save()

    # ---------- plan ----------
    def set_plan(self, topics: List[str]) -> None:
        self.plan = [
            {"topic": t, "status": "upcoming", "order": i}
            for i, t in enumerate(topics)
        ]
        self.save()

    def update_topic_status(self, topic: str, status: str) -> bool:
        for item in self.plan:
            if item["topic"].lower() == topic.lower():
                item["status"] = status
                self.save()
                return True
        return False

    # ---------- quiz history ----------
    def record_quiz_result(self, topic: str, score: float, num_questions: int, date: str) -> None:
        self.quiz_history.append(
            {
                "topic": topic,
                "score": score,
                "num_questions": num_questions,
                "date": date,
            }
        )
        self.save()

    def weak_topics(self, threshold: float) -> List[str]:
        latest_scores: Dict[str, float] = {}
        for entry in self.quiz_history:
            latest_scores[entry["topic"]] = entry["score"]
        return [t for t, s in latest_scores.items() if s < threshold]

    # ---------- session summary ----------
    def summary(self) -> str:
        if not self.plan and not self.quiz_history:
            return "No prior study history yet — this looks like a first session."

        lines = []
        if self.profile:
            course = self.profile.get("course", "your course")
            exam_date = self.profile.get("exam_date")
            lines.append(
                f"Welcome back! Course: {course}"
                + (f", exam date: {exam_date}" if exam_date else "")
            )
        if self.plan:
            done = [p["topic"] for p in self.plan if p["status"] == "completed"]
            upcoming = [p["topic"] for p in self.plan if p["status"] != "completed"]
            if done:
                lines.append(f"Completed so far: {', '.join(done)}")
            if upcoming:
                lines.append(f"Still to cover: {', '.join(upcoming)}")
        if self.quiz_history:
            last = self.quiz_history[-1]
            lines.append(
                f"Last quiz: {last['topic']} — {last['score']*100:.0f}%"
            )
        return "\n".join(lines)
