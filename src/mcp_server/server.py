"""MCP server exposing the study assistant's core capabilities
(search_materials / create_learning_plan / generate_quiz / update_progress)
as standard MCP tools.

This is the "extend RAG into Agentic RAG with MCP integration" piece: any
MCP-compatible client (Claude Desktop, another agent, etc.) can attach to
this server and call these tools directly, over the same RAG index and
per-student memory the CLI agent uses — the tools are the shared source of
truth, not duplicated logic.

Run standalone:
    python -m src.mcp_server.server

Then point an MCP client at it (stdio transport). Example Claude Desktop
config entry (claude_desktop_config.json):
    {
      "mcpServers": {
        "study-assistant": {
          "command": "python",
          "args": ["-m", "src.mcp_server.server"],
          "cwd": "/absolute/path/to/ai-learning-study-assistant"
        }
      }
    }
"""
from typing import List, Optional

from mcp.server.fastmcp import FastMCP

from src.rag.retriever import retrieve, format_context
from src.tools.learning_plan import create_learning_plan as _create_learning_plan
from src.tools.quiz import generate_quiz as _generate_quiz
from src.tools.progress import update_progress as _update_progress

mcp = FastMCP("study-assistant")


@mcp.tool()
def search_materials(student_id: str, query: str) -> str:
    """Retrieve relevant passages from a student's uploaded course materials."""
    results = retrieve(student_id, query)
    if not results:
        return "No relevant material found for this student/query."
    return format_context(results)


@mcp.tool()
def create_learning_plan(
    student_id: str, topics: List[str], exam_date: str, hours_per_week: float
) -> dict:
    """Create and persist a personalized study plan for a student.

    exam_date must be in YYYY-MM-DD format.
    """
    return _create_learning_plan(student_id, topics, exam_date, hours_per_week)


@mcp.tool()
def generate_quiz(student_id: str, topic: str, num_questions: int = 5) -> dict:
    """Generate a practice quiz on a topic, grounded in the student's
    uploaded materials."""
    return _generate_quiz(student_id, topic, num_questions)


@mcp.tool()
def update_progress(
    student_id: str,
    topic: Optional[str] = None,
    status: Optional[str] = None,
    quiz_score: Optional[float] = None,
    quiz_num_questions: Optional[int] = None,
) -> dict:
    """Update a topic's status and/or record a quiz result for a student."""
    return _update_progress(
        student_id,
        topic=topic,
        status=status,
        quiz_score=quiz_score,
        quiz_num_questions=quiz_num_questions,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
