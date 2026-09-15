"""LangGraph state schema for the study-assistant agent workflow.

`messages` is the running conversation (LangChain message objects); the
`add_messages` reducer appends new messages rather than overwriting the list,
which is what gives the graph its per-turn conversational memory.
"""
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    student_id: str
