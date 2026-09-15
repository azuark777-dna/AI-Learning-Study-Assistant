"""Agent façade: wraps the compiled LangGraph workflow behind the same
simple interface the CLI (src/main.py) already uses — `Agent(student_id)`
then `.handle(message)`.

This is the "Agent + RAG + Memory + Tools" piece from the PRD, now
implemented with LangChain (tools + LLM) and LangGraph (multi-step
workflow: nodes, edges, state, memory, error handling).
"""
from langchain_core.messages import HumanMessage

from src.memory import StudentMemory
from src.graph.workflow import build_graph


class Agent:
    def __init__(self, student_id: str):
        self.student_id = student_id
        self.memory = StudentMemory.load(student_id)
        self.graph = build_graph(student_id, self.memory.summary())
        # LangGraph threads a checkpoint by thread_id — one thread per student
        # gives each student their own short-term conversational memory.
        self.config = {"configurable": {"thread_id": student_id}}

    def handle(self, user_message: str) -> str:
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=user_message)], "student_id": self.student_id},
            config=self.config,
        )
        final_message = result["messages"][-1]
        return final_message.content
