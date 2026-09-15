"""LangGraph nodes for the study-assistant agent workflow.

Two nodes:
  - agent_node: calls the LLM (with tools bound) to decide what to say or
    which tool(s) to call next.
  - error-handling wrapper: catches exceptions from either node so a local
    model hiccup (e.g. malformed tool call, Ollama connection drop) surfaces
    as a graceful message instead of crashing the whole graph.

The tool-execution node itself is LangGraph's prebuilt `ToolNode` (wired up
in workflow.py) — it already handles running each requested tool call and
appending the results as ToolMessages to state.
"""
from langchain_core.messages import AIMessage, SystemMessage

from src.graph.state import AgentState
from src.llm import get_llm

SYSTEM_PROMPT = """You are an AI Learning & Study Assistant. You help a student \
study by answering questions grounded in their uploaded course materials, \
building personalized learning plans, generating practice quizzes, and \
tracking their progress across sessions.

Rules:
- When answering a factual question about course content, ALWAYS call \
search_materials first and ground your answer in the returned chunks. If \
nothing relevant is found, say so plainly rather than guessing.
- Use create_learning_plan when the student wants a study schedule and has \
given (or you can infer) topics, an exam date, and study hours per week.
- Use generate_quiz when the student wants to practice or test themselves \
on a topic.
- Use update_progress after a topic is completed or a quiz is scored, so \
memory stays current.
- Be concise, encouraging, and concrete.
"""


def make_agent_node(tools, memory_summary: str):
    """Build the agent node, closing over the tools list and a memory
    summary string so the LLM has continuity from prior sessions."""
    llm_with_tools = get_llm().bind_tools(tools)
    system_message = SystemMessage(
        content=SYSTEM_PROMPT + "\n\nStudent memory summary:\n" + memory_summary
    )

    def agent_node(state: AgentState) -> dict:
        try:
            messages = [system_message] + state["messages"]
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}
        except Exception as e:
            # Error handling: surface a clear message instead of raising and
            # killing the whole graph run (e.g. Ollama unreachable).
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "I hit an error talking to the local model "
                            f"({e}). Make sure Ollama is running "
                            "(`ollama serve`) and the model is pulled."
                        )
                    )
                ]
            }

    return agent_node
