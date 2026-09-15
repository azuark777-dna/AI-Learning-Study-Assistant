"""Builds the LangGraph StateGraph for the study-assistant agent.

Graph shape (this is the "multi-step agent workflow" from the program
outcomes: nodes, edges, state, memory, error handling):

    START -> agent -> [conditional] -> tools -> agent -> ... -> END
                    -> END (when the model returns a plain answer, no tool calls)

- State: AgentState (messages + student_id), threaded through every node.
- Nodes: "agent" (LLM decision node) and "tools" (LangGraph's prebuilt
  ToolNode, which executes whichever tool calls the agent requested).
- Edges: conditional edge out of "agent" (tools_condition) routes to "tools"
  if the model asked for a tool call, otherwise straight to END. A fixed
  edge routes "tools" back to "agent" so the model can use the tool result.
- Memory: a MemorySaver checkpointer gives short-term, per-thread
  conversational memory (so multi-turn chat keeps context); long-term memory
  (student profile/plan/quiz history) lives in src/memory.py and is loaded
  into the system prompt when the graph is built.
- Error handling: agent_node catches exceptions (see graph/nodes.py); the
  ToolNode surfaces tool exceptions as ToolMessages by default so the model
  can react to a failed tool call rather than crashing.
"""
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import AgentState
from src.graph.nodes import make_agent_node
from src.langchain_tools.tools import build_tools


def build_graph(student_id: str, memory_summary: str):
    """Compile a runnable LangGraph app for one student's session."""
    tools = build_tools(student_id)
    agent_node = make_agent_node(tools, memory_summary)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,  # inspects the last message for tool_calls
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")

    checkpointer = MemorySaver()  # short-term, per-thread conversation memory
    return graph.compile(checkpointer=checkpointer)
