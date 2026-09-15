"""Tests for the LangGraph workflow structure (no live Ollama server
required — this only checks the graph compiles with the expected shape)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph.workflow import build_graph
from src.langchain_tools.tools import build_tools


def test_graph_compiles_with_expected_nodes():
    app = build_graph("pytest_student", "No prior history.")
    nodes = set(app.get_graph().nodes.keys())
    assert {"agent", "tools", "__start__", "__end__"}.issubset(nodes)


def test_build_tools_returns_four_tools():
    tools = build_tools("pytest_student")
    names = {t.name for t in tools}
    assert names == {
        "search_materials",
        "create_learning_plan",
        "generate_quiz",
        "update_progress",
    }
