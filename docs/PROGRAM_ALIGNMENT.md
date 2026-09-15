# Program Alignment

This document maps the AI Agent Bootcamp's program highlights, objective, and
outcomes to concrete parts of this project — for your final submission and
for reviewers checking the project against the syllabus.

## 1. Program Highlights → Where They Show Up Here

| Highlight | In this project |
|---|---|
| Hands-on exposure to LangChain, LangGraph, RAG, MCP | `src/langchain_tools/` (LangChain tools), `src/graph/` (LangGraph workflow), `src/rag/` (RAG pipeline), `src/mcp_server/` (MCP server) |
| Complete hands-on sessions — every concept reinforced with a guided lab | `tests/` (runnable checks), sample data in `data/materials/`, step-by-step README |
| Real, working Agentic AI project built Day 1 → Day 5 | See the 5-day build plan in `docs/PRD.docx`, Section 11 |
| Final project submitted with documentation + GitHub link | This repo structure, README, and PRD are submission-ready; see "Preparing Your GitHub Submission" below |
| LMS tracking (submissions, quiz, content, feedback) | This project's own quiz + progress tracking (`src/tools/quiz.py`, `src/tools/progress.py`, `src/memory.py`) mirrors the same pattern at the application level |

## 2. Program Objective → Project Mapping

> To equip students with practical, hands-on capability to design and build
> an Agentic AI application — from concept to a fully working, demoable
> project — using industry-standard tools (LLMs, LangChain, LangGraph, RAG,
> MCP), while preparing them for IBM's global certification exam.

This project is exactly that application: a locally-run (Ollama) LLM,
orchestrated with LangChain tools inside a LangGraph workflow, grounded by a
RAG pipeline over real course materials, and exposed over MCP so any
MCP-compatible client can use the same tools.

## 3. Program Outcomes → Where Each One Is Demonstrated

| Outcome | Demonstrated in |
|---|---|
| Explain agentic AI fundamentals and core agent architecture (model, tools, memory) | `README.md` "How It Works"; `src/llm.py` (model), `src/langchain_tools/tools.py` (tools), `src/memory.py` (memory) |
| Build and test an AI agent with tool-calling using LangChain | `src/langchain_tools/tools.py`, `src/llm.py` (`bind_tools`), `tests/` |
| Design a multi-step agent workflow using LangGraph (nodes, edges, state, memory, error handling) | `src/graph/state.py` (state), `src/graph/nodes.py` (nodes + error handling), `src/graph/workflow.py` (edges + conditional routing + `MemorySaver` checkpointer) |
| Build a RAG pipeline and extend it into Agentic RAG with MCP integration | `src/rag/ingest.py` + `src/rag/retriever.py` (RAG); the agent deciding *when* to retrieve via the `search_materials` tool (Agentic RAG); `src/mcp_server/server.py` (MCP integration) |
| Independently build, test, document, and present a complete Agentic AI project with a GitHub repository | This whole repo: working code, `tests/`, `README.md`, `docs/PRD.docx`, this file |

## Preparing Your GitHub Submission

1. Initialize git and push:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: AI Learning & Study Assistant"
   git branch -M mainpython
   git remote add origin https://github.com/<your-username>/ai-learning-study-assistant.git
   git push -u origin main
   ```
2. Make sure `.env` is **not** committed (it's already in `.gitignore`) —
   only `.env.example` should be tracked.
3. In your submission, include:
   - The GitHub repository link
   - `docs/PRD.docx` (product requirements)
   - This file (`docs/PROGRAM_ALIGNMENT.md`)
   - A short demo (screen recording or screenshots) of `/upload`, a
     grounded Q&A answer, `/plan`, and `/quiz` in action
