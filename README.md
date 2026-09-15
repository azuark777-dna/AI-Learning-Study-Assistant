# AI Learning & Study Assistant

An Agentic AI application that helps a student study: it builds personalized
learning plans, answers questions grounded in the student's own course
materials (RAG), generates practice quizzes, and remembers progress across
sessions — built with **LangChain**, **LangGraph**, **RAG**, and **MCP**, running
entirely on a local **Ollama** model.

Built for Use Case #5, AI Agent Bootcamp. See `docs/PRD.docx` for the full
product requirements and `docs/PROGRAM_ALIGNMENT.md` for how this project
maps to the program's stated objectives and outcomes.

## Architecture

```
                     ┌─────────────────────────────────────────┐
                     │            LangGraph workflow            │
                     │                                           │
   user message ───▶ │   ┌─────────┐  tool call   ┌─────────┐   │
                      │   │  agent  │ ───────────▶ │  tools  │   │
                      │   │ (LLM)   │ ◀─────────── │ (exec)  │   │
                      │   └─────────┘  tool result └─────────┘   │
                      │        │ no tool call needed              │
                      │        ▼                                  │
   final answer  ◀────┼──── END                                   │
                     └─────────────────────────────────────────┘
                            │                    │
                     LangChain tools        Ollama (local LLM)
                            │
              ┌─────────────┼──────────────┬───────────────┐
              ▼             ▼              ▼               ▼
       search_materials  create_       generate_       update_
       (RAG retrieval)   learning_plan quiz            progress
              │                             │               │
              ▼                             ▼               ▼
     data/materials/*.chunks.json   Ollama (quiz gen)  data/memory/*.json
     (TF-IDF retrieval)                                 (StudentMemory)
```

The same four tools are also exposed over **MCP** (`src/mcp_server/server.py`)
so any MCP-compatible client can call them directly — that's the "Agentic
RAG extended with MCP integration" piece.

## Key Agent Capabilities

| Capability | Where it lives |
|---|---|
| **LLM** | `src/llm.py` — `ChatOllama` (local model via Ollama) |
| **LangChain Tools** | `src/langchain_tools/tools.py` |
| **LangGraph Workflow** | `src/graph/state.py` (state), `src/graph/nodes.py` (nodes + error handling), `src/graph/workflow.py` (edges + memory checkpointer) |
| **RAG** | `src/rag/ingest.py` (chunking), `src/rag/retriever.py` (TF-IDF retrieval) |
| **MCP** | `src/mcp_server/server.py` |
| **Long-term Memory** | `src/memory.py`, persisted under `data/memory/` |
| **Business logic tools** | `src/tools/learning_plan.py`, `src/tools/quiz.py`, `src/tools/progress.py` |

## Project Structure

```
ai-learning-study-assistant/
├── README.md
├── requirements.txt
├── .env.example
├── docs/
│   ├── PRD.docx                     Full product requirements document
│   └── PROGRAM_ALIGNMENT.md         Maps program objectives/outcomes to this project
├── src/
│   ├── config.py                    Env/config loading (Ollama host + model)
│   ├── llm.py                       ChatOllama LLM provider
│   ├── agent.py                     Thin façade over the compiled LangGraph app
│   ├── memory.py                    Student profile, plan, quiz history persistence
│   ├── main.py                      CLI chat entry point
│   ├── graph/
│   │   ├── state.py                 AgentState (messages, student_id)
│   │   ├── nodes.py                 agent_node (LLM call + error handling)
│   │   └── workflow.py              StateGraph: nodes, conditional edges, checkpointer
│   ├── langchain_tools/
│   │   └── tools.py                 LangChain @tool wrappers, bound per student
│   ├── rag/
│   │   ├── ingest.py                Parse, chunk, and store course materials
│   │   └── retriever.py             TF-IDF retrieval over chunks
│   ├── tools/                       Business logic (framework-agnostic)
│   │   ├── learning_plan.py
│   │   ├── quiz.py                  Calls Ollama directly for quiz JSON generation
│   │   └── progress.py
│   └── mcp_server/
│       └── server.py                FastMCP server exposing the same 4 tools
├── data/
│   ├── materials/                   Uploaded course materials + chunk store (per student)
│   │   └── sample_course_notes.txt
│   └── memory/                      Persisted per-student memory (JSON)
└── tests/
    ├── test_rag.py                  Ingestion/retrieval tests
    └── test_graph.py                LangGraph workflow structure tests
```

## Step-by-Step: VS Code + Ollama Setup

### 1. Install Ollama
Download from **ollama.com/download** and install for your OS.

### 2. Pull a tool-calling capable model
```bash
ollama pull llama3.1
```

### 3. Start the Ollama server (if not already running)
```bash
ollama serve
```

### 4. Open the project in VS Code
`File → Open Folder` → select `ai-learning-study-assistant`.
Install the **Python** extension (by Microsoft) if you haven't already.

### 5. Open a terminal in VS Code and create a virtual environment
```bash
python -m venv venv
```
Activate it:
- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

### 6. Install dependencies
```bash
pip install -r requirements.txt
```

### 7. Configure environment
```bash
cp .env.example .env
```
Defaults already point at `http://localhost:11434` / `llama3.1` — edit only
if you used a different Ollama host or model name.

### 8. Run the tests (optional but recommended)
```bash
python -m pytest tests/ -v
```
These pass without Ollama running (they test ingestion, retrieval, and the
LangGraph structure, not live model calls).

### 9. Run the CLI
```bash
python -m src.main
```

### 10. Try it
```
Student ID: demo_student
> /upload data/materials/sample_course_notes.txt
> What's the difference between supervised and unsupervised learning?
> /plan 2026-10-01 6
> /quiz neural networks
> /progress
```

### 11. (Optional) Run the MCP server
To expose the same tools to an MCP client (e.g. Claude Desktop):
```bash
python -m src.mcp_server.server
```

See the docstring at the top of `src/mcp_server/server.py` for a sample
Claude Desktop config entry.

## CLI Commands

- `/upload <path-to-file>` — ingest a `.txt`, `.pdf`, or `.docx` course material.
- `/plan <exam-date YYYY-MM-DD> <hours-per-week>` — generate a learning plan.
- `/quiz <topic>` — generate a short quiz on a topic from ingested materials.
- `/progress` — show current plan status and quiz history.
- Anything else — a free-form question, answered via RAG + the LangGraph agent.

## Notes on this implementation

- **Retrieval** uses TF-IDF + cosine similarity (`scikit-learn`) — no external
  vector database or embedding API required. Swap `src/rag/` for real
  embeddings + a vector store (Chroma, pgvector) later without touching the
  tool interfaces.
- **Memory** has two layers: LangGraph's `MemorySaver` checkpointer gives
  short-term, per-conversation memory (thread_id = student_id); `src/memory.py`
  gives long-term, cross-session memory (profile, plan, quiz history) as
  simple per-student JSON files.
- **Generation** runs entirely on a local Ollama model — no API key, no
  internet required once the model is pulled.
- **MCP integration** means the RAG index, learning plan, quiz, and progress
  tools aren't locked inside this CLI — any MCP-aware client can call them.

## Extending

- Swap the TF-IDF retriever for embeddings + a vector DB in `src/rag/`.
- Add more LangGraph nodes (e.g. a dedicated "grade_quiz" node, or a
  human-in-the-loop review step before finalizing a plan).
- Add a web UI on top of `src/agent.py` (`Agent.handle()` is UI-agnostic).
- Add spaced-repetition scheduling in `src/tools/learning_plan.py`.

## Preparing for Submission

See `docs/PROGRAM_ALIGNMENT.md` for a step-by-step guide to pushing this to
GitHub and what to include in your final submission (repo link, PRD,
program-alignment doc, and a short demo).
