# Atlas · AI Agent Studio

A research agent with a live "show your work" UI.

**Stack:** FastAPI · Uvicorn · Jinja2 · LangGraph · LangChain · OpenAI · Pinecone · Tavily · pypdf · python-docx

## What the agent does

```
router ─┬─ direct ───────────────────────────────┐
        ├─ docs   → retrieve_docs ─┐             │
        ├─ web    → web_search ────┤             │
        └─ hybrid → retrieve_docs → web_search   │
                                   ▼             ▼
                                 grade ─ ok ─▶ generate ─▶ followups
                                   │ not enough
                                   ▼
                             rewrite_query → (other source) → grade
```

- **Router**: an LLM with structured output picks direct / your files / web / both.
- **RAG**: upload PDF, DOCX, TXT or MD. Files are chunked and stored in Pinecone (serverless index is auto-created).
- **Web search**: Tavily.
- **Self-correction (corrective RAG)**: a grader checks the evidence; if it is weak the agent rewrites the query and tries the other source.
- **Cited answers**: streamed token by token with `[1]` chips that jump to the source card.
- **Memory**: per-thread conversation memory via LangGraph `MemorySaver`.
- **AG-UI**: `POST /agent` accepts `RunAgentInput` and streams AG-UI events over SSE
  (`RUN_STARTED`, `STEP_STARTED/FINISHED`, `TOOL_CALL_*`, `TEXT_MESSAGE_*`, `STATE_SNAPSHOT`, `CUSTOM`, `RUN_FINISHED/ERROR`).
  The bundled UI is a small AG-UI client; any AG-UI client (e.g. CopilotKit `HttpAgent`) can talk to the same endpoint.

## Run it (Windows)

```bat
cd ai-agent-studio
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and add `OPENAI_API_KEY`, `TAVILY_API_KEY`, `PINECONE_API_KEY`, then:

```bat
python -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000

## Python 3.14 note

This was built and tested here on Python 3.12 with the versions in `tested-versions.txt`.
Python 3.14 is new, so if `pip install` fails on a package without a 3.14 wheel
(common culprits: `httptools`, `uvloop`-related, `tiktoken`, `pydantic-core`), either:

1. change `uvicorn[standard]` to plain `uvicorn` in `requirements.txt`, or
2. install Python 3.12 or 3.13 and create the venv with `py -3.12 -m venv .venv`.

## Project layout

```
app/
  main.py          FastAPI routes + AG-UI SSE endpoint
  agui.py          AG-UI event names, RunAgentInput, SSE encoder
  graph.py         LangGraph agent (all nodes and edges)
  vectorstore.py   Pinecone setup, file parsing, chunking, ingest/delete
  config.py        pydantic-settings
  templates/index.html
  static/app.css, app.js
data/              uploaded-file registry (created at runtime)
```

## Ideas to extend

- Swap `MemorySaver` for `SqliteSaver` / `PostgresSaver` so chats survive restarts.
- Add more tools as graph nodes (code execution, SQL, email) and let the router pick them.
- Add human-in-the-loop approval with LangGraph `interrupt()` before sensitive tools.
- Add auth and per-user Pinecone namespaces (`PINECONE_NAMESPACE`) for multi-user use.
