import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_core.messages import AIMessageChunk, HumanMessage

from . import agui
from .config import BASE_DIR, get_settings
from .graph import get_graph
from .vectorstore import SUPPORTED, delete_file, ingest_file, load_registry

settings = get_settings()
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_graph()  # compile once at startup
    yield


app = FastAPI(title=f"{settings.app_name} · AI Agent Studio", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")


# ───────────────────────────── pages & REST ─────────────────────────────
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "app_name": settings.app_name,
        "model": settings.chat_model,
        "accept": ",".join(sorted(SUPPORTED)),
    })


@app.get("/api/health")
async def health():
    return {"ok": not settings.missing_keys(), "missing_keys": settings.missing_keys(),
            "model": settings.chat_model, "index": settings.pinecone_index}


@app.get("/api/documents")
async def list_documents():
    return [{"name": k, "chunks": v["chunks"], "uploaded": v["uploaded"]} for k, v in load_registry().items()]


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    if settings.missing_keys():
        raise HTTPException(400, f"Missing keys in .env: {', '.join(settings.missing_keys())}")
    try:
        return await ingest_file(file.filename, await file.read())
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Could not index the file: {e}")


@app.delete("/api/documents/{name}")
async def remove_document(name: str):
    if not await delete_file(name):
        raise HTTPException(404, "Document not found")
    return {"deleted": name}


# ───────────────────────────── AG-UI endpoint ─────────────────────────────
@app.post("/agent")
async def run_agent(body: agui.RunAgentInput):
    """AG-UI compatible endpoint: accepts RunAgentInput, streams AG-UI events over SSE."""
    return StreamingResponse(
        _run(body),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _run(body: agui.RunAgentInput):
    yield agui.sse({"type": agui.RUN_STARTED, "threadId": body.threadId, "runId": body.runId})

    if settings.missing_keys():
        yield agui.sse({"type": agui.RUN_ERROR,
                        "message": f"Add {', '.join(settings.missing_keys())} to your .env file and restart."})
        return

    last_user = next((m for m in reversed(body.messages) if m.get("role") == "user"), None)
    if not last_user or not str(last_user.get("content", "")).strip():
        yield agui.sse({"type": agui.RUN_ERROR, "message": "No user message received."})
        return

    mode = body.state.get("mode") or body.forwardedProps.get("mode")
    config = {"configurable": {"thread_id": body.threadId, "force_route": mode if mode != "auto" else None}}

    message_id, started = agui.new_id("msg_"), False
    try:
        async for kind, chunk in get_graph().astream(
            {"messages": [HumanMessage(content=str(last_user["content"]))]},
            config,
            stream_mode=["custom", "messages"],
        ):
            if kind == "custom":
                yield agui.sse(dict(chunk))  # nodes already emit AG-UI events
            elif kind == "messages":
                msg, meta = chunk
                if (meta.get("langgraph_node") == "generate"
                        and isinstance(msg, AIMessageChunk) and isinstance(msg.content, str) and msg.content):
                    if not started:
                        started = True
                        yield agui.sse({"type": agui.TEXT_MESSAGE_START, "messageId": message_id, "role": "assistant"})
                    yield agui.sse({"type": agui.TEXT_MESSAGE_CONTENT, "messageId": message_id, "delta": msg.content})
        if started:
            yield agui.sse({"type": agui.TEXT_MESSAGE_END, "messageId": message_id})
        yield agui.sse({"type": agui.RUN_FINISHED, "threadId": body.threadId, "runId": body.runId})
    except asyncio.CancelledError:
        raise
    except Exception as e:
        yield agui.sse({"type": agui.RUN_ERROR, "message": f"{type(e).__name__}: {str(e)[:300]}"})
