"""The Atlas agent: a LangGraph state machine with routing, RAG, web search,
self-grading (corrective RAG), query rewriting, cited generation and follow-ups.

            ┌────────┐
   START ──▶│ router │──┬── direct ─────────────────────────────┐
            └────────┘  │                                       │
                        ├── docs ──▶ retrieve_docs ─┐           │
                        ├── web ───▶ web_search ────┤           │
                        └── hybrid ▶ retrieve_docs ─▶ web_search┤
                                                    ▼           │
                                                  grade ─ ok ──▶│
                                                    │ not ok    │
                                                    ▼           ▼
                                              rewrite_query  generate ─▶ followups ─▶ END
                                                    └─▶ (other source, then grade again)
"""
import asyncio
import functools
import json
import time
from datetime import date
from functools import lru_cache
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from . import agui
from .config import get_settings
from .vectorstore import get_vectorstore


# ───────────────────────────── state ─────────────────────────────
class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    search_query: str
    route: str
    escalate: bool          # may the agent switch sources if the first one is not enough?
    contexts: list[dict]    # evidence gathered this turn
    used_docs: bool
    used_web: bool
    grade: str
    grade_reason: str
    refinements: int


# ─────────────────────── structured-output schemas ───────────────────────
class RouteDecision(BaseModel):
    route: Literal["direct", "docs", "web", "hybrid"]
    search_query: str = Field(description="Short, self-contained search query")
    reasoning: str = Field(description="One sentence explaining the choice")


class Grade(BaseModel):
    verdict: Literal["sufficient", "insufficient"]
    reason: str = Field(description="One sentence")


class Followups(BaseModel):
    questions: list[str] = Field(description="Three short follow-up questions")


# ───────────────────────────── helpers ─────────────────────────────
@lru_cache
def llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(model=s.chat_model, temperature=0)


@lru_cache
def tavily() -> TavilySearch:
    return TavilySearch(max_results=5, topic="general", include_answer=False)


def step(name: str, label: str):
    """Wrap a node so it emits AG-UI STEP_STARTED / STATE_SNAPSHOT / STEP_FINISHED events."""

    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(state: AgentState, config: RunnableConfig):
            w = get_stream_writer()
            w({"type": agui.STEP_STARTED, "stepName": name, "label": label})
            t0 = time.perf_counter()
            out = await fn(state, config)
            merged = {**state, **out}
            w({
                "type": agui.STATE_SNAPSHOT,
                "snapshot": {
                    "route": merged.get("route"),
                    "searchQuery": merged.get("search_query"),
                    "grade": merged.get("grade"),
                    "gradeReason": merged.get("grade_reason"),
                    "refinements": merged.get("refinements", 0),
                    "evidence": len(merged.get("contexts", [])),
                },
            })
            w({
                "type": agui.STEP_FINISHED,
                "stepName": name,
                "durationMs": int((time.perf_counter() - t0) * 1000),
            })
            return out

        return wrapper

    return deco


def tool_start(w, name: str, args: dict) -> str:
    tid = agui.new_id("call_")
    w({"type": agui.TOOL_CALL_START, "toolCallId": tid, "toolCallName": name})
    w({"type": agui.TOOL_CALL_ARGS, "toolCallId": tid, "delta": json.dumps(args)})
    w({"type": agui.TOOL_CALL_END, "toolCallId": tid})
    return tid


def tool_result(w, tid: str, payload) -> None:
    w({
        "type": agui.TOOL_CALL_RESULT,
        "messageId": agui.new_id("tool_"),
        "toolCallId": tid,
        "role": "tool",
        "content": json.dumps(payload, default=str),
    })


def history(state: AgentState, n: int = 8) -> list[BaseMessage]:
    return state["messages"][-n:]


# ───────────────────────────── nodes ─────────────────────────────
@step("router", "Choosing a strategy")
async def router(state: AgentState, config: RunnableConfig) -> dict:
    w = get_stream_writer()
    question = state["messages"][-1].content
    forced = (config.get("configurable") or {}).get("force_route")

    system = SystemMessage(content=(
        "You are the routing brain of a research agent. Decide how to answer the latest user message.\n"
        "- direct: chit-chat, maths, rewriting/writing, or stable general knowledge.\n"
        "- docs: about the user's uploaded documents / private knowledge base.\n"
        "- web: needs fresh or public information (news, prices, recent events, current facts).\n"
        "- hybrid: needs both the user's documents and up-to-date public information.\n"
        "Also write search_query: short and self-contained (resolve pronouns using the conversation)."
    ))
    decision: RouteDecision = await llm().with_structured_output(RouteDecision).ainvoke(
        [system, *history(state, 7)]
    )
    route = forced if forced in {"docs", "web", "hybrid"} else decision.route

    w({"type": agui.CUSTOM, "name": "route", "value": {
        "route": route,
        "forced": bool(forced),
        "reasoning": decision.reasoning if not forced else f"Mode locked to '{forced}' by you.",
        "searchQuery": decision.search_query,
    }})
    return {
        "question": question,
        "route": route,
        "search_query": decision.search_query or question,
        "escalate": not forced,
        # reset per-turn scratch space (the checkpointer persists state between turns)
        "contexts": [], "used_docs": False, "used_web": False,
        "grade": "", "grade_reason": "", "refinements": 0,
    }


@step("retrieve_docs", "Searching your documents")
async def retrieve_docs(state: AgentState, config: RunnableConfig) -> dict:
    w = get_stream_writer()
    s = get_settings()
    query = state["search_query"]
    tid = tool_start(w, "pinecone_search", {"query": query, "top_k": s.retrieval_k})
    found: list[dict] = []
    try:
        vs = await asyncio.to_thread(get_vectorstore)
        results = await vs.asimilarity_search_with_score(query, k=s.retrieval_k)
        for doc, score in results:
            page = doc.metadata.get("page")
            title = doc.metadata.get("source", "document") + (f" · p.{page}" if page else "")
            found.append({"kind": "doc", "title": title, "url": None,
                          "content": doc.page_content, "score": round(float(score), 3)})
        tool_result(w, tid, {"matches": len(found), "top_score": found[0]["score"] if found else None})
    except Exception as e:  # missing index, empty namespace, network…
        tool_result(w, tid, {"error": str(e)[:200]})
    return {"contexts": [*state.get("contexts", []), *found], "used_docs": True}


@step("web_search", "Searching the web")
async def web_search(state: AgentState, config: RunnableConfig) -> dict:
    w = get_stream_writer()
    query = state["search_query"]
    tid = tool_start(w, "tavily_search", {"query": query, "max_results": 5})
    found: list[dict] = []
    try:
        res = await tavily().ainvoke({"query": query})
        for r in (res.get("results", []) if isinstance(res, dict) else []):
            found.append({"kind": "web", "title": r.get("title") or r.get("url"), "url": r.get("url"),
                          "content": r.get("content", ""), "score": round(float(r.get("score") or 0), 3)})
        tool_result(w, tid, {"results": [{"title": f["title"], "url": f["url"]} for f in found]})
    except Exception as e:
        tool_result(w, tid, {"error": str(e)[:200]})
    return {"contexts": [*state.get("contexts", []), *found], "used_web": True}


@step("grade", "Checking the evidence")
async def grade(state: AgentState, config: RunnableConfig) -> dict:
    ctx = state.get("contexts", [])
    if not ctx:
        return {"grade": "insufficient", "grade_reason": "Nothing relevant was found."}
    snippets = "\n\n".join(f"[{i}] {c['title']}: {c['content'][:500]}" for i, c in enumerate(ctx, 1))
    verdict: Grade = await llm().with_structured_output(Grade).ainvoke([
        SystemMessage(content="Judge whether the evidence is enough to answer the question accurately. "
                              "Be strict: partial or off-topic evidence is 'insufficient'."),
        HumanMessage(content=f"Question: {state['question']}\n\nEvidence:\n{snippets}"),
    ])
    return {"grade": verdict.verdict, "grade_reason": verdict.reason}


@step("rewrite_query", "Rewriting the search")
async def rewrite_query(state: AgentState, config: RunnableConfig) -> dict:
    resp = await llm().ainvoke([
        SystemMessage(content="Rewrite the search query to fix the gap described. Reply with the new query only."),
        HumanMessage(content=f"Question: {state['question']}\nOld query: {state['search_query']}\n"
                             f"Problem: {state.get('grade_reason', '')}"),
    ])
    return {"search_query": str(resp.content).strip().strip('"') or state["search_query"],
            "refinements": state.get("refinements", 0) + 1}


@step("generate", "Writing the answer")
async def generate(state: AgentState, config: RunnableConfig) -> dict:
    w = get_stream_writer()
    ctx = state.get("contexts", [])
    today = date.today().strftime("%A, %d %B %Y")

    if ctx:
        numbered = "\n\n".join(f"[{i}] ({c['kind']}) {c['title']}\n{c['content']}" for i, c in enumerate(ctx, 1))
        caveat = ""
        if state.get("grade") == "insufficient":
            caveat = (f"\nThe evidence is incomplete ({state.get('grade_reason', '')}). "
                      "Say clearly what is missing instead of guessing.")
        system = (f"You are Atlas, a precise research assistant. Today is {today}.\n"
                  "Answer using ONLY the numbered evidence below. Cite claims inline like [1] or [2][3]. "
                  "Use clear Markdown; keep it tight." + caveat + f"\n\nEVIDENCE:\n{numbered}")
        w({"type": agui.CUSTOM, "name": "sources", "value": [
            {"id": i, "kind": c["kind"], "title": c["title"], "url": c["url"],
             "snippet": c["content"][:240], "score": c["score"]} for i, c in enumerate(ctx, 1)]})
    else:
        system = (f"You are Atlas, a precise, friendly assistant. Today is {today}. "
                  "Answer directly and concisely using Markdown.")

    resp = await llm().ainvoke([SystemMessage(content=system), *history(state, 10)])
    return {"messages": [resp]}


@step("followups", "Suggesting next questions")
async def followups(state: AgentState, config: RunnableConfig) -> dict:
    w = get_stream_writer()
    try:
        answer = state["messages"][-1].content
        out: Followups = await llm().with_structured_output(Followups).ainvoke([
            SystemMessage(content="Suggest three short follow-up questions the user might ask next."),
            HumanMessage(content=f"Question: {state['question']}\n\nAnswer: {answer}"),
        ])
        w({"type": agui.CUSTOM, "name": "followups", "value": out.questions[:3]})
    except Exception:
        pass
    return {}


# ───────────────────────────── edges ─────────────────────────────
def after_router(state: AgentState) -> str:
    return {"direct": "generate", "docs": "retrieve_docs", "web": "web_search",
            "hybrid": "retrieve_docs"}[state["route"]]


def after_retrieve(state: AgentState) -> str:
    return "web_search" if state["route"] == "hybrid" and not state.get("used_web") else "grade"


def after_grade(state: AgentState) -> str:
    if state["grade"] == "sufficient" or not state.get("escalate"):
        return "generate"
    if state.get("refinements", 0) >= get_settings().max_refinements:
        return "generate"
    if state.get("used_web") and state.get("used_docs"):
        return "generate"
    return "rewrite_query"


def after_rewrite(state: AgentState) -> str:
    return "retrieve_docs" if state.get("used_web") and not state.get("used_docs") else "web_search"


# ───────────────────────────── graph ─────────────────────────────
@lru_cache
def get_graph():
    g = StateGraph(AgentState)
    for name, fn in [("router", router), ("retrieve_docs", retrieve_docs), ("web_search", web_search),
                     ("grade", grade), ("rewrite_query", rewrite_query), ("generate", generate),
                     ("followups", followups)]:
        g.add_node(name, fn)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", after_router, ["generate", "retrieve_docs", "web_search"])
    g.add_conditional_edges("retrieve_docs", after_retrieve, ["web_search", "grade"])
    g.add_edge("web_search", "grade")
    g.add_conditional_edges("grade", after_grade, ["generate", "rewrite_query"])
    g.add_conditional_edges("rewrite_query", after_rewrite, ["web_search", "retrieve_docs"])
    g.add_edge("generate", "followups")
    g.add_edge("followups", END)
    # MemorySaver = per-thread conversation memory. Swap for SqliteSaver/PostgresSaver in production.
    return g.compile(checkpointer=MemorySaver())
