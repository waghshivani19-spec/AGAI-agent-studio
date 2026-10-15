"""Minimal AG-UI (Agent-User Interaction Protocol) event helpers.

AG-UI is an open, event-based protocol between an agent backend and a UI.
The backend streams typed JSON events over Server-Sent Events; any AG-UI client
(including CopilotKit's HttpAgent) can consume them. This file implements the
subset of event types Atlas uses, with the standard camelCase field names.
"""
import json
import time
import uuid

from pydantic import BaseModel, ConfigDict, Field

# Event type names (subset of the AG-UI spec)
RUN_STARTED = "RUN_STARTED"
RUN_FINISHED = "RUN_FINISHED"
RUN_ERROR = "RUN_ERROR"
STEP_STARTED = "STEP_STARTED"
STEP_FINISHED = "STEP_FINISHED"
TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
TOOL_CALL_START = "TOOL_CALL_START"
TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
TOOL_CALL_END = "TOOL_CALL_END"
TOOL_CALL_RESULT = "TOOL_CALL_RESULT"
STATE_SNAPSHOT = "STATE_SNAPSHOT"
CUSTOM = "CUSTOM"


class RunAgentInput(BaseModel):
    """Request body of POST /agent (matches the AG-UI RunAgentInput shape)."""

    model_config = ConfigDict(extra="allow")

    threadId: str
    runId: str = Field(default_factory=lambda: uuid.uuid4().hex)
    messages: list[dict] = Field(default_factory=list)
    state: dict = Field(default_factory=dict)
    forwardedProps: dict = Field(default_factory=dict)


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def sse(event: dict) -> str:
    event.setdefault("timestamp", int(time.time() * 1000))
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
