"""Loopback Streamable HTTP MCP facade for owned Jev browser sessions."""

import atexit
import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlsplit

from mcp.server.mcpserver import MCPServer

from .agent import Agent
from .browser import StalePage

TERMINAL_STATUSES = {"done", "blocked", "need_text", "handoff_required", "recovery_unavailable"}
MAX_SESSIONS = 32


@dataclass
class OwnedSession:
    agent: Agent
    lock: threading.RLock = field(default_factory=threading.RLock)
    closed: bool = False


class SessionRegistry:
    """Owns each Agent and serializes every operation within a session."""

    def __init__(self):
        self._sessions = {}
        self._starting = 0
        self._lock = threading.RLock()

    @property
    def count(self):
        with self._lock:
            return len(self._sessions)

    def start(self, url, goal, text_mode):
        with self._lock:
            if len(self._sessions) + self._starting >= MAX_SESSIONS:
                raise ValueError(f"The {MAX_SESSIONS}-session limit is reached; no browser target was opened.")
            self._starting += 1
        agent = None
        try:
            agent = Agent(url, goal, text_mode=text_mode, recovery=True, screenshots=False)
            snapshot = agent.snapshot()
        except Exception:
            if agent is not None:
                try:
                    agent.close()
                except Exception:
                    pass
            with self._lock:
                self._starting -= 1
            raise
        session_id = uuid.uuid4().hex
        with self._lock:
            self._starting -= 1
            self._sessions[session_id] = OwnedSession(agent)
        return session_id, snapshot

    def get(self, session_id):
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise ValueError("Unknown session; no browser operation was attempted.")
        return session

    def close(self, session_id):
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            return False
        with session.lock:
            if session.closed:
                return False
            session.agent.close()
            session.closed = True
        with self._lock:
            if self._sessions.get(session_id) is session:
                self._sessions.pop(session_id)
        return True

    def close_all(self):
        with self._lock:
            session_ids = list(self._sessions)
        for session_id in session_ids:
            try:
                self.close(session_id)
            except Exception:
                continue


SESSIONS = SessionRegistry()
atexit.register(SESSIONS.close_all)

mcp = MCPServer(
    "jev-ultrafast",
    version="0.1.0",
    instructions="Jev selects observed browser operations and targets; callers may supply text through resume tokens.",
)


def _with_session(session_id, operation):
    session = SESSIONS.get(session_id)
    with session.lock:
        if session.closed:
            raise ValueError("Session is closed; no browser operation was attempted.")
        return {"session_id": session_id, **operation(session.agent)}


@mcp.tool()
def jev_browser_start(
    url: str,
    goal: str,
    text_mode: Literal["caller", "internal"] = "caller",
) -> dict:
    """Start one owned browser target for a natural-language goal."""
    parsed = urlsplit(url)
    if len(url) > 2048 or parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("url must be an HTTP or HTTPS URL with a host and at most 2048 characters")
    goal = goal.strip()
    if not goal or len(goal) > 20_000:
        raise ValueError("goal must contain between 1 and 20000 characters")
    session_id, snapshot = SESSIONS.start(url, goal, text_mode)
    return {"session_id": session_id, **snapshot}


@mcp.tool()
def jev_browser_step(session_id: str) -> dict:
    """Run one bounded Jev decision step, including bounded recovery when blocked."""
    return _with_session(session_id, lambda agent: agent.step())


@mcp.tool()
def jev_browser_run(session_id: str, max_cycles: int = 120) -> dict:
    """Run until completion, caller text, recovery stop, handoff, or the cycle bound."""
    if not 1 <= max_cycles <= 120:
        raise ValueError("max_cycles must be between 1 and 120")

    def run(agent):
        result = agent.snapshot()
        for _ in range(max_cycles):
            if result["status"] in TERMINAL_STATUSES:
                return result
            result = agent.step()
        return {**result, "status": "step_limit"}

    return _with_session(session_id, run)


@mcp.tool()
def jev_browser_resume_text(session_id: str, resume_token: str, text: str) -> dict:
    """Supply caller-generated text after revalidating token, page, action, and target."""

    def resume(agent):
        try:
            return agent.resume_text(resume_token, text)
        except StalePage as error:
            return {**agent.snapshot(), "status": "stale", "error": str(error)}

    return _with_session(session_id, resume)


@mcp.tool()
def jev_browser_inspect(session_id: str) -> dict:
    """Inspect current page, decisions, history, recovery, and handoff metadata."""
    return _with_session(session_id, lambda agent: agent.snapshot())


@mcp.tool()
def jev_browser_close(session_id: str) -> dict:
    """Close the owned browser target. Repeated close calls are safe."""
    closed = SESSIONS.close(session_id)
    return {"status": "closed", "session_id": session_id, "closed": closed}


def main():
    host = os.environ.get("JEV_MCP_HOST", "127.0.0.1")
    if host != "127.0.0.1":
        raise ValueError("JEV_MCP_HOST must remain 127.0.0.1")
    try:
        port = int(os.environ.get("JEV_MCP_PORT", "18766"))
    except ValueError:
        raise ValueError("JEV_MCP_PORT must be an integer") from None
    if not 1 <= port <= 65535:
        raise ValueError("JEV_MCP_PORT must be between 1 and 65535")
    mcp.run(
        transport="streamable-http",
        host=host,
        port=port,
        stateless_http=True,
        json_response=True,
    )


if __name__ == "__main__":
    main()
