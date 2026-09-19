"""Offline MCP transport and session ownership contracts."""

import json
from unittest.mock import Mock

import pytest
from mcp.client import Client


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeAgent:
    instances = []

    def __init__(self, url, goal, *, text_mode, recovery, screenshots):
        self.url = url
        self.goal = goal
        self.text_mode = text_mode
        self.recovery = recovery
        self.screenshots = screenshots
        self.closed = False
        self.steps = 0
        self.state = {"status": "ready"}
        self.__class__.instances.append(self)

    def snapshot(self):
        return {"status": self.state["status"], "url": self.url, "steps": self.steps}

    def step(self):
        self.steps += 1
        self.state["status"] = "done"
        return self.snapshot()

    def resume_text(self, token, text):
        return {"status": "ready", "token": token, "text": text}

    def close(self):
        self.closed = True


@pytest.fixture
def backend(monkeypatch):
    from jev_ultrafast import mcp_server

    FakeAgent.instances.clear()
    monkeypatch.setattr(mcp_server, "Agent", FakeAgent)
    registry = mcp_server.SessionRegistry()
    monkeypatch.setattr(mcp_server, "SESSIONS", registry)
    yield mcp_server, registry
    registry.close_all()


def test_session_lifecycle_and_click_only_run_smoke(backend):
    server, registry = backend
    started = server.jev_browser_start("https://example.test", "Click Go")
    session_id = started["session_id"]

    result = server.jev_browser_run(session_id)

    assert result["status"] == "done"
    assert result["steps"] == 1
    assert len(FakeAgent.instances) == 1
    closed = server.jev_browser_close(session_id)
    assert closed == {"status": "closed", "session_id": session_id, "closed": True}
    assert FakeAgent.instances[0].closed
    assert registry.count == 0


def test_close_all_releases_every_owned_session(backend):
    server, registry = backend
    server.jev_browser_start("https://example.test/one", "Click One")
    server.jev_browser_start("https://example.test/two", "Click Two")

    registry.close_all()

    assert registry.count == 0
    assert all(agent.closed for agent in FakeAgent.instances)


def test_unknown_session_fails_without_creating_state(backend):
    server, registry = backend

    with pytest.raises(ValueError, match="Unknown session"):
        server.jev_browser_inspect("missing")

    assert registry.count == 0


@pytest.mark.anyio
async def test_mcp_exposes_only_the_six_browser_tools(backend):
    server, _registry = backend
    expected = {
        "jev_browser_start",
        "jev_browser_step",
        "jev_browser_run",
        "jev_browser_resume_text",
        "jev_browser_inspect",
        "jev_browser_close",
    }

    async with Client(server.mcp) as client:
        tools = await client.list_tools()

    assert {tool.name for tool in tools.tools} == expected


@pytest.mark.anyio
async def test_mcp_call_preserves_structured_result(backend):
    server, _registry = backend

    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "jev_browser_start",
            {"url": "https://example.test", "goal": "Click Go", "text_mode": "caller"},
        )

    assert not result.is_error
    payload = json.loads(result.content[0].text)
    assert payload["status"] == "ready"
    assert payload["session_id"]


def test_resume_is_forwarded_once_under_the_session_lock(backend):
    server, _registry = backend
    started = server.jev_browser_start("https://example.test", "Type a query")
    agent = FakeAgent.instances[0]
    agent.resume_text = Mock(return_value={"status": "done"})

    result = server.jev_browser_resume_text(started["session_id"], "opaque", "query")

    assert result["status"] == "done"
    agent.resume_text.assert_called_once_with("opaque", "query")


@pytest.mark.parametrize("url", ["", "file:///private.txt", "javascript:alert(1)", "https:///missing-host"])
def test_start_rejects_non_web_urls_before_creating_a_browser(backend, url):
    server, registry = backend

    with pytest.raises(ValueError, match="HTTP or HTTPS"):
        server.jev_browser_start(url, "Click Go")

    assert registry.count == 0
    assert not FakeAgent.instances


def test_start_rejects_unbounded_goal_before_creating_a_browser(backend):
    server, registry = backend

    with pytest.raises(ValueError, match="goal"):
        server.jev_browser_start("https://example.test", "x" * 20_001)

    assert registry.count == 0
    assert not FakeAgent.instances


def test_session_limit_fails_closed_without_opening_an_extra_target(backend, monkeypatch):
    server, registry = backend
    monkeypatch.setattr(server, "MAX_SESSIONS", 1)
    server.jev_browser_start("https://example.test/one", "Click One")

    with pytest.raises(ValueError, match="session limit"):
        server.jev_browser_start("https://example.test/two", "Click Two")

    assert registry.count == 1
    assert len(FakeAgent.instances) == 1


def test_failed_close_remains_owned_for_a_bounded_retry(backend):
    server, registry = backend
    started = server.jev_browser_start("https://example.test", "Click Go")
    agent = FakeAgent.instances[0]
    agent.close = Mock(side_effect=[RuntimeError("close failed"), None])

    with pytest.raises(RuntimeError, match="close failed"):
        server.jev_browser_close(started["session_id"])
    assert registry.count == 1

    result = server.jev_browser_close(started["session_id"])
    assert result["closed"]
    assert registry.count == 0


def test_failed_initial_snapshot_closes_unpublished_target(backend, monkeypatch):
    server, registry = backend
    original_snapshot = FakeAgent.snapshot
    monkeypatch.setattr(FakeAgent, "snapshot", Mock(side_effect=RuntimeError("snapshot failed")))

    with pytest.raises(RuntimeError, match="snapshot failed"):
        server.jev_browser_start("https://example.test", "Click Go")

    assert registry.count == 0
    assert FakeAgent.instances[0].closed
    monkeypatch.setattr(FakeAgent, "snapshot", original_snapshot)
