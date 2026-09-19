# MCP browser backend

Jev Ultrafast can run as an independently supervised, loopback-only
Streamable HTTP MCP backend:

```powershell
uv sync
$env:JEV_MCP_HOST = "127.0.0.1"
$env:JEV_MCP_PORT = "18766"
uv run jev-mcp
```

Japanese: [mcp.ja.md](mcp.ja.md)

The endpoint is `http://127.0.0.1:18766/mcp`. The public surface is limited to
`jev_browser_start`, `jev_browser_step`, `jev_browser_run`,
`jev_browser_resume_text`, `jev_browser_inspect`, and `jev_browser_close`.

Use `text_mode="caller"` when the MCP caller can provide text. A selected
`TYPE_TEXT` action returns `need_text` and an opaque token. `resume_text`
revalidates the token, page fingerprint, observed action, and actionable
target immediately before its single mutation attempt. `text_mode="internal"`
preserves the existing `TEXT_MODEL_*` helper path.

On the first block, recovery sends only the current page and bounded recent
history to an OpenAI-compatible model configured by `RECOVERY_MODEL_*`, with
fallback to `TEXT_MODEL_*`. It can return only diagnosis, a revised subgoal,
and bounded avoid hints. Jev still chooses every operation and observed
target. An equivalent second block, or exhaustion of the total recovery
budget, returns `handoff_required` and stops Jev mutations.

The handoff identifies the tab by Browser Harness CDP `target_id`, plus a
connection identity, URL, title, fingerprint, and block/recovery counts. It
does not expose a raw WebSocket URL or select a fallback browser backend.

Inject credentials through process environment variables. Do not store them
in this repository. Tests use fixtures and do not call paid APIs.
