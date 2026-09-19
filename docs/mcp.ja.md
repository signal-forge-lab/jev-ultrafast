# MCP browser backend

Jev Ultrafastは、独立監督されるloopback-onlyのStreamable HTTP MCP backendとして起動できます。

```powershell
uv sync
$env:JEV_MCP_HOST = "127.0.0.1"
$env:JEV_MCP_PORT = "18766"
uv run jev-mcp
```

English: [mcp.md](mcp.md)

endpointは `http://127.0.0.1:18766/mcp` です。公開面は
`jev_browser_start`、`jev_browser_step`、`jev_browser_run`、
`jev_browser_resume_text`、`jev_browser_inspect`、`jev_browser_close` の6 toolだけです。

MCP callerが文字を生成できる場合は `text_mode="caller"` を使います。Jevが `TYPE_TEXT` を選ぶと
`need_text` とopaque tokenを返します。`resume_text` は単一mutation attemptの直前にtoken、
page fingerprint、観測済みaction、対象のactionable状態を再検証します。
`text_mode="internal"` は既存の `TEXT_MODEL_*` helper経路を維持します。

最初のblockでは、現在pageと上限付きの直近historyだけをOpenAI-compatible Recovery modelへ送ります。
`RECOVERY_MODEL_API_KEY` / `RECOVERY_MODEL_BASE_URL` / `RECOVERY_MODEL` のいずれかを使う場合は3項目を一式で必須とし、Recovery専用設定が一切ない場合だけ `TEXT_MODEL_*` 一式へfallbackします。返せるのはdiagnosis、
revised subgoal、上限付きavoid hintだけです。operationと観測済みtargetは常にJevが選びます。
同等blockの2回目、またはtotal recovery budget消費時は `handoff_required` を返し、Jev mutationを停止します。

handoffはBrowser Harness CDP `target_id` を主identityとし、実際のBrowser Harness connection名、URL、title、fingerprint、
block/recovery countを含みます。raw WebSocket URLは公開せず、fallback browser backendも選定しません。

credentialはprocess environmentから注入し、repositoryへ保存しないでください。testはfixtureのみを使い、
paid APIを呼びません。
