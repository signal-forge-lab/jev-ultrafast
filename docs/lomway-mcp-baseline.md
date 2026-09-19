# Lomway MCP baseline

Recorded: 2026-09-19

- Upstream: `browser-use/jev-ultrafast`
- Upstream commit: `1231850a0bf1a0c0341fe408ef1668dbbfdfac46`
- Development branch: `feature/lomway-mcp-recovery`
- Python requirement: `>=3.12`
- Baseline interpreter: CPython `3.13.13`
- Browser Harness: `0.1.13`
- TypeSafe integration: direct HTTPS through `httpx`; no TypeSafe SDK dependency
- HTTP client: `httpx 0.28.1`

The unmodified upstream suite passed 31 tests with Python UTF-8 mode. On the
Windows CP932 locale, collection failed before any test because `snapshot.js`
was read using the platform default encoding. The branch fixes that baseline
portability issue by reading the JavaScript source explicitly as UTF-8.
