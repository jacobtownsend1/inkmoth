# Development

```bash
uv sync --all-extras
uv run pytest               # deterministic suite: no network, no model (CI runs this)
uv run pytest -m live       # also the live tests: your configured model, search backend and the public web
```

The deterministic suite uses fake brains, `respx`-mocked HTTP, real PDF bytes built with pypdf, and the MCP
protocol over in-memory streams. Live tests are deselected by default (`-m 'not live'` in `pyproject.toml`).

## Evals

`evals/` holds small exploratory evaluations against a real model and the real web: questions, a rubric the
model never sees, and scripts that keep every report, event stream and checkpoint. They are not a benchmark.
See [evals/README.md](../evals/README.md) for the cases, how to run them and how to score them.

## Layout

| Module | Role |
|---|---|
| `pipeline.py` | Orchestration: clarify, rounds, assessment, synthesis, checkpoints, resume |
| `agents.py` | Strands-based `StrandsBrain`: prompts, roles, structured outputs, tool wiring |
| `context.py`, `budget.py`, `config.py` | Run ownership, budgets, settings and presets |
| `schemas.py`, `sources.py`, `evidence.py` | Result models, source registry and evidence ledger, provenance checks and report finishing |
| `search.py` | Search seam: result models, `SearchBackend` protocol, `web_search` tool |
| `searx.py`, `brave.py`, `tavily.py` | Search backends |
| `fetch.py`, `passages.py`, `extract.py` | Download, per-run document cache, chunking/BM25/outline, extraction worker |
| `netguard.py`, `transport.py` | Destination checks and the DNS-pinning transport |
| `events.py`, `storage.py` | Event types and formatting, atomic writes |
| `cli.py`, `doctor.py`, `mcp_server.py` | Adapters |

Adding a search backend is described in [search backends](search-backends.md#writing-a-backend).
