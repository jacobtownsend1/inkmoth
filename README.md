# inkmoth

A deep-research agent for any OpenAI-compatible model, local or hosted. Ask a question; it plans
sub-questions, searches, reads pages and PDFs, and writes a cited report. What it does differently:

- **Every citation is backed by a verbatim quote** from text the researcher actually fetched. Claims whose
  quotes cannot be found are dropped, not softened, and never leak into later prompts.
- **It checks coverage before it stops:** after each round an assessor maps every part of your question to
  evidence and sends researchers after what is missing.
- **Weak spots are visible:** citations resting on a single website are marked `†`, and
  conflicting sources get their own *Disagreements* section.
- **Long documents are navigated, not truncated:** a 600-page datasheet or a book-length PDF comes back as an
  outline plus the best-matching passages, and the model can jump to any section.
- **Hardened retrieval:** SSRF-safe fetching, size- and time-bounded parsing in a separate process, run
  budgets, checkpoints and resume. [Security notes](docs/security.md).
- **Honest evals:** small, dated, exploratory, with the rubric. [Latest results](evals/RESULTS-2026-10-10.md)

It runs as a CLI, a Python library, and an MCP server. The MCP server also offers plain `web_search` and
`fetch_page` tools, so coding agents with their own model can use inkmoth as hardened web access.

## Install

```bash
uv tool install "git+https://github.com/jacobtownsend1/inkmoth"
# with the MCP server:
uv tool install "inkmoth[mcp] @ git+https://github.com/jacobtownsend1/inkmoth"
```

## Quick start

You need an OpenAI-compatible model endpoint with tool calling, and a search backend. Configure with
environment variables or a `.env` file in the working directory (see [`.env.example`](.env.example)):

```bash
export INKMOTH_LLM_BASE_URL=http://localhost:8000/v1   # vLLM, SGLang, llama.cpp, Ollama, LM Studio, a hosted API…
export INKMOTH_LLM_MODEL=your-model-name
export INKMOTH_SEARXNG_URL=http://localhost:8080       # or INKMOTH_SEARCH_BACKEND=brave|tavily + INKMOTH_SEARCH_API_KEY

inkmoth doctor                                          # checks the model, tool calling and search
inkmoth ask "How do you turn off DMA channel chaining on the RP2040?"
inkmoth research "Is a shared SQLite file over NFS safe with WAL?" --preset quick
```

`research` asks up to three clarifying questions first when your question could reasonably mean different
things, and otherwise states the assumptions it made in the report. Reports are saved under `reports/`; an
interrupted run continues with `inkmoth research --resume <run_id>`.

**Using SearXNG?** It ships with JSON output off. Add `json` to `search.formats` in its `settings.yml`, or
every query returns HTTP 403 (`inkmoth doctor` tells you).

## Documentation

- [How it works](docs/how-it-works.md): the research loop, what gets rejected and why, long documents and PDFs
- [Configuration](docs/configuration.md): every setting, presets, budgets, checkpoints
- [Model backends](docs/model-backends.md): OpenAI-compatible servers and reasoning toggles
- [Search backends](docs/search-backends.md): SearXNG, Brave, Tavily, writing your own
- [Integrations](docs/integrations.md): Python API, event streams, JSON CLI, MCP server
- [Security](docs/security.md): what the fetch path enforces and what it cannot
- [Development](docs/development.md): tests, evals, code layout

## Status

A personal project, shared as is. Issues and pull requests are welcome. MIT licensed.
