# Integrations

## Python

```python
import asyncio
from inkmoth import PRESETS, Settings, run_clarify, run_research, run_ask_result

async def main():
    settings = Settings(**PRESETS["quick"])
    question = "Is a shared SQLite file over NFS safe with WAL?"

    check = await run_clarify(question, settings)        # optional: only if you can ask someone
    if check.needs_clarification:
        print(check.questions)                           # ask the user, then pass answers=...
    report = await run_research(question, settings, assumptions=check.assumptions)

    print(report.status)        # complete, partial, failed, cancelled
    print(report.report_md)
    for result in report.results:
        for fact in (result.findings.facts if result.findings else []):
            print(fact.claim, fact.source_ids, [e.quote for e in fact.evidence])

    answer = await run_ask_result("What is HTTPX's asynchronous client called?", settings)
    print(answer.answer_md)

asyncio.run(main())
```

- `run_research(question, settings, sink=None, *, answers=None, assumptions=None, resume=None)` returns a
  `Report`. With neither `answers` nor `assumptions` it runs the clarification check itself and proceeds on
  its assumptions; the library never asks anyone. `resume` takes a saved `Report` (a checkpoint).
- `run_ask_result` returns an `Answer` with sources, status, warnings and usage.
- The library writes nothing to disk unless `Settings.checkpoint_dir` is set or you call
  `inkmoth.pipeline.save_report`.

### Progress events

Pass any object with `async emit(event)` as `sink`. Every event has `run_id`, a monotonic `seq` and a
timestamp; tool events have `call_id`, synthesis text events have `attempt`. Event types live in
`inkmoth.events`, and `inkmoth.events.describe(event)` gives the one-line text the CLI prints.

- `TextReset` invalidates the previous attempt's streamed text. Streamed text has not passed the final
  citation checks; the returned `Report`/`Answer` (or `ReportReady.report`) is authoritative.
- `CheckpointReady` carries a snapshot after each finished researcher.
- Sinks are awaited: return quickly, queue expensive work, and handle your own errors.

### Run ownership

Use `RunContext` when you need explicit control of the source registry, budgets and HTTP clients:

```python
from inkmoth import RunContext, Settings, run_research

async def research_with_sink(question, sink):
    settings = Settings(_env_file=None)
    async with RunContext(settings, sink=sink) as context:
        return await run_research(question, settings, context=context)
```

Contexts are single-use; never share one between concurrent runs. Contexts the library creates are closed on
success, failure and cancellation; contexts and brains you pass in remain yours to close.

A custom `Brain` (see `inkmoth.agents.Brain`) is passed as `brain=`. Its `registry` must be the context's
(without `context=`, one is created around it), and it must record fetched evidence under researcher names
`r1`, `r2`, …. The pipeline re-checks every quote it returns. `assess(question, results, max_n, next_id)`
runs after every round, including the last with `max_n=0`.

## Command line as a subprocess

```bash
inkmoth research "your question" --format json --events-json --no-save   # JSON report on stdout, NDJSON events on stderr
inkmoth ask "your question" --format json --events-json
```

`--format json` makes stdout machine-readable; `--events-json` makes stderr NDJSON (and never asks clarifying
questions). Validation errors and crashes can still print plain text, so check the exit code: 0 for complete
or partial, 1 for a failed run, 2 for bad arguments. `--no-save` disables reports and checkpoints.

Any editor or agent extension can drive inkmoth this way: spawn the CLI with an argument array, parse
stdout/stderr, cancel by terminating the child. Keep the host's privileged tools and private context out of
the research question.

## MCP server

Install with the `mcp` extra and register `inkmoth mcp` (stdio) with your MCP host. For hosts using the
common `mcpServers` JSON format:

```json
{
  "mcpServers": {
    "inkmoth": {
      "command": "inkmoth",
      "args": ["mcp"],
      "env": {
        "INKMOTH_LLM_BASE_URL": "http://localhost:8000/v1",
        "INKMOTH_LLM_MODEL": "your-model-name",
        "INKMOTH_SEARXNG_URL": "http://localhost:8080"
      }
    }
  }
}
```

| Tool | What it does |
|---|---|
| `research(question, preset="standard", answers=None)` | Full research; returns the Markdown report. Streams progress as MCP log and progress notifications. If the host supports elicitation and the question is ambiguous, it asks the user through the host; otherwise it proceeds on stated assumptions. |
| `start_research(question, preset, answers)` → `run_id` | Same, in the background, for hosts that time out long tool calls. Never asks. |
| `get_report(run_id)` | Status, plus the report (partial while running). Reads the run's checkpoint, so finished runs survive a server restart. |
| `clarify(question)` | Questions worth asking and the assumptions used otherwise; pass the replies to `research(answers=...)`. |
| `ask(question)` | Quick single-researcher answer. |
| `web_search(query, category, time_range)` | Plain search through the configured backend, for agents that do their own research. No model needed. |
| `fetch_page(url, focus, at)` | Plain page/PDF reading with the same SSRF-safe fetch, outline and passage navigation. No model needed. |

Checkpoints go to `$XDG_STATE_HOME/inkmoth/.checkpoints` (default `~/.local/state/inkmoth/.checkpoints`),
or to `<INKMOTH_REPORTS_DIR>/.checkpoints` when that is set: hosts start servers in arbitrary directories.

Runs can take from a few minutes (`quick`) to an hour (`thorough`). `web_search` and `fetch_page` work without
any `INKMOTH_LLM_*` settings, so a coding agent with its own model can use inkmoth just as its web access,
with whichever search backend you configured.

Example with the [Pi](https://pi.dev) coding agent (1.0 or later):

```bash
pi mcp add inkmoth --env INKMOTH_SEARXNG_URL=http://localhost:8080 -- inkmoth mcp
pi mcp list
```
