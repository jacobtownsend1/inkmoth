# Configuration

Every setting can be passed as a keyword to `inkmoth.Settings`, set as an `INKMOTH_<NAME>` environment
variable, or put in a `.env` file in the working directory. Explicit arguments beat environment variables,
which beat `.env`. Libraries embedding inkmoth can pass `_env_file=None` to skip `.env` discovery. Values are
validated on creation and on assignment.

## Presets

`inkmoth research --preset quick|standard|thorough` (or `Settings(**inkmoth.PRESETS["quick"])`) sets the
run size in one go. A preset overrides environment defaults; explicit options such as `--depth` override the
preset. `standard` is exactly the defaults below. `inkmoth ask` is the fastest path when one researcher is
enough.

| | quick | standard | thorough |
|---|---|---|---|
| breadth × depth | 2 × 1 | 4 × 2 | 6 × 3 |
| searches per researcher | 4 | 6 | 8 |
| run searches / fetches | 12 / 12 | 48 / 48 | 120 / 120 |
| model calls | 60 | 160 | 400 |
| token cap | 400 k | 2 M | 6 M |
| run timeout | 10 min | 30 min | 60 min |

## Settings

| Variable (`INKMOTH_…`) | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible endpoint, up to `/v1` |
| `LLM_MODEL` | *(required)* | Model name as the server lists it |
| `LLM_API_KEY` | `EMPTY` | Provider credential |
| `LLM_EXTRA_BODY_THINKING` / `LLM_EXTRA_BODY_NO_THINKING` | `{}` / `{}` | JSON merged into requests; see [model backends](model-backends.md) |
| `SEARCH_BACKEND` | `searxng` | `searxng`, `brave` or `tavily` |
| `SEARCH_API_KEY` | empty | Required for `brave` and `tavily` |
| `SEARXNG_URL` | `http://localhost:8080` | SearXNG instance root |
| `BREADTH` / `DEPTH` / `PARALLEL` | 4 / 2 / 4 | Sub-questions per round / rounds / concurrent researchers |
| `RESEARCHER_SEARCHES` / `RESEARCHER_TURNS` | 6 / 12 | Per-researcher limits |
| `MAX_SEARCHES` / `MAX_FETCHES` | 48 / 48 | Run-wide network budgets, shared by researchers |
| `MAX_MODEL_CALLS` | 160 | Run-wide model request cap, including extraction, assessment and retries |
| `MAX_TOTAL_TOKENS` | 2000000 | Soft token cap, checked before each model call |
| `RESEARCHER_TIMEOUT` / `PHASE_TIMEOUT` / `RUN_TIMEOUT` | 900 / 600 / 1800 | Deadlines in seconds |
| `REQUEST_TIMEOUT` | 600 | Model HTTP timeout in seconds |
| `SEARCH_MAX_RESULTS` | 8 | Results shown to the model per search |
| `FETCH_MAX_CHARS` | 6000 | Characters returned per `fetch_page` call (min 2000) |
| `FETCH_MAX_BYTES` / `FETCH_MAX_PDF_BYTES` | 2 MB / 50 MB | Download caps for HTML/text and for PDFs |
| `FETCH_TIMEOUT` | 30 | Download deadline in seconds |
| `EXTRACTION_TIMEOUT` | 60 | Parsing deadline; also the parser process's CPU limit (pypdf needs ~0.65 s per MB) |
| `THINKING_PLANNER` / `THINKING_RESEARCHER` / `THINKING_SYNTHESIZER` | true / false / false | Reasoning per role, if the extra bodies are set |
| `REPAIR_EVIDENCE` | true | One re-extraction when quotes fail verification |
| `REVIEW_SYNTHESIS` | true | Separate reviewer pass over the draft |
| `TEMPERATURE` | 0.3 | Sampling temperature |
| `MAX_TOKENS` / `SYNTH_MAX_TOKENS` | 8192 / 16384 | Output caps per response |
| `REPORTS_DIR` | `reports` | Where the CLI saves reports (and `.checkpoints/`); the MCP server defaults to `~/.local/state/inkmoth` |
| `CHECKPOINT_DIR` | unset | Library checkpoints; the CLI uses `<reports_dir>/.checkpoints` unless `--no-save` |

## Budgets and limits

- Network attempts consume budgets even when they fail. Cache hits do not consume run-wide budgets, but
  each `web_search` call still counts against the researcher's own allowance. Reading more of an already
  fetched document never counts as a fetch.
- Token totals use provider-reported usage. Concurrent requests can overshoot the soft cap; streams that end
  without usage set `stats.usage_incomplete`. Call counts and deadlines are hard limits.
- The model client does not retry HTTP requests itself; explicit fallbacks count against the run budget.
- Redirects are limited to five per fetch.

## Checkpoints and resume

The CLI writes `reports/.checkpoints/<run_id>.json` after every researcher and at the end; the library only
writes checkpoints when `checkpoint_dir` is set. `inkmoth research --resume <run_id or path>` (library:
`run_research(..., resume=report)`) continues an interrupted run; see [how it works](how-it-works.md#resuming).
