"""CLI adapter. Reports on stdout; progress (human or NDJSON) on stderr."""
from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Optional

import httpx
import typer
from pydantic import BaseModel, ValidationError
from rich.console import Console

from .config import PRESETS, Settings
from .doctor import run_checks
from .events import Event, ReasoningDelta, describe
from .pipeline import run_ask_result, run_clarify, run_research, save_report
from .schemas import Report
from .search import format_results, make_backend
from .sources import SourceRegistry

app = typer.Typer(help="Deep research with any OpenAI-compatible model and a pluggable search backend.", no_args_is_help=True)
err = Console(stderr=True, markup=False, highlight=False, soft_wrap=True)  # no hard breaks in URLs or commands


class OutputFormat(str, Enum):
    markdown = "markdown"
    json = "json"


class JsonSink:
    run_id = ""  # remembered so an interrupted run can say how to resume

    async def emit(self, event: Event):
        self.run_id = event.run_id
        data = asdict(event)
        data["type"] = type(event).__name__
        sys.stderr.write(json.dumps(data, default=lambda v: v.model_dump(mode="json")
                                    if isinstance(v, BaseModel) else str(v)) + "\n")
        sys.stderr.flush()


class RichSink:
    run_id = ""

    def __init__(self, show_thinking=False):
        self.show_thinking = show_thinking

    async def emit(self, e):
        self.run_id = e.run_id
        if isinstance(e, ReasoningDelta):
            if self.show_thinking:
                err.print(e.text, end="", style="dim")
        elif (line := describe(e)) is not None:
            err.print(line)


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stderr.isatty()


def _ask_user(question: str, s: Settings, force: bool) -> tuple[str | None, list[str] | None]:
    """One clarification round on the terminal: (answers, None) or (None, assumptions)."""
    try:
        check = asyncio.run(run_clarify(question, s, force=force))
    except Exception as e:
        err.print(f"clarification check failed ({type(e).__name__}); continuing")
        return None, []  # [] also stops the pipeline re-running the check
    if not check.questions or not (check.needs_clarification or force):
        return None, check.assumptions
    err.print("Before researching:")
    for i, q in enumerate(check.questions, 1):
        err.print(f"  {i}. {q}")
    if check.assumptions:
        err.print("If you just press Enter (or type 'proceed'), I will assume: " + "; ".join(check.assumptions))
    reply = err.input("> ").strip()
    return (None, check.assumptions) if reply.lower() in ("", "proceed") else (reply, None)


Preset = Enum("Preset", {name: name for name in PRESETS}, type=str)


def _settings(preset: str | None = None, **overrides):
    try:
        return Settings(**{**PRESETS.get(preset or "", {}), **{k: v for k, v in overrides.items() if v is not None}})
    except ValidationError as e:
        err.print(f"invalid settings (set INKMOTH_* env vars or a .env file):\n{e}")
        raise typer.Exit(2)


@app.command()
def research(question: Optional[str] = typer.Argument(None, help="Omit with --resume"),
             preset: Optional[Preset] = typer.Option(None, "--preset",
                                                     help="Run size: quick, standard (the defaults) or thorough"),
             breadth: Optional[int] = typer.Option(None, min=1, max=20),
             depth: Optional[int] = typer.Option(None, min=1, max=10),
             parallel: Optional[int] = typer.Option(None, min=1, max=32),
             no_thinking: bool = typer.Option(False, "--no-thinking"),
             show_thinking: bool = typer.Option(False, "--show-thinking"),
             out: Optional[Path] = typer.Option(None, "--out"),
             json_out: bool = typer.Option(False, "--json", help="Also save a JSON report sidecar"),
             format: OutputFormat = typer.Option(OutputFormat.markdown, "--format", help="stdout format"),
             events_json: bool = typer.Option(False, "--events-json", help="NDJSON progress on stderr"),
             no_save: bool = typer.Option(False, "--no-save", help="Disable reports and checkpoints"),
             resume: Optional[str] = typer.Option(None, "--resume", help="Continue an interrupted run: run id or checkpoint path"),
             clarify: bool = typer.Option(False, "--clarify", help="Always ask clarifying questions first"),
             no_clarify: bool = typer.Option(False, "--no-clarify", help="Never ask; proceed on stated assumptions")):
    """Plan, research, verify evidence, and produce a report."""
    s = _settings(preset.value if preset else None, breadth=breadth, depth=depth, parallel=parallel, reports_dir=out)
    if no_thinking:
        s.thinking_planner = s.thinking_researcher = s.thinking_synthesizer = False
    checkpoints = s.checkpoint_dir or s.reports_dir / ".checkpoints"
    s.checkpoint_dir = None if no_save else checkpoints
    saved = None
    if resume:
        path = Path(resume) if Path(resume).is_file() else checkpoints / f"{resume}.json"
        if not path.is_file():
            err.print(f"no checkpoint at {path}")
            raise typer.Exit(2)
        saved = Report.model_validate_json(path.read_text(encoding="utf-8"))
    elif not question:
        err.print("a question is required (or --resume)")
        raise typer.Exit(2)
    answers = assumptions = None
    interactive = _interactive() and not events_json
    if clarify and not interactive:
        err.print("--clarify needs an interactive terminal")
        raise typer.Exit(2)
    if not saved and (clarify or (interactive and not no_clarify)):
        answers, assumptions = _ask_user(question, s, force=clarify)
    sink = JsonSink() if events_json else RichSink(show_thinking)
    try:
        report = asyncio.run(run_research(question or "", s, sink, resume=saved, answers=answers,
                                          assumptions=assumptions))
    except ValueError as e:
        err.print(str(e))
        raise typer.Exit(2)
    except KeyboardInterrupt:
        if s.checkpoint_dir and sink.run_id:
            err.print(f"interrupted; continue with: inkmoth research --resume {sink.run_id}")
        raise typer.Exit(130)
    if not no_save:
        try:
            path = save_report(report, s.reports_dir, as_json=json_out)
            if not events_json:
                err.print(f"Saved: {path}")
        except OSError as e:  # still print the report below
            err.print(f"report not saved: {e}")
    typer.echo(report.model_dump_json() if format == OutputFormat.json else report.report_md)
    if report.status == "failed":
        raise typer.Exit(1)


@app.command()
def ask(question: str, show_thinking: bool = typer.Option(False, "--show-thinking"),
        format: OutputFormat = typer.Option(OutputFormat.markdown, "--format"),
        events_json: bool = typer.Option(False, "--events-json")):
    """Quick answer with fetched evidence and verified citation provenance."""
    try:
        answer = asyncio.run(run_ask_result(question, _settings(), JsonSink() if events_json else RichSink(show_thinking)))
    except ValueError as e:
        err.print(str(e))
        raise typer.Exit(2)
    typer.echo(answer.model_dump_json() if format == OutputFormat.json else answer.answer_md)
    if answer.status == "failed":
        raise typer.Exit(1)


@app.command()
def search(query: str, category: str = typer.Option("general"),
           time_range: Optional[str] = typer.Option(None), n: int = typer.Option(10, min=1, max=30)):
    """Raw query against the configured search backend."""
    async def run():
        backend = make_backend(_settings(llm_model="(unused by search)"))
        try:
            return await backend.search(query, category=category, time_range=time_range)
        finally:
            await backend.aclose()
    try:
        response = asyncio.run(run())
    except ValueError as e:
        err.print(str(e))
        raise typer.Exit(2)
    except httpx.HTTPError as e:
        err.print(f"search failed: {type(e).__name__}: {e}")
        raise typer.Exit(1)
    typer.echo(format_results(response, SourceRegistry(), max_results=n))


@app.command()
def doctor():
    """Check the LLM, tool calling, and the search backend."""
    checks = asyncio.run(run_checks(_settings()))
    for check in checks:
        typer.echo(f"{'OK' if check.ok else 'FAIL'} {check.name}: {check.detail}")
    if not all(check.ok for check in checks):
        raise typer.Exit(1)


@app.command()
def mcp():
    """Serve research, ask and clarify as MCP tools over stdio (needs the 'mcp' extra)."""
    try:
        from .mcp_server import main
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] != "mcp":
            raise
        err.print("MCP support is not installed; install inkmoth with the 'mcp' extra, e.g. inkmoth[mcp]")
        raise typer.Exit(2)
    main()


if __name__ == "__main__":
    app()
