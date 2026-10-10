"""Progress events. The pipeline emits these; adapters (CLI, MCP server) render them."""
import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .schemas import Plan, Report


@dataclass
class Event:
    ts: float = field(default_factory=time.time, init=False)
    run_id: str = field(default="", kw_only=True)
    seq: int = field(default=0, kw_only=True)


@dataclass
class PhaseStarted(Event):
    phase: str
    detail: str = ""


@dataclass
class PlanReady(Event):
    plan: Plan
    round: int


@dataclass
class SubQuestionStarted(Event):
    sq_id: int
    question: str
    round: int


@dataclass
class ToolCalled(Event):
    agent: str
    tool: str
    args: dict[str, Any]
    call_id: str = ""


@dataclass
class ToolResult(Event):
    agent: str
    tool: str
    ok: bool
    chars: int
    seconds: float
    call_id: str = ""


@dataclass
class ReasoningDelta(Event):
    agent: str
    text: str


@dataclass
class TextDelta(Event):
    agent: str
    text: str
    attempt: int = 1


@dataclass
class TextReset(Event):
    agent: str
    attempt: int


@dataclass
class SubQuestionDone(Event):
    sq_id: int
    ok: bool
    facts: int = 0
    gaps: int = 0
    error: str | None = None


@dataclass
class CheckpointReady(Event):
    report: Report


@dataclass
class RunCancelled(Event):
    pass


@dataclass
class RoundDone(Event):
    round: int
    gaps: list[str]


@dataclass
class ReportReady(Event):
    report: Report


@dataclass
class ErrorEvent(Event):
    where: str
    message: str


class EventSink(Protocol):
    async def emit(self, event: Event) -> None: ...


class NullSink:
    async def emit(self, event: Event) -> None:
        return None


class CollectingSink:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)


def describe(e: Event) -> str | None:
    """One human-readable progress line per event (None: not worth showing). Shared by the CLI and MCP."""
    if isinstance(e, PhaseStarted):
        return f"▶ {e.phase}: {e.detail}"
    if isinstance(e, PlanReady):
        return "\n".join(f"  {sq.id}. {sq.question}" for sq in e.plan.sub_questions)
    if isinstance(e, SubQuestionStarted):
        return f"  r{e.sq_id} start (round {e.round})"
    if isinstance(e, ToolCalled):
        return f"    {e.agent} {e.tool}: {e.args.get('query') or e.args.get('url') or json.dumps(e.args)}"
    if isinstance(e, ToolResult) and not e.ok:
        return f"    {e.agent} {e.tool} failed"
    if isinstance(e, SubQuestionDone):
        return f"  r{e.sq_id} done: facts={e.facts} gaps={e.gaps}" if e.ok else f"  r{e.sq_id} failed: {e.error}"
    if isinstance(e, RoundDone):
        return f"round {e.round} done; gaps={len(e.gaps)}"
    if isinstance(e, ErrorEvent):
        return f"error {e.where}: {e.message}"
    if isinstance(e, ReportReady):
        s = e.report.stats
        return f"{e.report.status}: {s.seconds}s; sources={s.sources}; tokens={s.input_tokens}/{s.output_tokens}"
    return None
