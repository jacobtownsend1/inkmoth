"""Bounded research orchestration with partial results and optional atomic checkpoints."""
from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from .config import Settings
from .context import RunContext
from .events import (CheckpointReady, ErrorEvent, EventSink, PhaseStarted, PlanReady, ReportReady,
                     RoundDone, RunCancelled, SubQuestionDone, SubQuestionStarted)
from .evidence import (cited_ids, ensure_disagreements, fallback_report, finish_markdown, frame_report,
                       rejection_notes, validate_findings)
from .schemas import Answer, Assessment, ClarificationCheck, Plan, Report, Stats, SubQuestion, SubQuestionResult
from .sources import SourceRegistry
from .storage import atomic_write

if TYPE_CHECKING:
    from .agents import Brain


def _dependencies(settings, sink, brain, context, registry=None):
    """The run's context (created here unless passed in), its brain, and whether this call owns the context."""
    owned = context is None
    if owned:
        context = RunContext(settings, sink, brain.registry if brain is not None else registry)
    elif sink is not None and context.sink is not sink:
        raise ValueError("sink must match the run context")
    if brain is not None and brain.registry is not context.registry:
        raise ValueError("brain must use the run context's registry")
    context.claim()
    if brain is None:
        from .agents import StrandsBrain  # loads the model stack only when a real run starts
        brain = StrandsBrain(context)
    return context, brain, owned


def _plan(plan: Plan, breadth: int, next_id: int, previous: list[SubQuestion]) -> Plan:
    seen = {" ".join(sq.question.lower().split()) for sq in previous}
    questions = []
    for sq in plan.sub_questions:
        key = " ".join(sq.question.lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        questions.append(sq.model_copy(update={"id": next_id + len(questions), "search_hints": sq.search_hints[:4]}))
        if len(questions) == breadth:
            break
    return Plan(sub_questions=questions)


def _stats(ctx, brain, start, rounds, results):
    budget = ctx.budget
    usage = brain.usage
    return Stats(seconds=round(time.monotonic() - start, 1), rounds=rounds, sub_questions=len(results),
                 tool_calls=ctx.tool_calls, sources=len(ctx.registry),
                 input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0),
                 searches=budget.searches, fetches=budget.fetches, model_calls=budget.model_calls,
                 usage_incomplete=budget.usage_incomplete)


async def run_research(question: str, settings: Settings, sink: EventSink | None = None,
                       brain: Brain | None = None, *, context: RunContext | None = None, resume: Report | None = None,
                       answers: str | None = None, assumptions: list[str] | None = None) -> Report:
    """`answers`: the user's replies to clarifying questions (see run_clarify). `assumptions`: proceed on these.
    With neither, a clarification check runs and its assumptions are used; nothing is ever asked here.
    `resume`: a saved checkpoint. Its sources keep their IDs, sub-questions with findings are kept, the rest of
    its last round is re-run, then the run continues normally. Budgets and timers apply to the resumed part."""
    registry = None
    if resume is not None:
        if resume.status == "complete":
            raise ValueError("run is already complete; nothing to resume")
        if question and question != resume.question:
            raise ValueError("question does not match the checkpoint")
        question = resume.question
        registry = SourceRegistry.from_sources(resume.sources)
    if not question.strip() or len(question) > 20000:
        raise ValueError("question must contain 1-20000 characters")
    ctx, brain, owned = _dependencies(settings, sink, brain, context, registry)
    settings, registry = ctx.settings, ctx.registry
    if resume is not None and [(s.id, s.url) for s in registry.all()[:len(resume.sources)]] != [
            (s.id, SourceRegistry.normalize(s.url)) for s in resume.sources]:
        # Kept findings cite sources by ID; any other registry would silently re-attribute them.
        raise ValueError("the brain or context does not hold the checkpoint's sources; omit them to rebuild it")
    ctx.question = question
    start = time.monotonic()
    sem = asyncio.Semaphore(settings.parallel)
    all_sqs, results, warnings = [], [], []
    rounds = 0
    continuing = resume is not None and bool(resume.plan)
    if resume is not None:
        ctx.assumptions, ctx.clarifications = list(resume.assumptions), resume.clarifications
    if answers and answers.strip():
        ctx.clarifications = answers.strip()[:4000]
    elif assumptions is not None:
        ctx.assumptions = list(assumptions)
    if continuing:
        ctx.run_id = resume.run_id or ctx.run_id
        ctx.assessment = resume.assessment
        all_sqs = list(resume.plan)
        results = [r for r in resume.results if r.findings]
        rounds = max(1, resume.stats.rounds)

    def snapshot(status="partial", text=None):
        ordered = sorted(results, key=lambda r: r.sub_question.id)
        md, _ = finish_markdown(text or fallback_report(question, ordered), registry, cited_ids(ordered))
        return Report(question=question, plan=list(all_sqs), results=ordered,
                      report_md=frame_report(md, ctx.assumptions, ctx.clarifications),
                      assumptions=list(ctx.assumptions), clarifications=ctx.clarifications,
                      sources=registry.all(), stats=_stats(ctx, brain, start, rounds, results),
                      run_id=ctx.run_id, status=status, warnings=list(dict.fromkeys(warnings)),
                      assessment=ctx.assessment, review=ctx.review)

    def save(report):
        if settings.checkpoint_dir is None:
            return
        try:
            atomic_write(settings.checkpoint_dir / f"{ctx.run_id}.json", report.model_dump_json(indent=2))
        except OSError as exc:  # a checkpoint is a convenience; losing one must not lose the run
            message = f"Checkpoint not saved: {exc}"
            warnings.append(message)
            report.warnings = list(dict.fromkeys(report.warnings + [message]))

    async def checkpoint(status="partial"):
        report = snapshot(status)
        save(report)
        await ctx.emit(CheckpointReady(report=report))

    async def research_one(sq, round_no):
        async with sem:
            await ctx.emit(SubQuestionStarted(sq_id=sq.id, question=sq.question, round=round_no))
            try:
                findings, searches = await asyncio.wait_for(brain.research(sq, round_no), settings.researcher_timeout)
                # Re-checked here so that no brain, including a custom one, can bypass quote provenance.
                findings, issues = validate_findings(findings, registry, f"r{sq.id}")
                warnings.extend(issues)
                result = SubQuestionResult(sub_question=sq, round=round_no, findings=findings, searches=searches)
                done = SubQuestionDone(sq_id=sq.id, ok=True, facts=len(findings.facts), gaps=len(findings.gaps))
            except Exception as exc:
                message = (f"timed out after {settings.researcher_timeout:g}s" if isinstance(exc, TimeoutError)
                           else f"{type(exc).__name__}: {exc}")
                result = SubQuestionResult(sub_question=sq, round=round_no, error=message)
                warnings.append(f"Research {sq.id}: {message}")
                await ctx.emit(ErrorEvent(where=f"research:{sq.id}", message=message))
                done = SubQuestionDone(sq_id=sq.id, ok=False, error=message)
            results.append(result)
            await checkpoint()
            await ctx.emit(done)
            return result

    async def phase(call):
        return await asyncio.wait_for(call, settings.phase_timeout)

    try:
        try:
            async with asyncio.timeout(settings.run_timeout):
                if continuing:
                    done = {r.sub_question.id for r in results}
                    todo = [sq for sq in all_sqs if sq.id not in done]
                    await ctx.emit(PhaseStarted(phase="resume", detail=f"round {rounds}: {len(results)} kept, "
                                                f"{len(todo)} to research"))
                else:
                    if resume is None and not ctx.clarifications and assumptions is None:
                        await ctx.emit(PhaseStarted(phase="clarify", detail="choosing working assumptions"))
                        try:
                            ctx.assumptions = (await phase(brain.clarify(question))).assumptions
                        except Exception as exc:
                            warnings.append(f"Clarification check failed: {type(exc).__name__}")
                    await ctx.emit(PhaseStarted(phase="plan", detail=question))
                    plan = _plan(await phase(brain.plan(question, settings.breadth)), settings.breadth, 1, [])
                    if not plan.sub_questions:
                        raise RuntimeError("planner returned no usable questions")
                    todo = plan.sub_questions
                while continuing or (todo and rounds < settings.depth):
                    if not continuing:  # a resumed run finishes its recorded round first
                        rounds += 1
                        all_sqs.extend(todo)
                        await ctx.emit(PlanReady(plan=Plan(sub_questions=todo), round=rounds))
                    continuing = False
                    ctx.research_notes = rejection_notes(results, registry)
                    await ctx.emit(PhaseStarted(phase="research", detail=f"round {rounds}"))
                    tasks = [asyncio.create_task(research_one(sq, rounds)) for sq in todo]
                    try:
                        batch = await asyncio.gather(*tasks)
                    finally:
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                    gaps = list(dict.fromkeys(g for r in batch for g in
                        (r.findings.gaps if r.findings else [f"Unanswered: {r.sub_question.question} ({r.error})"])))
                    await ctx.emit(RoundDone(round=rounds, gaps=gaps))
                    todo = []
                    can_follow = (rounds < settings.depth and ctx.budget.searches < settings.max_searches
                                  and ctx.budget.fetches < settings.max_fetches)
                    next_id = len(all_sqs) + 1
                    await ctx.emit(PhaseStarted(phase="assess", detail="checking accumulated evidence and coverage"))
                    try:
                        assessment = await phase(brain.assess(question, results,
                                                              settings.breadth if can_follow else 0, next_id))
                    except Exception as exc:
                        warnings.append(f"Coverage assessment failed: {type(exc).__name__}")
                        ctx.assessment = Assessment(gaps=["Final coverage could not be assessed."])
                        break
                    ctx.assessment = assessment
                    if assessment.sufficient or not can_follow:
                        break
                    # Failed tasks may be retried; completed questions are not repeated verbatim.
                    completed = [r.sub_question for r in results if r.findings and r.findings.facts]
                    todo = _plan(Plan(sub_questions=assessment.followups), settings.breadth, next_id,
                                 completed).sub_questions
                if not cited_ids(results):
                    raise RuntimeError("research produced no verified evidence")
                await ctx.emit(PhaseStarted(phase="synthesize", detail=f"{len(results)} results"))
                text = await phase(brain.synthesize(question, sorted(results, key=lambda r: r.sub_question.id)))
                text = ensure_disagreements(text, ctx.assessment, results)
                _, issues = finish_markdown(text, registry, cited_ids(results))
                warnings.extend(issues)
                limitations = (any(r.findings and r.findings.gaps for r in results)
                               or (ctx.assessment and (ctx.assessment.gaps or ctx.assessment.contradictions)))
                report = snapshot("partial" if warnings or limitations else "complete", text)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}" if str(exc) else "run or phase deadline exceeded"
            warnings.append(message)
            await ctx.emit(ErrorEvent(where="run", message=message))
            report = snapshot("partial" if cited_ids(results) else "failed")
        save(report)
        await ctx.emit(ReportReady(report=report))
        return report
    except asyncio.CancelledError:
        warnings.append("Run cancelled; completed findings preserved")
        await checkpoint("cancelled")
        await ctx.emit(RunCancelled())
        raise
    finally:
        if owned:
            await ctx.aclose()


async def run_clarify(question: str, settings: Settings, sink: EventSink | None = None,
                      brain: Brain | None = None, *, force: bool = False) -> ClarificationCheck:
    """Questions worth asking before research, and the assumptions used if they go unanswered.
    `force` asks even when the question looks clear. Pass the user's replies to run_research(answers=...)."""
    if not question.strip() or len(question) > 20000:
        raise ValueError("question must contain 1-20000 characters")
    ctx, brain, owned = _dependencies(settings, sink, brain, None)
    ctx.question = question
    try:
        return await asyncio.wait_for(brain.clarify(question, force), ctx.settings.phase_timeout)
    finally:
        if owned:
            await ctx.aclose()


async def run_ask_result(question: str, settings: Settings, sink: EventSink | None = None,
                         brain: Brain | None = None, *, context: RunContext | None = None) -> Answer:
    if not question.strip() or len(question) > 20000:
        raise ValueError("question must contain 1-20000 characters")
    ctx, brain, owned = _dependencies(settings, sink, brain, context)
    ctx.question = question
    start = time.monotonic()
    warnings = []
    try:
        await ctx.emit(PhaseStarted(phase="ask", detail=question))
        try:
            async with asyncio.timeout(ctx.settings.run_timeout):
                text = await asyncio.wait_for(brain.ask(question), ctx.settings.researcher_timeout + ctx.settings.phase_timeout)
            results = getattr(brain, "last_results", [])
            for result in results:
                if result.findings:
                    result.findings, issues = validate_findings(result.findings, ctx.registry, f"r{result.sub_question.id}")
                    warnings.extend(issues)
            allowed = cited_ids(results)
            text, issues = finish_markdown(text, ctx.registry, allowed)
            warnings.extend(issues)
            status = "partial" if warnings or any(r.findings and r.findings.gaps for r in results) else "complete"
            if not allowed:
                status = "failed"
        except Exception as exc:
            warnings.append(f"{type(exc).__name__}: {exc}")
            results = getattr(brain, "last_results", [])
            text, _ = finish_markdown(fallback_report(question, results), ctx.registry, cited_ids(results))
            status = "partial" if cited_ids(results) else "failed"
        return Answer(question=question, answer_md=text, sources=ctx.registry.all(),
                      stats=_stats(ctx, brain, start, 1, results), run_id=ctx.run_id, status=status,
                      warnings=warnings, review=ctx.review)
    finally:
        if owned:
            await ctx.aclose()


def _slug(text: str, n: int = 48) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n].rstrip("-") or "report"


def save_report(report: Report, reports_dir: Path, as_json: bool = False) -> Path:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    stem = f"{stamp}-{uuid4().hex[:8]}-{_slug(report.question)}"
    path = Path(reports_dir) / f"{stem}.md"
    atomic_write(path, report.report_md + "\n")
    if as_json:
        atomic_write(path.with_suffix(".json"), report.model_dump_json(indent=2))
    return path
