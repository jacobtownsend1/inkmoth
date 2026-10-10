"""Run a small repeatable evaluation; the rubric is never included in model inputs.

Usage: uv run python evals/run.py --out reports/eval-name [--case python313]
Outputs raw reports, per-case events and a machine-readable metrics summary.
Quality is assessed separately against primary sources, not keyword matches.
"""

import argparse
import asyncio
import hashlib
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from pydantic import BaseModel
from inkmoth import Settings, run_clarify, run_research
from inkmoth.events import CheckpointReady, ErrorEvent, PhaseStarted, SubQuestionDone


class Sink:
    def __init__(self, case_id, stream):
        self.case_id, self.stream = case_id, stream

    async def emit(self, event):
        # Checkpoints already persist to disk; avoid duplicating document bodies in the event log.
        if not isinstance(event, CheckpointReady):
            data = asdict(event)
            data["type"] = type(event).__name__
            self.stream.write(
                json.dumps(data, default=lambda v: v.model_dump(mode="json") if isinstance(v, BaseModel) else str(v))
                + "\n"
            )
            self.stream.flush()
        if isinstance(event, PhaseStarted):
            print(self.case_id, event.phase, event.detail[:90], flush=True)
        elif isinstance(event, SubQuestionDone):
            print(
                self.case_id,
                "research",
                event.sq_id,
                "ok=",
                event.ok,
                "facts=",
                event.facts,
                "gaps=",
                event.gaps,
                flush=True,
            )
        elif isinstance(event, ErrorEvent):
            print(self.case_id, "ERROR", event.where, event.message[:200], flush=True)


async def main(args):
    cases = json.loads(Path(__file__).with_name("cases.json").read_text())
    if args.case:
        cases = [c for c in cases if c["id"] in args.case]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        _env_file=None,
        breadth=2,
        depth=2,
        parallel=2,
        researcher_searches=4,
        researcher_turns=8,
        max_searches=12,
        max_fetches=16,
        max_model_calls=60,
        researcher_timeout=180,
        phase_timeout=120,
        run_timeout=420,
        max_tokens=6000,
        synth_max_tokens=4000,
        thinking_planner=False,
        thinking_researcher=False,
        thinking_synthesizer=False,
    )
    src = Path(__file__).resolve().parents[1] / "src" / "inkmoth"
    fingerprint = hashlib.sha256(
        b"".join(p.name.encode() + p.read_bytes() for p in sorted(src.glob("*.py")))
    ).hexdigest()
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": fingerprint,
        "python": platform.python_version(),
        "strands": version("strands-agents"),
        "settings": settings.model_dump(mode="json", exclude={"llm_api_key", "search_api_key"}),
        "cases": cases,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    results = []
    for case in cases:
        case_dir = out / case["id"]
        case_dir.mkdir(exist_ok=True)
        settings.checkpoint_dir = case_dir / "checkpoints"
        # The clarify check runs for every case; cases with clarify_expected score whether it asked.
        check = await run_clarify(case["question"], settings)
        (case_dir / "clarification.json").write_text(check.model_dump_json(indent=2))
        clarify = {
            "clarify_asked": check.needs_clarification,
            **(
                {"clarify_ok": check.needs_clarification == case["clarify_expected"]}
                if "clarify_expected" in case
                else {}
            ),
        }
        if case.get("kind") == "clarify_only":
            results.append({"id": case["id"], **clarify})
            (out / "metrics.json").write_text(json.dumps(results, indent=2))
            print("DONE", json.dumps(results[-1]), flush=True)
            continue
        with (case_dir / "events.jsonl").open("w") as stream:
            report = await run_research(
                case["question"], settings, Sink(case["id"], stream), assumptions=check.assumptions
            )
        (case_dir / "report.json").write_text(report.model_dump_json(indent=2))
        (case_dir / "report.md").write_text(report.report_md + "\n")
        facts = [f for r in report.results if r.findings for f in r.findings.facts]
        metrics = {
            "id": case["id"],
            "status": report.status,
            **clarify,
            **report.stats.model_dump(),
            "facts": len(facts),
            "fetched_sources": sum(s.fetched for s in report.sources),
            "warning_count": len(report.warnings),
            "disagreements": len(report.assessment.contradictions) if report.assessment else 0,
            "single_site_marks": report.report_md.split("## Sources")[0].count("†")
            - ("† Single-site evidence" in report.report_md),
            "failed_researchers": sum(r.error is not None for r in report.results),
        }
        results.append(metrics)
        (out / "metrics.json").write_text(json.dumps(results, indent=2))
        print("DONE", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--case", action="append")
    asyncio.run(main(parser.parse_args()))
