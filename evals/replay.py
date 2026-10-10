"""Re-evaluate writing against saved evidence, without new search/fetch requests.

Replay timings/usage measure synthesis and review only, not a complete research run.
"""

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

from inkmoth import RunContext, Settings
from inkmoth.agents import StrandsBrain
from inkmoth.evidence import cited_ids, finish_markdown
from inkmoth.schemas import Report
from inkmoth.sources import SourceRegistry
from run import Sink


async def main(args):
    seed, out = Path(args.seed), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((seed / "manifest.json").read_text())
    settings = Settings.model_validate(manifest["settings"])
    settings.checkpoint_dir = None
    src = Path(__file__).resolve().parents[1] / "src" / "inkmoth"
    fingerprint = hashlib.sha256(
        b"".join(p.name.encode() + p.read_bytes() for p in sorted(src.glob("*.py")))
    ).hexdigest()
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "synthesis-only replay",
                "seed": str(seed.resolve()),
                "source_sha256": fingerprint,
                "settings": settings.model_dump(mode="json", exclude={"llm_api_key", "search_api_key"}),
            },
            indent=2,
        )
    )
    for path in sorted(seed.glob("*/report.json")):
        report = Report.model_validate_json(path.read_text())
        registry = SourceRegistry()
        for source in report.sources:
            assert registry.add(source.url, source.title, source.published) == source.id
            for excerpt in source.excerpts:
                registry.record(source.id, excerpt, actor="replay", fetched=True)
        dest = out / path.parent.name
        dest.mkdir()
        started = time.monotonic()
        with (dest / "events.jsonl").open("w") as stream:
            async with RunContext(settings, Sink(path.parent.name, stream), registry) as ctx:
                ctx.question, ctx.assessment = report.question, report.assessment
                brain = StrandsBrain(ctx)
                text = await asyncio.wait_for(brain.synthesize(report.question, report.results), settings.phase_timeout)
                text, warnings = finish_markdown(text, registry, cited_ids(report.results))
                result = {
                    "question": report.question,
                    "seed_report": str(path.resolve()),
                    "report_md": text,
                    "review": ctx.review.model_dump() if ctx.review else None,
                    "warnings": warnings,
                    "seconds": round(time.monotonic() - started, 1),
                    "model_calls": ctx.budget.model_calls,
                    **ctx.budget.usage,
                    "usage_incomplete": ctx.budget.usage_incomplete,
                }
        (dest / "replay.json").write_text(json.dumps(result, indent=2))
        (dest / "report.md").write_text(text + "\n")
        print("DONE", path.parent.name, result["seconds"], result["model_calls"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True)
    parser.add_argument("--out", required=True)
    asyncio.run(main(parser.parse_args()))
