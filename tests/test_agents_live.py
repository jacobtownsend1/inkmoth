"""Small, bounded checks using the configured services and real public documents."""
import pytest

from inkmoth import run_ask_result, run_research
from inkmoth.events import CollectingSink, ToolCalled

pytestmark = pytest.mark.live


def bounded(settings):
    settings.breadth = settings.parallel = 1
    settings.depth = 2
    settings.researcher_searches = 2
    settings.researcher_turns = 6
    settings.max_searches = 4
    settings.max_fetches = 8
    settings.max_model_calls = 24
    settings.researcher_timeout = 100
    settings.phase_timeout = 80
    settings.run_timeout = 240
    settings.max_tokens = 4000
    settings.synth_max_tokens = 1800
    settings.thinking_planner = settings.thinking_researcher = settings.thinking_synthesizer = False
    return settings


async def test_plan_research_assess_synthesize(settings):
    sink = CollectingSink()
    report = await run_research("What is HTTPX's async HTTP client called, and how should it be closed? Use official documentation.",
                                bounded(settings), sink)
    assert report.status != 'failed', report.warnings
    assert 'AsyncClient' in report.report_md and 'aclose' in report.report_md
    assert '## Sources' in report.report_md and report.stats.output_tokens > 0
    assert report.assessment is not None
    assert any(r.findings and r.findings.facts for r in report.results)
    assert any(isinstance(e, ToolCalled) and e.tool == 'fetch_page' for e in sink.events)


async def test_ask_returns_cited_final_answer(settings):
    answer = await run_ask_result("What is HTTPX's async client class called? Use https://www.python-httpx.org/async/ and answer in one sentence.",
                                  bounded(settings))
    assert answer.status != 'failed', answer.warnings
    assert 'AsyncClient' in answer.answer_md and '## Sources' in answer.answer_md
    assert any(source.fetched for source in answer.sources)
