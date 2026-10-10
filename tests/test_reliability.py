import asyncio
import json

import pytest
from pydantic import ValidationError

from inkmoth.config import Settings
from inkmoth.context import RunContext
from inkmoth.events import CollectingSink, CheckpointReady, RunCancelled
from inkmoth.pipeline import run_research, save_report
from inkmoth.schemas import Assessment, Plan, SubQuestion
from test_pipeline import FakeBrain


@pytest.mark.parametrize('field,value', [('parallel',0), ('breadth',0), ('depth',0), ('max_fetches',0),
                                        ('researcher_timeout',-1), ('run_timeout',float('inf'))])
def test_invalid_settings_rejected_on_creation_and_assignment(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field:value})
    with pytest.raises(ValidationError):
        setattr(Settings(_env_file=None), field, value)


async def test_plan_bounds_and_unique_ids_are_enforced(settings):
    class Oversized(FakeBrain):
        async def plan(self, question, breadth):
            return Plan(sub_questions=[SubQuestion(id=99,question=q,why='w') for q in ['a','a','b','c']])
    settings.breadth, settings.depth = 2, 1
    report = await run_research('Q', settings, brain=Oversized(False))
    assert [(q.id,q.question) for q in report.plan] == [(1,'a'),(2,'b')]


async def test_synthesis_failure_preserves_evidence_and_checkpoint(settings, tmp_path):
    class Fails(FakeBrain):
        async def synthesize(self, *args): raise RuntimeError('model unavailable')
    settings.depth, settings.checkpoint_dir = 1, tmp_path
    sink = CollectingSink()
    report = await run_research('Q', settings, sink, brain=Fails(False))
    assert report.status == 'partial' and len(report.results) == 4
    assert 'c [1]' in report.report_md and 'model unavailable' in ' '.join(report.warnings)
    saved = json.loads((tmp_path / f'{report.run_id}.json').read_text())
    assert saved['status'] == 'partial' and len(saved['results']) == 4
    checkpoints = [e for e in sink.events if isinstance(e, CheckpointReady)]
    assert len(checkpoints) == 4
    assert len({e.run_id for e in sink.events}) == 1
    assert [e.seq for e in sink.events] == sorted(e.seq for e in sink.events)


async def test_assessor_sees_successes_and_failures(settings):
    class Assessing(FakeBrain):
        async def assess(self, question, results, max_n, next_id):
            self.assessed = results[:]
            return Assessment(sufficient=True)
    brain = Assessing(False, fail_ids={2})
    report = await run_research('Q', settings, brain=brain)
    assert any(r.error for r in brain.assessed)
    assert any(r.findings and r.findings.facts for r in brain.assessed)
    assert report.stats.rounds == 1


async def test_planner_has_deadline(settings):
    class Hanging(FakeBrain):
        async def plan(self, *args): await asyncio.sleep(60)
    settings.phase_timeout = .01
    report = await run_research('Q', settings, brain=Hanging())
    assert report.status == 'failed' and report.warnings


async def test_run_deadline_preserves_completed_research(settings):
    settings.run_timeout = .15
    report = await run_research('Q', settings, brain=FakeBrain(False, delay=.01, hang_ids={3}))
    assert report.status == 'partial'
    assert {r.sub_question.id for r in report.results} == {1,2,4}


async def test_cancellation_preserves_completed_results_and_propagates(settings, tmp_path):
    settings.checkpoint_dir = tmp_path
    ready = asyncio.Event()
    class Sink(CollectingSink):
        async def emit(self, event):
            await super().emit(event)
            if isinstance(event, CheckpointReady): ready.set()
    sink = Sink()
    task = asyncio.create_task(run_research('Q', settings, sink, brain=FakeBrain(False, hang_ids={2,3,4})))
    await asyncio.wait_for(ready.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    assert any(isinstance(e, RunCancelled) for e in sink.events)
    saved = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert saved['status'] == 'cancelled' and len(saved['results']) == 1


async def test_context_is_single_use_and_borrowed(settings):
    brain = FakeBrain(False)
    async with RunContext(settings, registry=brain.registry) as ctx:
        await run_research('Q', settings, brain=brain, context=ctx)
        assert not ctx.closed
        with pytest.raises(ValueError, match='single-use'):
            await run_research('Q', settings, brain=brain, context=ctx)
    assert ctx.closed


async def test_mismatched_registries_rejected(settings):
    async with RunContext(settings) as ctx:
        with pytest.raises(ValueError, match='registry'):
            await run_research('Q', settings, brain=FakeBrain(), context=ctx)


async def test_unwritable_checkpoints_do_not_lose_the_run(settings, tmp_path):
    blocker = tmp_path / 'a-file'
    blocker.write_text('x')
    settings.depth, settings.checkpoint_dir = 1, blocker / 'checkpoints'
    report = await run_research('Q', settings, brain=FakeBrain(False))
    assert 'body [1]' in report.report_md and any('Checkpoint not saved' in w for w in report.warnings)


async def test_report_saves_do_not_collide(settings, tmp_path):
    settings.depth = 1
    report = await run_research('Q', settings, brain=FakeBrain(False))
    a, b = save_report(report, tmp_path), save_report(report, tmp_path)
    assert a != b and a.exists() and b.exists()


async def test_owned_clients_close_on_failure(settings, monkeypatch):
    contexts = []
    real_init = RunContext.__init__
    def init(ctx, *args, **kwargs):
        real_init(ctx, *args, **kwargs)
        contexts.append(ctx)
        _ = ctx.search
        _ = ctx.fetch
    monkeypatch.setattr(RunContext, '__init__', init)
    class Failing(FakeBrain):
        async def plan(self, *args): raise RuntimeError('failed')
    result = await run_research('Q', settings, brain=Failing())
    assert result.status == 'failed'
    assert contexts[0].closed
    assert contexts[0]._search._client.is_closed and contexts[0]._fetch._client.is_closed
