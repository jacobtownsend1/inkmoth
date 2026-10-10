import json
from types import SimpleNamespace

import pytest

from inkmoth import agents
from inkmoth.context import RunContext
from inkmoth.events import CollectingSink, TextDelta, TextReset
from inkmoth.evidence import fallback_report, validate_findings
from inkmoth.pipeline import run_research
from inkmoth.schemas import (Assessment, Coverage, Evidence, Fact, Findings, SubQuestion,
                                   SubQuestionResult, SynthesisReview)
from inkmoth.sources import SourceRegistry
from test_pipeline import FakeBrain


def record(registry, quote, actor='r1'):
    sid = registry.add('https://example.com/paper')
    registry.record(sid, quote, actor=actor, fetched=True)
    return sid


def fact(quote, claim='A supported claim', sid=1):
    return Fact(claim=claim, source_ids=[sid], evidence=[Evidence(source_id=sid, quote=quote)])


def result(findings):
    return SubQuestionResult(sub_question=SubQuestion(id=1, question='Original question', why='w'),
                             round=1, findings=findings)


def test_rejected_claims_are_diagnostics_never_downstream_evidence():
    registry = SourceRegistry()
    quote = 'The encoder is fixed during training.'
    record(registry, quote)
    raw = Findings(summary='invented summary', facts=[fact(quote), fact('POISONED invented quotation', 'POISONED claim')],
                   gaps=['Unverified claim omitted: POISONED claim', 'POISONED speculation in a gap'])
    validated, warnings = validate_findings(raw, registry, 'r1')
    assert len(validated.rejected) == 1
    assert validated.rejected[0].fact.evidence[0].quote == 'POISONED invented quotation'
    assert 'invalid/unseen quote indices [1]' in validated.rejected[0].reason
    assert 'POISONED' not in json.dumps(agents.evidence_payload([result(validated)]))
    assert 'POISONED' not in fallback_report('Q', [result(validated)])
    assert 'POISONED' not in ' '.join(warnings)
    again, _ = validate_findings(validated, registry, 'r1')
    assert len(again.rejected) == 1  # pipeline revalidation doesn't duplicate diagnostics


def test_pdf_ligatures_match_but_numbers_negation_and_boundaries_do_not():
    registry = SourceRegistry()
    record(registry, 'We keep the encoder ﬁxed with 100-word chunks and 21M documents. a² is squared.')
    assert registry.supports('r1', 1, 'We keep the encoder fixed with 100-word chunks')
    assert not registry.supports('r1', 1, 'We keep the encoder fixed with 1000-word chunks')
    assert not registry.supports('r1', 1, 'We keep the encoder not fixed')
    assert not registry.supports('r1', 1, 'a2 is squared.')
    assert not registry.supports('r2', 1, 'We keep the encoder fixed')
    passages = registry.passages('r1')
    passages[0]['passages'].clear()
    assert registry.passages('r1')[0]['passages']
    assert registry.passages('r2') == []


def test_coverage_cannot_use_missing_facts_or_gaps_as_evidence():
    results = [result(Findings(summary='s', facts=[fact('A quote with enough characters.')]))]
    assessment = Assessment(sufficient=True, coverage=[
        Coverage(question='Corpus size?', status='answered', fact_refs=['r1f9']),
        Coverage(question='Encoder status?', status='answered', fact_refs=[]),
        Coverage(question='Known detail?', status='answered', fact_refs=['r1f1'])])
    checked = agents.check_coverage(assessment, results)
    assert not checked.sufficient
    assert [c.status for c in checked.coverage] == ['unanswered', 'unanswered', 'answered']
    assert set(checked.gaps) == {'Corpus size?', 'Encoder status?'}
    assert not agents.check_coverage(Assessment(sufficient=True), results).sufficient


async def test_final_round_assessment_replaces_stale_metadata(settings):
    class Brain(FakeBrain):
        async def assess(self, question, results, max_n, next_id):
            self.calls.append(('assess', len(results), max_n))
            if max_n:
                return Assessment(followups=[SubQuestion(id=next_id, question='Follow up', why='w')],
                                  gaps=['Missing official recipe'])
            return Assessment(sufficient=True)
    settings.breadth, settings.depth = 1, 2
    brain = Brain(False)
    report = await run_research('Q', settings, brain=brain)
    assert [c for c in brain.calls if c[0] == 'assess'] == [('assess', 1, 1), ('assess', 2, 0)]
    assert report.assessment.sufficient and not report.assessment.gaps


async def test_assessment_failure_still_synthesizes_available_evidence(settings):
    class Brain(FakeBrain):
        async def assess(self, *args): raise RuntimeError('Unavailable')
    settings.depth = 1
    report = await run_research('Q', settings, brain=Brain(False))
    assert report.report_md.startswith('# Report')
    assert report.status == 'partial' and not report.assessment.sufficient


async def test_repair_uses_actor_passages_not_rejected_claims_or_conversation(settings, monkeypatch):
    registry = SourceRegistry()
    quote = 'The document encoder stays fixed during training.'
    record(registry, quote)
    record(registry, 'OTHER ACTOR ONLY PRIVATE PASSAGE', actor='r2')
    prompts = []
    class Agent:
        def __init__(self, **kwargs):
            assert 'messages' not in kwargs
        async def invoke_async(self, prompt, **kwargs):
            prompts.append(prompt)
            if not kwargs.get('structured_output_model'):
                return SimpleNamespace(stop_reason='end_turn')
            if len(prompts) == 2:
                return SimpleNamespace(structured_output=Findings(summary='s', facts=[
                    fact('POISONED wrong quote', 'POISONED claim')]))
            return SimpleNamespace(structured_output=Findings(summary='s', facts=[fact(quote)]))
    monkeypatch.setattr(agents, 'Agent', Agent)
    async with agents.StrandsBrain(RunContext(settings, CollectingSink(), registry)) as brain:
        brain.context.question = 'The actual original question'
        monkeypatch.setattr(brain, '_tools', lambda *args: ([], []))
        findings, _ = await brain.research(SubQuestion(id=1, question='Subquestion', why='w'), 1)
    assert len(prompts) == 3 and len(findings.facts) == 1 and len(findings.rejected) == 1
    assert all('The actual original question' in p for p in prompts)
    assert all('POISONED' not in p and 'OTHER ACTOR' not in p for p in prompts)
    assert quote in prompts[-1]


async def test_review_sees_quotes_and_replaces_provisional_draft(settings, monkeypatch):
    sink, registry = CollectingSink(), SourceRegistry()
    quote = 'If checkpointing runs separately, the main thread never blocks on sync.'
    record(registry, quote)
    results = [result(Findings(summary='s', facts=[fact(quote)], gaps=['POISONED gap']))]
    class Agent:
        def __init__(self, **kwargs):
            assert not kwargs.get('tools')
        async def stream_async(self, *args, **kwargs):
            yield {'data': 'The main thread never blocks [1].'}
        async def invoke_async(self, prompt, **kwargs):
            assert quote in prompt and 'POISONED' not in prompt and 'r1f1' not in prompt
            return SimpleNamespace(structured_output=SynthesisReview(
                report_md='If checkpointing runs separately, the main thread never blocks [1].',
                issues=['Restored the separate-checkpoint condition.']))
    monkeypatch.setattr(agents, 'Agent', Agent)
    async with agents.StrandsBrain(RunContext(settings, sink, registry)) as brain:
        brain.context.assessment = Assessment(sufficient=True, coverage=[Coverage(
            question='Does it block?', status='answered', fact_refs=['r1f1'])])
        text = await brain.synthesize('Does it block?', results)
        assert brain.context.assessment.coverage[0].fact_refs == ['r1f1']
        assert brain.context.review.issues
    assert text.startswith('If checkpointing')
    assert [e.attempt for e in sink.events if isinstance(e, TextReset)] == [1, 2]
    assert [e.text for e in sink.events if isinstance(e, TextDelta)][-1] == text


async def test_review_failure_does_not_release_unreviewed_draft(settings, monkeypatch):
    class Agent:
        def __init__(self, **kwargs): pass
        async def stream_async(self, *args, **kwargs): yield {'data': 'Unsupported draft'}
        async def invoke_async(self, *args, **kwargs): raise RuntimeError('review unavailable')
    monkeypatch.setattr(agents, 'Agent', Agent)
    async with agents.StrandsBrain(RunContext(settings, CollectingSink(), SourceRegistry())) as brain:
        with pytest.raises(RuntimeError, match='review unavailable'):
            await brain.synthesize('Q', [])
        assert brain.context.review is None


async def test_sdk_cancels_model_before_network_when_fetch_budget_exhausted(settings):
    # Real Strands hook dispatch, no mocked Agent or external request.
    async with RunContext(settings, CollectingSink()) as ctx:
        ctx.budget.fetches = settings.max_fetches
        hook = agents._ToolHooks('r1', ctx, ctx.budget)
        agent = agents.Agent(model=agents.make_model(settings, False, budget=ctx.budget),
                             hooks=[hook], callback_handler=None)
        response = await agent.invoke_async('Should not contact the model')
        assert response.stop_reason == 'end_turn'
        assert ctx.budget.model_calls == 0 and hook.stopped


async def test_optional_repair_timeout_preserves_accepted_facts(settings, monkeypatch):
    import asyncio
    settings.researcher_timeout = 2.05
    registry = SourceRegistry()
    quote = 'The document encoder stays fixed during training.'
    record(registry, quote)
    class Agent:
        calls = 0
        def __init__(self, **kwargs): pass
        async def invoke_async(self, prompt, **kwargs):
            if not kwargs.get('structured_output_model'):
                return SimpleNamespace(stop_reason='end_turn')
            Agent.calls += 1
            if Agent.calls > 1:
                await asyncio.sleep(10)
            return SimpleNamespace(structured_output=Findings(summary='s', facts=[
                fact(quote), fact('This nonexistent quote should fail.','Unsupported')]))
    monkeypatch.setattr(agents, 'Agent', Agent)
    async with agents.StrandsBrain(RunContext(settings, CollectingSink(), registry)) as brain:
        monkeypatch.setattr(brain, '_tools', lambda *args: ([], []))
        findings, _ = await asyncio.wait_for(brain.research(SubQuestion(id=1,question='Q',why='w'),1), 1)
    assert len(findings.facts) == 1 and len(findings.rejected) == 1


async def test_tool_phase_deadline_leaves_time_to_extract_collected_evidence(settings, monkeypatch):
    import asyncio
    settings.researcher_timeout = .2
    registry = SourceRegistry()
    quote = 'The document encoder stays fixed during training.'
    record(registry, quote)
    class Agent:
        def __init__(self, **kwargs): pass
        async def invoke_async(self, prompt, **kwargs):
            if not kwargs.get('structured_output_model'):
                await asyncio.sleep(10)
            return SimpleNamespace(structured_output=Findings(summary='s', facts=[fact(quote)]))
    monkeypatch.setattr(agents, 'Agent', Agent)
    async with agents.StrandsBrain(RunContext(settings, CollectingSink(), registry)) as brain:
        monkeypatch.setattr(brain, '_tools', lambda *args: ([], []))
        findings, _ = await asyncio.wait_for(brain.research(SubQuestion(id=1,question='Q',why='w'),1), .4)
    assert len(findings.facts) == 1 and 'limit' in findings.gaps[-1]
