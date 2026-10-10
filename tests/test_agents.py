from types import SimpleNamespace

import pytest

from inkmoth import agents
from inkmoth.budget import BudgetExceeded, RunBudget
from inkmoth.context import RunContext
from inkmoth.events import CollectingSink, TextDelta, TextReset
from inkmoth.evidence import validate_findings
from inkmoth.schemas import Evidence, Fact, Findings, SubQuestion
from inkmoth.sources import SourceRegistry


async def test_model_budget_accounts_metadata_even_without_agent_result(settings, monkeypatch):
    async def stream(self, *args, **kwargs):
        yield {'metadata':{'usage':{'inputTokens':40,'outputTokens':10}}}
    monkeypatch.setattr(agents.OpenAIModel, 'stream', stream)
    settings.max_model_calls = 1
    budget = RunBudget(settings)
    model = agents.make_model(settings, False, budget=budget)
    assert len([e async for e in model.stream([])]) == 1
    assert budget.usage == {'input_tokens':40,'output_tokens':10}
    with pytest.raises(BudgetExceeded):
        async for _ in model.stream([]): pass


def test_extra_body_only_when_configured(settings):
    assert "extra_body" not in agents.make_model(settings, True).config["params"]
    on, off = {"chat_template_kwargs": {"enable_thinking": True}}, {"chat_template_kwargs": {"enable_thinking": False}}
    settings.llm_extra_body_thinking, settings.llm_extra_body_no_thinking = on, off
    assert agents.make_model(settings, True).config["params"]["extra_body"] == on
    assert agents.make_model(settings, False).config["params"]["extra_body"] == off


async def test_missing_usage_is_explicit(settings, monkeypatch):
    async def stream(self, *args, **kwargs):
        if False: yield None
        raise RuntimeError('connection lost')
    monkeypatch.setattr(agents.OpenAIModel, 'stream', stream)
    budget = RunBudget(settings)
    with pytest.raises(RuntimeError):
        async for _ in agents.make_model(settings, False, budget=budget).stream([]): pass
    assert budget.usage_incomplete


async def test_research_always_wraps_after_turn_limit(settings, monkeypatch):
    calls = []
    class Agent:
        def __init__(self, **kwargs): self.messages = kwargs.get('messages', [])
        async def invoke_async(self, prompt, **kwargs):
            calls.append(prompt)
            if kwargs.get('structured_output_model'):
                return SimpleNamespace(structured_output=Findings(summary='final evidence', gaps=['not established']))
            self.messages.append({'role':'assistant','content':[{'text':'I will look that up.'}]})
            return SimpleNamespace(stop_reason='limit_turns')
    monkeypatch.setattr(agents, 'Agent', Agent)
    async with agents.StrandsBrain(RunContext(settings, CollectingSink(), SourceRegistry())) as brain:
        monkeypatch.setattr(brain, '_tools', lambda *args: ([],[]))
        findings, _ = await brain.research(SubQuestion(id=1, question='Q', why='w'), 1)
    assert len(calls) == 2 and findings.summary == ''
    assert 'limit' in findings.gaps[-1]


async def test_ask_renders_evidence_without_returning_preamble(settings, monkeypatch):
    registry = SourceRegistry()
    registry.add('https://example.com/source')
    quote = 'The documented memory size is 128 GB.'
    registry.record(1, quote, actor='r1', fetched=True)
    async with agents.StrandsBrain(RunContext(settings, CollectingSink(), registry)) as brain:
        async def research(*args):  # research() returns validated findings, which drops the model's summary
            return validate_findings(Findings(summary='I will look that up.', facts=[Fact(claim='128 GB',source_ids=[1],
                evidence=[Evidence(source_id=1,quote=quote)])]), registry, 'r1')[0], 1
        async def render(question, results, system, name):
            assert results[0].findings.summary == '128 GB'
            return '128 GB [1]'
        monkeypatch.setattr(brain, 'research', research)
        monkeypatch.setattr(brain, '_render', render)
        assert await brain.ask('Q') == '128 GB [1]'


async def test_synthesis_retry_resets_provisional_output(settings, monkeypatch):
    from strands.types.exceptions import MaxTokensReachedException
    settings.thinking_synthesizer = True
    settings.review_synthesis = False
    sink = CollectingSink()
    class Agent:
        count = 0
        def __init__(self, **kwargs):
            Agent.count += 1
            self.attempt = Agent.count
        async def stream_async(self, *args, **kwargs):
            yield {'data':'abandoned' if self.attempt == 1 else 'final'}
            if self.attempt == 1:
                raise MaxTokensReachedException('limit')
    monkeypatch.setattr(agents, 'Agent', Agent)
    async with agents.StrandsBrain(RunContext(settings, sink, SourceRegistry())) as brain:
        assert await brain.synthesize('Q', []) == 'final'
    assert [e.attempt for e in sink.events if isinstance(e, TextReset)] == [1,2]
    assert [(e.attempt,e.text) for e in sink.events if isinstance(e, TextDelta)] == [(1,'abandoned'),(2,'final')]


async def test_structured_fallback_keeps_original_task(settings):
    from strands.types.exceptions import MaxTokensReachedException
    class Fails:
        async def invoke_async(self, *args, **kwargs): raise MaxTokensReachedException('limit')
    class Fallback:
        async def invoke_async(self, prompt, **kwargs):
            assert prompt == 'original question and evidence'
            return SimpleNamespace(structured_output=Findings(summary='done'))
    async with agents.StrandsBrain(RunContext(settings, CollectingSink(), SourceRegistry())) as brain:
        result = await brain._structured(Fails(), 'original question and evidence', Findings, fallback=Fallback())
    assert result.summary == 'done'


async def test_research_prompt_carries_rejection_notes(settings, monkeypatch):
    prompts = []
    class Agent:
        def __init__(self, **kwargs): self.messages = []
        async def invoke_async(self, prompt, **kwargs):
            prompts.append(prompt)
            if kwargs.get('structured_output_model'):
                return SimpleNamespace(structured_output=Findings(summary=''))
            return SimpleNamespace(stop_reason='end_turn')
    monkeypatch.setattr(agents, 'Agent', Agent)
    async with agents.StrandsBrain(RunContext(settings, CollectingSink(), SourceRegistry())) as brain:
        monkeypatch.setattr(brain, '_tools', lambda *args: ([], []))
        brain.context.research_notes = 'Earlier claims citing [3] (1) were rejected'
        await brain.research(SubQuestion(id=2, question='Q', why='w'), 2)
    assert 'Notes from earlier rounds: Earlier claims citing [3] (1) were rejected' in prompts[0]


async def test_clarifications_and_assumptions_reach_every_phase_header(settings):
    async with agents.StrandsBrain(RunContext(settings, CollectingSink(), SourceRegistry())) as brain:
        brain.context.assumptions = ['Linux']
        assert 'Working assumptions (the user did not specify): ["Linux"]' in brain._dated('Q')
        brain.context.clarifications = 'Windows only'
        header = brain._dated('Q')
        assert 'The user clarified (authoritative): Windows only' in header and 'Working assumptions' not in header
