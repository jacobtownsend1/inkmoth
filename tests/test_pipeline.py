import asyncio, json
from inkmoth.events import CollectingSink, SubQuestionDone
from inkmoth.pipeline import run_research, run_ask_result, save_report
from inkmoth.schemas import Assessment, ClarificationCheck, Evidence, Fact, Findings, Plan, SubQuestion, SubQuestionResult
from inkmoth.sources import SourceRegistry


class FakeBrain:
    def __init__(self, gaps_first_round=True, fail_ids=(), delay=0.05, hang_ids=()):
        self.registry = SourceRegistry()
        self.registry.add("https://example.com/evidence", "Evidence")
        self.usage = {"input_tokens": 10, "output_tokens": 5}
        self.gaps_first_round, self.fail_ids, self.delay, self.hang_ids = gaps_first_round, set(fail_ids), delay, set(hang_ids)
        self.active = self.max_active = 0
        self.calls = []

    async def plan(self, question, breadth):
        self.calls.append(("plan", breadth))
        return Plan(sub_questions=[SubQuestion(id=i, question=f"q{i}", why="w", search_hints=[]) for i in range(1, breadth + 1)])

    async def clarify(self, question, force=False):
        return ClarificationCheck()

    async def assess(self, question, results, max_n, next_id):
        gaps = tuple(g for r in results if r.findings for g in r.findings.gaps)
        self.calls.append(("assess", gaps, max_n, next_id))
        if gaps and max_n:
            return Assessment(gaps=list(gaps), followups=[SubQuestion(id=next_id, question="follow", why="w")])
        return Assessment(sufficient=True)

    async def research(self, sq, round):
        self.active += 1; self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(10 if sq.id in self.hang_ids else self.delay)
        self.active -= 1
        if sq.id in self.fail_ids:
            raise RuntimeError("boom")
        gaps = ["missing X"] if (round == 1 and self.gaps_first_round and sq.id == 1) else []
        self.registry.record(1, "A verifiable claim about the subject.", actor=f"r{sq.id}", fetched=True)
        return Findings(summary=f"s{sq.id}", facts=[Fact(claim="c", source_ids=[1], evidence=[Evidence(source_id=1, quote="A verifiable claim about the subject.")])], gaps=gaps), 2

    async def synthesize(self, question, results):
        self.calls.append(("synth", [r.sub_question.id for r in results]))
        return "# Report\n\nbody [1]\n\n## Sources\n[1] t — u"

    async def ask(self, question):
        sq = SubQuestion(id=1, question=question, why="w")
        findings, searches = await self.research(sq, 1)
        self.last_results = [SubQuestionResult(sub_question=sq, round=1, findings=findings, searches=searches)]
        return "answer [1]"


async def test_gap_round_and_events(settings):
    settings.breadth, settings.depth, settings.parallel = 3, 2, 2
    sink, brain = CollectingSink(), FakeBrain()
    report = await run_research("Q?", settings, sink, brain=brain)
    assert [c[0] for c in brain.calls] == ["plan", "assess", "assess", "synth"]
    assert brain.calls[1] == ("assess", ("missing X",), 3, 4) and brain.calls[2][2] == 0  # last round: no followups
    assert [r.sub_question.id for r in report.results] == [1, 2, 3, 4]
    assert report.results[3].round == 2 and report.stats.rounds == 2
    assert brain.max_active <= 2
    kinds = [type(e).__name__ for e in sink.events]
    assert kinds[0] == "PhaseStarted" and kinds[-1] == "ReportReady"
    assert kinds.count("PlanReady") == 2 and kinds.count("RoundDone") == 2 and kinds.count("SubQuestionDone") == 4
    assert report.stats.sub_questions == 4 and report.stats.tool_calls == 0 and report.stats.input_tokens == 10

async def test_depth_one_skips_followups(settings):
    settings.depth = 1
    brain = FakeBrain()
    report = await run_research("Q?", settings, CollectingSink(), brain=brain)
    assert [c[0] for c in brain.calls] == ["plan", "assess", "synth"] and brain.calls[1][2] == 0
    assert report.stats.rounds == 1

async def test_partial_failure_is_recorded(settings):
    settings.depth = 1
    brain = FakeBrain(gaps_first_round=False, fail_ids={2})
    sink = CollectingSink()
    report = await run_research("Q?", settings, sink, brain=brain)
    failed = [r for r in report.results if r.error]
    assert len(failed) == 1 and failed[0].sub_question.id == 2 and "boom" in failed[0].error
    assert any(isinstance(e, SubQuestionDone) and not e.ok for e in sink.events)
    assert brain.calls[-1] == ("synth", [1, 2, 3, 4])   # synthesizer still sees all, incl. failed

async def test_researcher_timeout_is_recorded(settings):
    settings.depth, settings.researcher_timeout = 1, 0.2
    brain = FakeBrain(gaps_first_round=False, hang_ids={3})
    report = await run_research("Q?", settings, CollectingSink(), brain=brain)
    timed_out = [r for r in report.results if r.error]
    assert len(timed_out) == 1 and timed_out[0].sub_question.id == 3 and "timed out" in timed_out[0].error

async def test_save_report(settings, tmp_path):
    settings.depth = 1
    report = await run_research("How fast is it?", settings, CollectingSink(), brain=FakeBrain(gaps_first_round=False))
    path = save_report(report, tmp_path, as_json=True)
    assert path.suffix == ".md" and path.name.endswith("-how-fast-is-it.md")
    assert path.read_text().startswith("# Report")
    data = json.loads(path.with_suffix(".json").read_text())
    assert data["question"] == "How fast is it?"

async def test_run_ask(settings):
    answer = await run_ask_result("Q?", settings, CollectingSink(), brain=FakeBrain(gaps_first_round=False))
    assert answer.answer_md.startswith("answer [1]") and answer.status == "complete"


async def test_resume_keeps_findings_and_reruns_only_the_rest(settings):
    settings.breadth, settings.depth, settings.parallel = 3, 1, 3
    first = FakeBrain(gaps_first_round=False, fail_ids={2})
    partial = await run_research("Q?", settings, brain=first)
    assert partial.status == "partial" and [r.sub_question.id for r in partial.results if r.error] == [2]
    saved = partial.model_validate_json(partial.model_dump_json())  # as read back from a checkpoint file

    second = FakeBrain(gaps_first_round=False)
    second.registry = SourceRegistry.from_sources(saved.sources)
    second.research_ids = []
    original = second.research
    async def research(sq, round):
        second.research_ids.append((sq.id, round))
        return await original(sq, round)
    second.research = research
    report = await run_research("", settings, brain=second, resume=saved)
    assert second.research_ids == [(2, 1)]  # only the failed sub-question, in its recorded round
    assert [c[0] for c in second.calls] == ["assess", "synth"]  # no re-planning
    assert [r.sub_question.id for r in report.results] == [1, 2, 3] and not any(r.error for r in report.results)
    assert report.run_id == saved.run_id and report.question == "Q?" and report.stats.rounds == 1


async def test_resume_refuses_complete_runs_and_other_questions(settings):
    import pytest
    from inkmoth.schemas import Report, Stats
    done = Report(question="Q?", plan=[], results=[], report_md="", sources=[], stats=Stats(), status="complete")
    with pytest.raises(ValueError, match="already complete"):
        await run_research("", settings, brain=FakeBrain(), resume=done)
    with pytest.raises(ValueError, match="does not match"):
        await run_research("Other?", settings, brain=FakeBrain(), resume=done.model_copy(update={"status": "partial"}))


def test_registry_rebuilt_from_saved_sources_keeps_ids_and_excerpts():
    from inkmoth.schemas import Source
    import pytest
    reg = SourceRegistry.from_sources([Source(id=1, url="https://a.test/", excerpts=["seen text here ok"]),
                                       Source(id=2, url="https://b.test/")])
    assert reg.add("https://a.test/") == 1 and reg.add("https://c.test/") == 3
    reg.record(1, " more", actor="r9", fetched=True, start=None)
    assert reg.get(1).excerpts == ["seen text here ok", " more"]
    with pytest.raises(ValueError):
        SourceRegistry.from_sources([Source(id=2, url="https://b.test/")])


async def test_contradictions_reach_the_report(settings):
    from inkmoth.agents import check_coverage
    from inkmoth.schemas import Assessment, Contradiction, Coverage
    settings.breadth, settings.depth = 2, 1

    class Assessing(FakeBrain):
        async def research(self, sq, round):  # the second sub-question cites a different site
            findings, n = await super().research(sq, round)
            if sq.id == 2:
                sid = self.registry.add("https://other.example/b", "B")
                self.registry.record(sid, "A verifiable claim about the subject.", actor="r2", fetched=True)
                findings.facts[0].source_ids = [sid]
                findings.facts[0].evidence[0].source_id = sid
            return findings, n

        async def assess(self, question, results, max_n, next_id):
            return check_coverage(Assessment(sufficient=True,
                coverage=[Coverage(question="q", status="answered", fact_refs=["r1f1"])],
                contradictions=[Contradiction(summary="Sources disagree", fact_refs=["r1f1", "r2f1"])]), results)

    report = await run_research("Q?", settings, brain=Assessing(gaps_first_round=False))
    body = report.report_md.split("## Sources")[0]
    assert "## Disagreements" in body and "- Sources disagree: c [1]†; c [2]†" in body


class Clarifying(FakeBrain):
    def __init__(self, **kw):
        super().__init__(gaps_first_round=False, **kw)
    async def clarify(self, question, force=False):
        from inkmoth.schemas import ClarificationCheck
        self.calls.append(("clarify", force))
        return ClarificationCheck(needs_clarification=True, questions=["Which OS?", "a", "b", "c"],
                                  assumptions=["Linux", "latest stable release"])


async def test_unanswered_runs_assume_and_say_so(settings):
    settings.breadth, settings.depth = 1, 1
    brain = Clarifying()
    report = await run_research("Q?", settings, brain=brain)
    assert brain.calls[0] == ("clarify", False)
    assert report.assumptions == ["Linux", "latest stable release"]
    assert report.report_md.startswith("# Report\n\n**Assumptions:** Linux; latest stable release\n")


async def test_answers_skip_the_check_and_survive_resume(settings):
    settings.breadth, settings.depth = 2, 1
    brain = Clarifying(fail_ids={2})
    report = await run_research("Q?", settings, brain=brain, answers="  Windows 11 only ")
    assert ("clarify", False) not in brain.calls
    assert report.clarifications == "Windows 11 only" and "**Clarified by the user:** Windows 11 only" in report.report_md
    again = Clarifying()
    again.registry = SourceRegistry.from_sources(report.sources)
    resumed = await run_research("", settings, brain=again, resume=report)
    assert resumed.clarifications == "Windows 11 only" and not any(c[0] == "clarify" for c in again.calls)


async def test_run_clarify_caps_questions_and_forwards_force(settings):
    from inkmoth.pipeline import run_clarify
    brain = Clarifying()
    check = await run_clarify("Q?", settings, brain=brain, force=True)
    assert check.questions == ["Which OS?", "a", "b"] and brain.calls == [("clarify", True)]


async def test_resume_refuses_a_registry_that_would_reattribute_citations(settings):
    import pytest
    settings.breadth, settings.depth = 2, 1
    partial = await run_research("Q?", settings, brain=FakeBrain(gaps_first_round=False, fail_ids={2}))
    other = FakeBrain(gaps_first_round=False)
    other.registry = SourceRegistry()
    other.registry.add("https://new.example/first")  # [1] would now mean a different page
    with pytest.raises(ValueError, match="checkpoint's sources"):
        await run_research("", settings, brain=other, resume=partial)
