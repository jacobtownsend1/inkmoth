from typer.testing import CliRunner
from inkmoth import cli
from inkmoth.schemas import Answer, Report, Stats, Source

runner = CliRunner()

def _fake_report(q):
    return Report(question=q, plan=[], results=[], report_md="# R\n\nbody\n\n## Sources\n[1] t — u", sources=[Source(id=1, url="u", title="t")], stats=Stats(seconds=1.0, rounds=1, sub_questions=2, sources=1))

def test_research_prints_and_saves(monkeypatch, tmp_path):
    async def fake_run(question, settings, sink, **kw):
        return _fake_report(question)
    monkeypatch.setattr(cli, "run_research", fake_run)
    monkeypatch.setenv("INKMOTH_REPORTS_DIR", str(tmp_path))
    r = runner.invoke(cli.app, ["research", "Why sky blue?", "--breadth", "2", "--json"])
    assert r.exit_code == 0, r.output
    assert "body" in r.output
    files = sorted(p.name for p in tmp_path.iterdir())
    assert any(f.endswith("-why-sky-blue.md") for f in files) and any(f.endswith(".json") for f in files)

def test_ask(monkeypatch):
    async def fake_ask(question, settings, sink, brain=None):
        return Answer(question=question, answer_md="answer [1]")
    monkeypatch.setattr(cli, "run_ask_result", fake_ask)
    r = runner.invoke(cli.app, ["ask", "Q?"])
    assert r.exit_code == 0 and "answer [1]" in r.stdout

def test_doctor_exit_code(monkeypatch):
    from inkmoth.doctor import Check
    async def fake_checks(settings):
        return [Check(name="LLM reachable", ok=True, detail="ok"), Check(name="SearXNG JSON", ok=False, detail="403")]
    monkeypatch.setattr(cli, "run_checks", fake_checks)
    r = runner.invoke(cli.app, ["doctor"])
    assert r.exit_code == 1 and "SearXNG JSON" in r.output


def test_machine_output_is_parseable_and_does_not_save(monkeypatch, tmp_path):
    import json
    from inkmoth.events import PhaseStarted
    async def fake_run(question, settings, sink, **kwargs):
        assert settings.checkpoint_dir is None
        await sink.emit(PhaseStarted(phase='plan',detail='[brackets]'))
        return _fake_report(question)
    monkeypatch.setattr(cli, 'run_research', fake_run)
    result = runner.invoke(cli.app, ['research','Q','--format','json','--events-json','--no-save','--out',str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)['question'] == 'Q'
    assert json.loads(result.stderr)['type'] == 'PhaseStarted'
    assert not list(tmp_path.iterdir())


def test_stdout_preserves_markdown(monkeypatch):
    async def fake_run(question, settings, sink, **kwargs): return _fake_report(question)
    monkeypatch.setattr(cli, 'run_research', fake_run)
    result = runner.invoke(cli.app, ['research','Q','--no-save'])
    assert result.stdout.startswith('# R\n')


def test_zero_parallel_rejected():
    result = runner.invoke(cli.app, ['research','Q','--parallel','0'])
    assert result.exit_code != 0


def test_resume_by_run_id_reads_the_default_checkpoint(monkeypatch, tmp_path):
    seen = {}
    async def fake_run(question, settings, sink, resume=None, **kw):
        seen["question"], seen["resume"] = question, resume
        return _fake_report(resume.question)
    monkeypatch.setattr(cli, "run_research", fake_run)
    monkeypatch.setenv("INKMOTH_REPORTS_DIR", str(tmp_path))
    saved = _fake_report("Saved Q?").model_copy(update={"run_id": "abc123", "status": "partial"})
    (tmp_path / ".checkpoints").mkdir()
    (tmp_path / ".checkpoints" / "abc123.json").write_text(saved.model_dump_json())
    r = runner.invoke(cli.app, ["research", "--resume", "abc123"])
    assert r.exit_code == 0, r.output
    assert seen["question"] == "" and seen["resume"].question == "Saved Q?"
    assert runner.invoke(cli.app, ["research", "--resume", "missing"]).exit_code == 2
    assert runner.invoke(cli.app, ["research"]).exit_code == 2


def test_preset_applies_and_explicit_flags_win(monkeypatch, tmp_path):
    seen = {}
    async def fake_run(question, settings, sink, resume=None, **kw):
        seen["s"] = settings
        return _fake_report(question)
    monkeypatch.setattr(cli, "run_research", fake_run)
    monkeypatch.setenv("INKMOTH_REPORTS_DIR", str(tmp_path))
    monkeypatch.setenv("INKMOTH_BREADTH", "7")
    r = runner.invoke(cli.app, ["research", "Q?", "--preset", "quick", "--depth", "2", "--no-save"])
    assert r.exit_code == 0, r.output
    s = seen["s"]
    assert (s.breadth, s.depth, s.max_searches, s.run_timeout) == (2, 2, 12, 600.0)


def _clarify_setup(monkeypatch, tmp_path, interactive=True):
    from inkmoth.schemas import ClarificationCheck
    seen = {"clarify": []}
    async def fake_clarify(question, settings, force=False):
        seen["clarify"].append(force)
        return ClarificationCheck(needs_clarification=True, questions=["Which OS?"], assumptions=["Linux"])
    async def fake_run(question, settings, sink, resume=None, answers=None, assumptions=None):
        seen["answers"], seen["assumptions"] = answers, assumptions
        return _fake_report(question)
    monkeypatch.setattr(cli, "run_clarify", fake_clarify)
    monkeypatch.setattr(cli, "run_research", fake_run)
    monkeypatch.setattr(cli, "_interactive", lambda: interactive)
    monkeypatch.setenv("INKMOTH_REPORTS_DIR", str(tmp_path))
    return seen


def test_interactive_clarification_round(monkeypatch, tmp_path):
    seen = _clarify_setup(monkeypatch, tmp_path)
    r = runner.invoke(cli.app, ["research", "Q?", "--no-save"], input="Windows 11\n")
    assert r.exit_code == 0, r.output
    assert seen["answers"] == "Windows 11" and seen["assumptions"] is None and seen["clarify"] == [False]
    r = runner.invoke(cli.app, ["research", "Q?", "--no-save"], input="proceed\n")
    assert seen["answers"] is None and seen["assumptions"] == ["Linux"]


def test_no_clarify_and_non_interactive_never_ask(monkeypatch, tmp_path):
    seen = _clarify_setup(monkeypatch, tmp_path)
    assert runner.invoke(cli.app, ["research", "Q?", "--no-save", "--no-clarify"]).exit_code == 0
    assert seen["clarify"] == [] and seen["answers"] is None and seen["assumptions"] is None
    seen = _clarify_setup(monkeypatch, tmp_path, interactive=False)
    assert runner.invoke(cli.app, ["research", "Q?", "--no-save"]).exit_code == 0 and seen["clarify"] == []
    assert runner.invoke(cli.app, ["research", "Q?", "--no-save", "--clarify"]).exit_code == 2


def test_search_errors_are_messages_and_need_no_model(monkeypatch):
    monkeypatch.delenv("INKMOTH_LLM_MODEL", raising=False)
    result = runner.invoke(cli.app, ["search", "x", "--category", "bogus"])
    assert result.exit_code == 2 and "invalid search category" in result.output and result.exception.__class__ is SystemExit


def test_interrupt_says_how_to_resume(monkeypatch, tmp_path):
    from inkmoth.events import PhaseStarted
    async def interrupted(question, settings, sink, **kwargs):
        await sink.emit(PhaseStarted(phase="plan", run_id="a" * 32))
        raise KeyboardInterrupt
    monkeypatch.setattr(cli, "run_research", interrupted)
    result = runner.invoke(cli.app, ["research", "Q", "--out", str(tmp_path)])
    assert result.exit_code == 130 and f"--resume {'a' * 32}" in result.output

