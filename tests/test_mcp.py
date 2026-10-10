import asyncio
import json

import pytest

pytest.importorskip("mcp")
from mcp import types  # noqa: E402
from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from inkmoth import mcp_server  # noqa: E402
from inkmoth.events import PhaseStarted  # noqa: E402
from inkmoth.schemas import ClarificationCheck, Report, Stats  # noqa: E402
from inkmoth.storage import atomic_write  # noqa: E402


def _report(question, run_id="", status="complete"):
    return Report(question=question, plan=[], results=[], report_md=f"# {question}\n\nbody [1]", sources=[],
                  stats=Stats(), run_id=run_id, status=status)


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    monkeypatch.setenv("INKMOTH_REPORTS_DIR", str(tmp_path))
    seen = {"clarify": 0}

    async def fake_clarify(question, settings, **kw):
        seen["clarify"] += 1
        return ClarificationCheck(needs_clarification=True, questions=["Which OS?"], assumptions=["Linux"])

    async def fake_research(question, settings, sink=None, *, context=None, answers=None, assumptions=None, **kw):
        seen.update(answers=answers, assumptions=assumptions, settings=settings)
        if sink is not None:
            await sink.emit(PhaseStarted(phase="plan", detail=question))
        run_id = context.run_id if context else "f" * 32
        report = _report(question, run_id)
        atomic_write(settings.checkpoint_dir / f"{run_id}.json", report.model_dump_json())
        return report

    monkeypatch.setattr(mcp_server, "run_clarify", fake_clarify)
    monkeypatch.setattr(mcp_server, "run_research", fake_research)
    return seen


def _text(result):
    return result.content[0].text


async def test_tools_are_listed():
    async with create_connected_server_and_client_session(mcp_server.server) as client:
        names = {t.name for t in (await client.list_tools()).tools}
    assert names == {"research", "start_research", "get_report", "ask", "clarify", "web_search", "fetch_page"}


async def test_get_report_rejects_paths_and_unknown_ids(fakes):
    async with create_connected_server_and_client_session(mcp_server.server) as client:
        for run_id in ("../../etc/passwd", "0" * 32):
            result = await client.call_tool("get_report", {"run_id": run_id})
            assert result.isError and "unknown run_id" in _text(result)


async def test_start_then_poll_the_checkpoint(fakes):
    async with create_connected_server_and_client_session(mcp_server.server) as client:
        started = json.loads(_text(await client.call_tool("start_research", {"question": "Q?", "preset": "quick"})))
        assert started["status"] == "running"
        for _ in range(50):
            got = json.loads(_text(await client.call_tool("get_report", {"run_id": started["run_id"]})))
            if got["status"] != "running":
                break
            await asyncio.sleep(0.02)
    assert got["status"] == "complete" and got["report_md"].startswith("# Q?")
    assert fakes["settings"].breadth == 2  # the quick preset


async def test_research_asks_through_hosts_that_support_elicitation(fakes):
    logs = []

    async def elicit(context, params):
        assert "Which OS?" in params.message and "Linux" in params.message
        return types.ElicitResult(action="accept", content={"answers": "Windows 11"})

    async def log(params):
        logs.append(params.data)

    async with create_connected_server_and_client_session(mcp_server.server, elicitation_callback=elicit,
                                                          logging_callback=log) as client:
        result = await client.call_tool("research", {"question": "Q?"})
    assert not result.isError and "status: complete" in _text(result)
    assert fakes["answers"] == "Windows 11" and fakes["assumptions"] is None
    assert any("plan: Q?" in str(entry) for entry in logs)


async def test_research_without_elicitation_proceeds_on_assumptions(fakes):
    async with create_connected_server_and_client_session(mcp_server.server) as client:
        result = await client.call_tool("research", {"question": "Q?"})
    assert not result.isError
    assert fakes["clarify"] == 0 and fakes["answers"] is None and fakes["assumptions"] is None


async def test_bad_preset_is_a_tool_error(fakes):
    async with create_connected_server_and_client_session(mcp_server.server) as client:
        result = await client.call_tool("research", {"question": "Q?", "preset": "huge"})
    assert result.isError and "preset must be one of" in _text(result)


async def test_failed_clarify_never_blocks_research(fakes, monkeypatch):
    async def broken(question, settings, **kw):
        raise TimeoutError("model slow")
    monkeypatch.setattr(mcp_server, "run_clarify", broken)

    async def elicit(context, params):
        raise AssertionError("must not ask without questions")

    async with create_connected_server_and_client_session(mcp_server.server, elicitation_callback=elicit) as client:
        result = await client.call_tool("research", {"question": "Q?"})
    assert not result.isError and fakes["assumptions"] == []  # [] also stops the pipeline re-running the check


async def test_failed_and_cancelled_background_runs_are_reported_and_released(fakes, monkeypatch):
    async def boom(question, settings, **kw):
        raise RuntimeError("model down")
    monkeypatch.setattr(mcp_server, "run_research", boom)
    async with create_connected_server_and_client_session(mcp_server.server) as client:
        run_id = json.loads(_text(await client.call_tool("start_research", {"question": "Q?"})))["run_id"]
        for _ in range(50):
            got = json.loads(_text(await client.call_tool("get_report", {"run_id": run_id})))
            if got["status"] != "running":
                break
            await asyncio.sleep(0.02)
    assert got == {"run_id": run_id, "status": "failed", "error": "RuntimeError: model down"}
    assert run_id not in mcp_server._tasks
    task = asyncio.get_running_loop().create_future()
    task.cancel()
    mcp_server._finished("c" * 32, task)
    assert mcp_server._failed["c" * 32] == "cancelled"


@pytest.fixture
def web(monkeypatch):
    from inkmoth.search import SearchResponse, SearchResult

    class Backend:
        async def search(self, query, category="general", time_range=None):
            return SearchResponse(query=query, results=[
                SearchResult(url="https://docs.test/a", title="A", content="snippet a"),
                SearchResult(url="http://10.0.0.5/admin", title="internal", content="x")])
        async def aclose(self):
            pass

    monkeypatch.setattr(mcp_server, "make_backend", lambda settings: Backend())
    monkeypatch.setattr(mcp_server, "_web", None)
    monkeypatch.delenv("INKMOTH_LLM_MODEL", raising=False)  # the web tools must work without a model


async def test_web_search_and_fetch_page_without_a_model(web):
    import httpx
    import respx
    page = "Intro. " + "Filler sentence about nothing. " * 400 + "The answer is forty-two."
    with respx.mock:
        route = respx.get("https://docs.test/a").mock(return_value=httpx.Response(200, text=page, headers={"content-type": "text/plain"}))
        async with create_connected_server_and_client_session(mcp_server.server) as client:
            found = _text(await client.call_tool("web_search", {"query": "answer"}))
            assert "[1] A — https://docs.test/a" in found and "10.0.0.5" not in found
            first = _text(await client.call_tool("fetch_page", {"url": "https://docs.test/a", "focus": "answer forty-two"}))
            assert "The answer is forty-two." in first and 'Best matches for focus "answer forty-two"' in first
            again = _text(await client.call_tool("fetch_page", {"url": "https://docs.test/a", "at": 0}))
            assert "--- chunks 0-1 of" in again and route.call_count == 1
            refused = _text(await client.call_tool("fetch_page", {"url": "http://127.0.0.1:8000/v1/models"}))
            assert refused.startswith("ERROR")


async def test_web_tools_start_over_after_many_sources(web, monkeypatch):
    monkeypatch.setattr(mcp_server._Web, "MAX_SOURCES", 1)
    web = mcp_server._web_tools()
    first = web.fetcher
    for i in range(3):
        first.registry.add(f"https://s{i}.test/")
    assert mcp_server._web_tools().fetcher is not first and len(web.fetcher.registry) == 0
    assert not web.client.is_closed  # shared: a call still using the old fetcher keeps working


async def test_web_search_keeps_provider_page_text(web, monkeypatch):
    from inkmoth.search import SearchResponse, SearchResult

    class Backend:
        async def search(self, query, category="general", time_range=None):
            return SearchResponse(query=query, results=[
                SearchResult(url="https://docs.test/t", title="T", content="s", text="Full provider text here.")])

    monkeypatch.setattr(mcp_server, "make_backend", lambda settings: Backend())
    async with create_connected_server_and_client_session(mcp_server.server) as client:
        await client.call_tool("web_search", {"query": "t"})
        page = _text(await client.call_tool("fetch_page", {"url": "https://docs.test/t"}))  # no HTTP mock: no download
    assert "Full provider text here." in page


def test_reports_default_to_a_state_dir_not_the_working_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("INKMOTH_REPORTS_DIR", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert mcp_server._settings().checkpoint_dir == tmp_path / "inkmoth" / ".checkpoints"
    monkeypatch.setenv("INKMOTH_REPORTS_DIR", "/srv/reports")
    assert mcp_server._settings().checkpoint_dir == mcp_server.Path("/srv/reports/.checkpoints")
