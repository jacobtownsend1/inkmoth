import httpx, respx, pytest
from inkmoth.searx import SearxClient
from inkmoth.search import SearchBudget, format_results, make_web_search_tool
from inkmoth.sources import SourceRegistry

PAYLOAD = {
    "query": "vllm",
    "results": [
        {"url": "https://a.test/1", "title": "A one", "content": "alpha " * 100, "engine": "duckduckgo", "engines": ["duckduckgo", "brave"], "score": 3.0, "publishedDate": "2026-01-02T00:00:00"},
        {"url": "https://b.test/2", "title": "B two", "content": "beta", "engine": "google", "engines": ["google"], "score": 1.0, "publishedDate": None},
        {"url": "https://a.test/1/", "title": "dup", "content": "dup", "engine": "bing", "engines": ["bing"], "score": 0.5},
    ],
    "unresponsive_engines": [["qwant", "CAPTCHA", True], ["wikidata", "timeout"]],
    "suggestions": ["vllm tensor parallel"],
}

@respx.mock
async def test_search_parses_and_sends_params():
    route = respx.get("http://searx.test/search").mock(return_value=httpx.Response(200, json=PAYLOAD))
    resp = await SearxClient("http://searx.test").search("vllm", category="it", time_range="year", page=2)
    q = dict(route.calls.last.request.url.params)
    assert q == {"q": "vllm", "format": "json", "categories": "it", "language": "en", "pageno": "2", "time_range": "year"}
    assert [r.url for r in resp.results] == ["https://a.test/1", "https://b.test/2", "https://a.test/1/"]
    assert resp.results[0].engines == ["duckduckgo", "brave"]
    assert resp.unresponsive == ["qwant", "wikidata"]

@respx.mock
async def test_search_http_error_raises():
    respx.get("http://searx.test/search").mock(return_value=httpx.Response(403))
    with pytest.raises(httpx.HTTPStatusError):
        await SearxClient("http://searx.test").search("x")

@respx.mock
async def test_format_dedups_registers_and_truncates():
    respx.get("http://searx.test/search").mock(return_value=httpx.Response(200, json=PAYLOAD))
    resp = await SearxClient("http://searx.test").search("vllm")
    reg = SourceRegistry()
    text = format_results(resp, reg, max_results=5, snippet_chars=40)
    assert text.startswith("[1] A one — https://a.test/1\n")
    assert "[2] B two — https://b.test/2" in text
    assert "[3] dup" in text  # trailing slash may be a different resource
    assert "alpha alpha" in text and len(text.splitlines()[1]) < 80
    assert "unresponsive: qwant, wikidata" in text
    assert reg.get(1).title == "A one"

def test_budget():
    b = SearchBudget(2)
    assert b.take() == 1 and b.take() == 2 and b.take() is None
    assert (b.used, b.limit) == (2, 2)

@respx.mock
async def test_tool_budget_line_and_cache():
    route = respx.get("http://searx.test/search").mock(return_value=httpx.Response(200, json=PAYLOAD))
    reg, budget, cache = SourceRegistry(), SearchBudget(2), {}
    tool = make_web_search_tool(SearxClient("http://searx.test"), reg, budget, max_results=5, cache=cache)
    fn = tool.original_function
    out1 = await fn(query="vllm")
    assert "(searches used: 1/2)" in out1
    out2 = await fn(query="vllm")                  # cached: no second HTTP call, budget still counts
    assert route.call_count == 1 and "(search budget exhausted" in out2
    out3 = await fn(query="other")
    assert "budget exhausted" in out3 and route.call_count == 1   # no search performed

@respx.mock
async def test_tool_reports_search_errors_as_text():
    respx.get("http://searx.test/search").mock(side_effect=httpx.ConnectError("down"))
    tool = make_web_search_tool(SearxClient("http://searx.test"), SourceRegistry(), SearchBudget(3), max_results=5)
    out = await tool.original_function(query="x")
    assert out.startswith("ERROR: search failed") and "(searches used: 1/3)" in out


@respx.mock
async def test_internal_results_are_dropped():
    payload = {"query": "x", "results": [
        {"url": "http://192.168.1.100:8000/v1/models", "title": "internal", "content": "c", "engines": ["x"]},
        {"url": "http://localhost/admin", "title": "local", "content": "c", "engines": ["x"]},
        {"url": "https://ok.test/a", "title": "ok", "content": "c", "engines": ["x"]},
    ], "unresponsive_engines": []}
    respx.get("http://searx.test/search").mock(return_value=httpx.Response(200, json=payload))
    resp = await SearxClient("http://searx.test").search("x")
    text = format_results(resp, SourceRegistry(), max_results=5)
    assert "[1] ok — https://ok.test/a" in text and "internal" not in text and "localhost" not in text
