import httpx, respx
from inkmoth.doctor import run_checks

@respx.mock
async def test_doctor_all_ok(settings):
    settings.llm_base_url, settings.searxng_url = "http://llm.test/v1", "http://searx.test"
    respx.get("http://llm.test/v1/models").mock(return_value=httpx.Response(200, json={"data": [{"id": settings.llm_model}]}))
    respx.post("http://llm.test/v1/chat/completions").mock(return_value=httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [{"function": {"name": "web_search", "arguments": "{\"query\":\"x\"}"}}]}}]}))
    respx.get("http://searx.test/search").mock(return_value=httpx.Response(200, json={"query": "test", "results": [{"url": "https://a.test", "title": "t", "engine": "duckduckgo", "engines": ["duckduckgo", "wikipedia"]}] * 3, "unresponsive_engines": [["qwant", "CAPTCHA", True]]}))
    checks = await run_checks(settings)
    assert [c.ok for c in checks] == [True, True, True]
    assert "duckduckgo" in checks[2].detail and "qwant" in checks[2].detail

@respx.mock
async def test_doctor_reports_failures(settings):
    settings.llm_base_url, settings.searxng_url = "http://llm.test/v1", "http://searx.test"
    respx.get("http://llm.test/v1/models").mock(return_value=httpx.Response(200, json={"data": [{"id": "other"}]}))
    respx.post("http://llm.test/v1/chat/completions").mock(return_value=httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]}))
    respx.get("http://searx.test/search").mock(return_value=httpx.Response(403))
    checks = await run_checks(settings)
    assert [c.ok for c in checks] == [False, False, False]
    assert "403" in checks[2].detail
