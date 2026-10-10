import json

import httpx
import pytest
import respx
from pydantic import ValidationError

from inkmoth.brave import BraveBackend
from inkmoth.config import Settings
from inkmoth.search import make_backend
from inkmoth.searx import SearxClient
from inkmoth.tavily import TavilyBackend

# shortcut: hand-written from the documented response shapes, not recorded; replace with real captures once keys exist
BRAVE_WEB = {"type": "search", "query": {"original": "vllm"}, "web": {"type": "search", "results": [
    {"title": "vLLM <strong>docs</strong>", "url": "https://docs.vllm.test/", "description": "Fast &amp; <strong>easy</strong>",
     "page_age": "2026-09-01T00:00:00", "age": "September 1, 2026"},
    {"title": "no url"}]}}
BRAVE_NEWS = {"type": "news", "results": [{"title": "News", "url": "https://news.test/a", "description": "d", "age": "2 days ago"}]}
TAVILY = {"query": "vllm", "results": [
    {"title": "vLLM", "url": "https://docs.vllm.test/", "content": "Fast and easy", "score": 0.9, "published_date": "2026-09-01",
     "raw_content": "# vLLM\n\nFull page text."}],
    "response_time": 0.5}


@respx.mock
async def test_brave_web_and_news():
    web = respx.get("https://api.search.brave.com/res/v1/web/search").mock(return_value=httpx.Response(200, json=BRAVE_WEB))
    news = respx.get("https://api.search.brave.com/res/v1/news/search").mock(return_value=httpx.Response(200, json=BRAVE_NEWS))
    async with BraveBackend("k") as b:
        resp = await b.search("vllm", time_range="week")
        assert [(r.title, r.content, r.published) for r in resp.results] == [("vLLM docs", "Fast & easy", "2026-09-01T00:00:00")]
        assert web.calls[0].request.headers["X-Subscription-Token"] == "k"
        assert web.calls[0].request.url.params["freshness"] == "pw"
        assert [r.url for r in (await b.search("x", category="news")).results] == ["https://news.test/a"]
        assert news.called


@respx.mock
async def test_tavily_maps_topic_and_time_range():
    route = respx.post("https://api.tavily.com/search").mock(return_value=httpx.Response(200, json=TAVILY))
    async with TavilyBackend("tvly-k") as t:
        resp = await t.search("vllm", category="news", time_range="month")
    assert [(r.url, r.content, r.published, r.text) for r in resp.results] == [
        ("https://docs.vllm.test/", "Fast and easy", "2026-09-01", "# vLLM\n\nFull page text.")]
    req = route.calls[0].request
    assert req.headers["Authorization"] == "Bearer tvly-k"
    assert json.loads(req.content) == {"query": "vllm", "topic": "news", "max_results": 20, "time_range": "month",
                                       "include_raw_content": "markdown"}


@respx.mock
async def test_hosted_backends_keep_a_byte_cap(monkeypatch):
    respx.get("https://api.search.brave.com/res/v1/web/search").mock(return_value=httpx.Response(200, content=b"x" * 2_000_001))
    with pytest.raises(ValueError, match="byte limit"):
        await BraveBackend("k").search("x")
    monkeypatch.setattr(TavilyBackend, "max_response_bytes", 1000)  # larger by default: it carries page text
    respx.post("https://api.tavily.com/search").mock(return_value=httpx.Response(200, content=b"x" * 1001))
    with pytest.raises(ValueError, match="byte limit"):
        await TavilyBackend("k").search("x")


@respx.mock  # any real HTTP request would fail: no routes are mocked
async def test_provider_page_text_is_read_without_downloading(settings):
    from inkmoth.budget import RunBudget
    from inkmoth.fetch import Fetcher
    from inkmoth.search import SearchBudget, SearchResponse, SearchResult, make_web_search_tool
    from inkmoth.sources import SourceRegistry
    page = "The retriever is trained jointly with the generator. " * 3

    class Provider:
        async def search(self, query, category="general", time_range=None):
            return SearchResponse(query=query, results=[
                SearchResult(url="https://paper.test/rag", title="RAG", content="snippet", text=page),
                SearchResult(url="http://192.168.1.5/admin", title="internal", text="secret")])

    reg, budget = SourceRegistry(), RunBudget(settings)
    fetcher = Fetcher(reg, budget=budget)
    cache = {}
    tool = make_web_search_tool(Provider(), reg, SearchBudget(3), 5, cache=cache, fetcher=fetcher)
    await tool.original_function("rag")
    assert all(r.text is None for resp in cache.values() for r in resp.results)  # only the fetcher keeps page text
    out = await fetcher.fetch("https://paper.test/rag", actor="r1")
    assert out == f"Source [1]: RAG\nhttps://paper.test/rag\n\n{page.strip()}"
    assert budget.fetches == 0 and reg.get(1).fetched and reg.get(1).content_hash
    assert reg.supports("r1", 1, "trained jointly with the generator")
    assert "secret" not in await fetcher.fetch("http://192.168.1.5/admin")


async def test_backends_reject_bad_arguments():
    with pytest.raises(ValueError):
        await BraveBackend("k").search("x", category="images")


def test_registry_and_key_validation():
    assert isinstance(make_backend(Settings(_env_file=None)), SearxClient)
    assert isinstance(make_backend(Settings(_env_file=None, search_backend="brave", search_api_key="k")), BraveBackend)
    assert isinstance(make_backend(Settings(_env_file=None, search_backend="tavily", search_api_key="k")), TavilyBackend)
    with pytest.raises(ValidationError, match="INKMOTH_SEARCH_API_KEY"):
        Settings(_env_file=None, search_backend="brave")
