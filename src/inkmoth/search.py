"""Search-backend seam: generic result models, the backend Protocol, and the `web_search` tool."""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Protocol

import httpx
from pydantic import BaseModel

from . import __version__, netguard
from .budget import RunBudget
from .sources import SourceRegistry

CATEGORIES = {"general", "science", "news", "it"}
TIME_RANGES = {None, "day", "week", "month", "year"}
MAX_RESPONSE_BYTES = 2_000_000


class SearchResult(BaseModel):
    url: str
    title: str = ""
    content: str = ""  # snippet shown to the model
    engines: list[str] = []  # upstream engine/source names, if the backend reports them
    published: str | None = None
    text: str | None = None  # full extracted page text, when the provider supplies it; fetch_page reuses it


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    unresponsive: list[str] = []


class SearchBackend(Protocol):
    async def search(self, query: str, category: str = "general",
                     time_range: str | None = None) -> SearchResponse: ...
    async def aclose(self) -> None: ...


def check_query(query: str, category: str, time_range: str | None) -> None:
    if not query.strip() or len(query) > 2000:
        raise ValueError("search query must contain 1-2000 characters")
    if category not in CATEGORIES:
        raise ValueError("invalid search category")
    if time_range not in TIME_RANGES:
        raise ValueError("invalid time range")


class HttpBackend:
    """Owns (or borrows) an httpx client and reads size-capped, uncompressed JSON."""

    max_response_bytes = MAX_RESPONSE_BYTES

    def __init__(self, timeout: float = 20.0, client: httpx.AsyncClient | None = None) -> None:
        self._owned = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout, trust_env=False,
                                                   headers={"User-Agent": f"inkmoth/{__version__}"})

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    async def aclose(self):
        if self._owned:
            await self._client.aclose()

    async def _json(self, method: str, url: str, headers: dict | None = None, **kwargs) -> Any:
        async with self._client.stream(method, url, headers={"Accept-Encoding": "identity", **(headers or {})},
                                       **kwargs) as r:
            r.raise_for_status()
            if r.headers.get("content-encoding", "identity").lower() != "identity":
                raise ValueError("compressed search responses are not accepted")
            data = bytearray()
            async for chunk in r.aiter_bytes(chunk_size=65536):
                if len(data) + len(chunk) > self.max_response_bytes:
                    raise ValueError("search response exceeds byte limit")
                data.extend(chunk)
        return json.loads(data)


def make_backend(settings) -> SearchBackend:
    name = settings.search_backend
    if name == "searxng":
        from .searx import SearxClient
        return SearxClient(settings.searxng_url)
    if name == "brave":
        from .brave import BraveBackend
        return BraveBackend(settings.search_api_key)
    if name == "tavily":
        from .tavily import TavilyBackend
        return TavilyBackend(settings.search_api_key)
    raise ValueError(f"unknown search backend {name!r}")


def format_results(resp: SearchResponse, registry: SourceRegistry, max_results: int, snippet_chars: int = 300) -> str:
    lines: list[str] = []
    seen: set[int] = set()
    for res in resp.results:
        if netguard.looks_internal(res.url):
            continue  # never show the model LAN/loopback URLs, whatever the engine returned
        sid = registry.add(res.url, res.title, res.published)
        if sid in seen:
            continue
        seen.add(sid)
        snippet = " ".join(res.content.split())
        if len(snippet) > snippet_chars:
            snippet = snippet[:snippet_chars].rsplit(" ", 1)[0] + "…"
        meta = ", ".join(x for x in [", ".join(res.engines), (res.published or "")[:10]] if x)
        source = registry.get(sid)
        lines.append(f"[{sid}] {source.title or '(untitled)'} — {source.url}")
        lines.append(f"    {snippet}" + (f" ({meta})" if meta else ""))
        if len(seen) >= max_results:
            break
    if not lines:
        lines.append("No results.")
    if resp.unresponsive:
        lines.append(f"(unresponsive: {', '.join(resp.unresponsive)})")
    return "\n".join(lines)


class SearchBudget:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0
        self._lock = threading.Lock()

    def take(self) -> int | None:
        with self._lock:
            if self.used >= self.limit:
                return None
            self.used += 1
            return self.used


def make_web_search_tool(client: SearchBackend, registry: SourceRegistry, budget: SearchBudget,
                         max_results: int, cache: dict | None = None, run_budget: RunBudget | None = None,
                         locks: dict | None = None, fetcher=None):
    """`fetcher` (a fetch.Fetcher) receives page text that results carry, so fetch_page needs no download."""
    from strands import tool
    cache = cache if cache is not None else {}
    locks = locks if locks is not None else {}

    async def _impl(query: str, category: str = "general", time_range: str | None = None) -> str:
        n = budget.take()
        if n is None:
            return (f"ERROR: search budget exhausted: {budget.used}/{budget.limit} used — do not search again; "
                    "write your findings now from what you already have)")
        key = (query.strip(), category, time_range or "")
        async with locks.setdefault(key, asyncio.Lock()):
            resp = cache.get(key)
            if resp is None:
                try:
                    if run_budget:
                        run_budget.take("searches")
                    resp = await client.search(query, category=category, time_range=time_range)
                except Exception as e:  # network / 4xx / 5xx: tell the model, never raise
                    return f"ERROR: search failed ({type(e).__name__}: {e}). (searches used: {n}/{budget.limit})"
                cache[key] = resp
                if fetcher is not None:
                    for res in resp.results:
                        if res.text:
                            fetcher.add_text(res.url, res.title, res.text)
                            res.text = None  # the fetcher's bounded cache owns it now; don't keep a copy per search
        body = format_results(resp, registry, max_results)
        tail = (f"(search budget exhausted: {n}/{budget.limit} used — answer now with what you have)"
                if n >= budget.limit else f"(searches used: {n}/{budget.limit})")
        return f"{body}\n\n{tail}"

    @tool(name="web_search")
    async def web_search(query: str, category: str = "general", time_range: str | None = None) -> str:
        """Search the web. Returns numbered results [n] with url and snippet; cite those ids.

        Args:
            query: Search query. Be specific; use quotes for exact phrases.
            category: One of general, science, news, it.
            time_range: Optional recency filter: day, week, month, year.
        """
        return await _impl(query, category, time_range)

    web_search.original_function = _impl  # tested directly; the decorator wraps _impl unchanged
    return web_search
