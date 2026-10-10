"""Brave Search API backend (https://api-dashboard.search.brave.com/app/documentation)."""
from __future__ import annotations

import html
import re

import httpx

from .search import HttpBackend, SearchResponse, SearchResult, check_query

FRESHNESS = {"day": "pd", "week": "pw", "month": "pm", "year": "py"}


def _text(s: str | None) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s or ""))  # descriptions carry <strong> highlights


class BraveBackend(HttpBackend):
    base_url = "https://api.search.brave.com/res/v1"

    def __init__(self, api_key: str, timeout: float = 20.0, client: httpx.AsyncClient | None = None) -> None:
        super().__init__(timeout, client)
        self.api_key = api_key

    async def search(self, query: str, category: str = "general", time_range: str | None = None) -> SearchResponse:
        check_query(query, category, time_range)
        news = category == "news"  # science/it have no Brave equivalent and use plain web search
        params = {"q": query, "count": 20}
        if time_range:
            params["freshness"] = FRESHNESS[time_range]
        d = await self._json("GET", f"{self.base_url}/{'news' if news else 'web'}/search", params=params,
                             headers={"X-Subscription-Token": self.api_key, "Accept": "application/json"})
        items = d.get("results", []) if news else (d.get("web") or {}).get("results", [])
        return SearchResponse(query=query, results=[
            SearchResult(url=x["url"], title=_text(x.get("title")), content=_text(x.get("description")),
                         published=x.get("page_age"))
            for x in items if x.get("url")])
