"""SearXNG JSON API backend. The instance must have `json` in `search.formats` (off by default)."""
from __future__ import annotations

from typing import Any

import httpx

from .search import HttpBackend, SearchResponse, SearchResult, check_query


class SearxClient(HttpBackend):
    def __init__(self, base_url: str, timeout: float = 20.0, client: httpx.AsyncClient | None = None) -> None:
        super().__init__(timeout, client)
        self.base_url = base_url.rstrip("/")

    async def search(self, query: str, category: str = "general", time_range: str | None = None,
                     page: int = 1, language: str = "en") -> SearchResponse:
        check_query(query, category, time_range)
        if page < 1:
            raise ValueError("invalid page")
        params: dict[str, Any] = {"q": query, "format": "json", "categories": category,
                                  "language": language, "pageno": page}
        if time_range:
            params["time_range"] = time_range
        d = await self._json("GET", f"{self.base_url}/search", params=params)
        results = [
            SearchResult(
                url=x["url"], title=x.get("title") or "", content=x.get("content") or "",
                engines=list(x.get("engines") or ([x["engine"]] if x.get("engine") else [])),
                published=x.get("publishedDate"),
            )
            for x in d.get("results", []) if x.get("url")
        ]
        return SearchResponse(query=d.get("query", query), results=results,
                              unresponsive=[u[0] for u in d.get("unresponsive_engines", []) if u])
