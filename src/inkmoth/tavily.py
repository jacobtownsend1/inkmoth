"""Tavily search API backend (https://docs.tavily.com/documentation/api-reference/endpoint/search)."""
from __future__ import annotations

import httpx

from .search import HttpBackend, SearchResponse, SearchResult, check_query


class TavilyBackend(HttpBackend):
    url = "https://api.tavily.com/search"
    max_response_bytes = 20_000_000  # full page text for every result

    def __init__(self, api_key: str, timeout: float = 20.0, client: httpx.AsyncClient | None = None) -> None:
        super().__init__(timeout, client)
        self.api_key = api_key

    async def search(self, query: str, category: str = "general", time_range: str | None = None) -> SearchResponse:
        check_query(query, category, time_range)
        body = {"query": query, "topic": "news" if category == "news" else "general", "max_results": 20,
                "include_raw_content": "markdown"}  # free with basic search; fetch_page then needs no download
        if time_range:
            body["time_range"] = time_range
        d = await self._json("POST", self.url, json=body, headers={"Authorization": f"Bearer {self.api_key}"})
        return SearchResponse(query=query, results=[
            SearchResult(url=x["url"], title=x.get("title") or "", content=x.get("content") or "",
                         published=x.get("published_date"), text=x.get("raw_content"))
            for x in d.get("results", []) if x.get("url")])
