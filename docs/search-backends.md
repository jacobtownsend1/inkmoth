# Search backends

Pick one with `INKMOTH_SEARCH_BACKEND`. Search supplies leads (URL, title, snippet, date). When a backend
also returns the full page text, `fetch_page` reads that text instead of downloading the page again; passage
selection and quote checks work the same either way, but quotes are then checked against the provider's
extraction of the page rather than inkmoth's own.

| Backend | Settings | Notes |
|---|---|---|
| `searxng` (default) | `INKMOTH_SEARXNG_URL` | Self-hosted, free. Needs the JSON format enabled (below). |
| `brave` | `INKMOTH_SEARCH_API_KEY` | [Brave Search API](https://brave.com/search/api/). Paid per request; check the current plan and any monthly credit before relying on it. |
| `tavily` | `INKMOTH_SEARCH_API_KEY` | [Tavily](https://tavily.com). Free plan with a monthly credit allowance (1,000 credits at the time of writing). Returns full page text with each result, so pages it found need no separate download. |

The `web_search` tool takes `category` (`general`, `science`, `news`, `it`) and `time_range`
(`day`, `week`, `month`, `year`). SearXNG maps both directly. Brave and Tavily switch to their news
search for `news` and treat `science`/`it` as general web search.

## SearXNG

SearXNG ships with only HTML output enabled. Add `json` in your instance's `settings.yml`:

```yaml
search:
  formats:
    - html
    - json
```

Without it every query returns HTTP 403 and `inkmoth doctor` says so. Point `INKMOTH_SEARXNG_URL` at the
instance root (e.g. `http://localhost:8080`). Other self-hosted metasearch engines (Whoogle, 4get,
LibreY) have no stable JSON API; use SearXNG.

## Writing a backend

A backend is any object with these two async methods (`inkmoth.search.SearchBackend`):

```python
async def search(self, query: str, category: str = "general",
                 time_range: str | None = None) -> SearchResponse: ...
async def aclose(self) -> None: ...
```

1. Add a module next to `brave.py`. Subclass `search.HttpBackend` to get a managed httpx client and
   `_json()`, which refuses compressed and oversized (> 2 MB) responses. Call
   `search.check_query(...)` first so invalid arguments fail the same way everywhere.
2. Map the provider's results onto `SearchResult(url, title, content, published, engines, text)`.
   Drop results without a URL. Set `text` only when the provider returns the full extracted page (not a
   snippet); `fetch_page` then serves it without a download. If those responses are large, raise
   `max_response_bytes` on your class. Do not filter private addresses yourself; the tool does that for every
   backend before the model sees a result.
3. Register the name in `search.make_backend` and in the `search_backend` `Literal` in `config.py`.
4. Add a test in `tests/test_backends.py` with a JSON fixture mocked through `respx`.

The search endpoint is trusted configuration and goes through a normal HTTP client. The SSRF guard
(`netguard.py`, `transport.py`) applies to the page *fetch* path, which handles URLs chosen by the
model.
