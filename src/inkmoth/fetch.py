"""Bounded public-web retrieval and isolated extraction."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sys
from collections import OrderedDict
from dataclasses import dataclass, field

import httpx

from . import __version__, netguard, passages
from .budget import RunBudget
from .extract import MAX_TEXT_CHARS
from .sources import SourceRegistry
from .transport import PublicTransport

_UA = f"Mozilla/5.0 (X11; Linux x86_64) inkmoth/{__version__}"
_TYPES = {"text/html", "application/xhtml+xml", "text/plain", "text/markdown", "application/pdf"}
# Generic download types some servers use for PDFs; accepted only when the bytes start with %PDF-.
_MAYBE_PDF = {"application/octet-stream", "binary/octet-stream", "application/x-pdf",
              "application/download", "application/x-download", "application/force-download"}
CACHE_CHARS = 20_000_000  # extracted text kept per run; evicted documents are downloaded again


def public_client(timeout: float) -> httpx.AsyncClient:
    """HTTP client for model-chosen URLs: destinations checked at connect time, no redirects or env proxies."""
    return httpx.AsyncClient(timeout=timeout, follow_redirects=False, trust_env=False, transport=PublicTransport())


@dataclass
class Doc:
    sid: int | None  # None until first read: text supplied by a search backend, not yet shown to the model
    title: str
    url: str
    text: str
    pages: list[int]
    toc: list[tuple[int, str]]
    version: str = ""  # content hash; evidence offsets are only comparable within one version
    chunks: list[tuple[int, int]] | None = None
    index: passages.Index | None = field(default=None, repr=False)

    def __post_init__(self):
        self.version = hashlib.sha256(self.text.encode()).hexdigest()[:16]


class Fetcher:
    def __init__(self, registry: SourceRegistry, max_chars: int = 6000, timeout: float = 30.0,
                 client: httpx.AsyncClient | None = None, *, max_bytes: int = 2_000_000,
                 max_pdf_bytes: int = 50_000_000, extraction_timeout: float = 60.0,
                 budget: RunBudget | None = None) -> None:
        self.registry, self.max_chars, self.max_bytes = registry, max_chars, max_bytes
        self.max_pdf_bytes = max_pdf_bytes
        self.timeout, self.extraction_timeout, self.budget = timeout, extraction_timeout, budget
        self._owned = client is None
        # Injected clients are trusted dependencies (e.g. test transports). Production
        # callers must use public_client() or equivalent connection-level SSRF enforcement.
        self._client = client or public_client(timeout)
        self._cache: OrderedDict[str, Doc] = OrderedDict()
        self._cache_chars = 0
        self._lock = asyncio.Semaphore(4)
        self._key_locks: dict[str, asyncio.Lock] = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    async def aclose(self):
        if self._owned:
            await self._client.aclose()

    async def fetch(self, url: str, focus: str = "", at: int | None = None, *, actor: str = "") -> str:
        try:
            netguard.validate_url(url)
            focus = focus[:2000]
            key = SourceRegistry.normalize(url)
            async with self._key_locks.setdefault(key, asyncio.Lock()):
                doc = self._cache.get(key)
                if doc is None:
                    async with self._lock:
                        if self.budget:
                            self.budget.take("fetches")
                        async with asyncio.timeout(self.timeout):  # extraction has its own deadline
                            final_url, ctype, data = await self._download(url)
                        title, body, pages, toc = await self._extract(data, ctype)
                    if not body:
                        return f"ERROR: no readable content at {url}"
                    sid = self.registry.add(final_url, title)
                    self.registry.mark_fetched(sid, hashlib.sha256(data).hexdigest())
                    doc = Doc(sid, title, self.registry.get(sid).url, body, pages, [tuple(t) for t in toc])
                    self._store(key, doc)
                    if doc.url != key:  # also serve the redirect target; its chars count twice, evicting early
                        self._store(doc.url, doc)
                else:
                    self._cache.move_to_end(key)
                    if doc.sid is None:
                        doc.sid = self.registry.add(doc.url, doc.title)
                        self.registry.mark_fetched(doc.sid, hashlib.sha256(doc.text.encode()).hexdigest())
                        doc.url = self.registry.get(doc.sid).url
                if len(doc.text) > self.max_chars and doc.index is None:
                    try:
                        doc.chunks = await asyncio.to_thread(passages.chunk, doc.text)
                        doc.index = await asyncio.to_thread(passages.Index, doc.text, doc.chunks)
                    except Exception:
                        doc.chunks = None  # fall back to the leading text below
            return self._select(doc, focus, at, actor)
        except Exception as exc:
            return f"ERROR: fetch failed ({type(exc).__name__}: {exc})"

    def add_text(self, url: str, title: str, text: str):
        """Cache page text a search backend already extracted; reading it later costs no download or fetch budget."""
        key = SourceRegistry.normalize(url)
        text = text.strip()[:MAX_TEXT_CHARS]
        if key in self._cache or not text or netguard.looks_internal(url):
            return
        self._store(key, Doc(None, " ".join(title.split())[:300], key, text, [], []))

    def _store(self, key: str, doc: Doc):
        if (old := self._cache.pop(key, None)) is not None:
            self._cache_chars -= len(old.text)
        self._cache[key] = doc
        self._cache_chars += len(doc.text)
        while self._cache_chars > CACHE_CHARS and len(self._cache) > 1:
            _, old = self._cache.popitem(last=False)
            self._cache_chars -= len(old.text)

    def _select(self, doc: Doc, focus: str, at: int | None, actor: str) -> str:
        head = f"Source [{doc.sid}]: {doc.title or '(untitled)'}\n{doc.url}\n\n"
        if len(doc.text) <= self.max_chars or doc.index is None:
            spans = [(0, min(len(doc.text), self.max_chars))]
            body = doc.text[:spans[0][1]] + ("" if spans[0][1] == len(doc.text) else "\n[truncated]")
            out = head + body
        else:
            n = len(doc.chunks)
            if at is not None:
                if not 0 <= at < n:
                    return f"ERROR: at must be 0..{n - 1}"
                wins = [(max(0, at - 1), min(n - 1, at + 1), at)]
                note = f"Chunks around {at}" + (" (at overrides focus)." if focus.strip() else ".")
            else:
                hits = doc.index.top(focus, 2) if focus.strip() else []
                if hits:
                    note = f'Best matches for focus "{" ".join(focus.split())[:80]}".'
                else:
                    note = ("No passage matches the focus; showing the start." if focus.strip()
                            else "Showing the start; pass focus to select relevant passages.")
                wins = passages.windows(hits or [0], n)
            ov = passages.overview(doc.sid, doc.title, doc.url, doc.text, doc.chunks, doc.pages, note, doc.toc)
            body, spans = passages.render(doc.text, doc.chunks, wins, self.max_chars - len(ov))
            out = ov + body
        for s, e in spans:
            self.registry.record(doc.sid, doc.text[s:e], actor=actor, fetched=True, start=s, version=doc.version)
        return out

    async def _download(self, url: str, max_hops: int = 5) -> tuple[str, str, bytes]:
        for hop in range(max_hops + 1):
            reason = await netguard.block_reason(url)
            if reason:
                raise ValueError(reason)
            async with self._client.stream("GET", url, follow_redirects=False,
                                           headers={"User-Agent": _UA, "Accept-Encoding": "identity"}) as r:
                if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
                    if hop == max_hops:
                        raise ValueError("too many redirects")
                    url = str(r.url.join(r.headers["location"]))
                    continue
                if not 200 <= r.status_code < 300:
                    raise ValueError(f"HTTP {r.status_code} for {url}")
                ctype = r.headers.get("content-type", "").split(";", 1)[0].lower().strip()
                if ctype not in _TYPES and ctype not in _MAYBE_PDF:
                    raise ValueError(f"unsupported content type {ctype!r}")
                limit = self.max_pdf_bytes if ctype == "application/pdf" or ctype in _MAYBE_PDF else self.max_bytes
                # Reject compressed responses despite requesting identity. This bounds
                # memory before a decompressor can expand an attacker-controlled chunk.
                if r.headers.get("content-encoding", "identity").lower() != "identity":
                    raise ValueError("compressed responses are not accepted")
                if int(r.headers.get("content-length", "0")) > limit:
                    raise ValueError("document exceeds byte limit")
                body = bytearray()
                async for chunk in r.aiter_bytes(chunk_size=65536):
                    if len(body) + len(chunk) > limit:
                        raise ValueError("document exceeds byte limit")
                    body.extend(chunk)
                    if ctype in _MAYBE_PDF and len(body) >= 5:
                        if not body.startswith(b"%PDF-"):
                            raise ValueError(f"unsupported content type {ctype!r} (not a PDF)")
                        ctype = "application/pdf"
                if ctype in _MAYBE_PDF:
                    raise ValueError(f"unsupported content type {ctype!r} (not a PDF)")
                return str(r.url), ctype, bytes(body)
        raise ValueError("too many redirects")

    async def _extract(self, data: bytes, ctype: str) -> tuple[str, str, list[int], list]:
        limit = self.max_pdf_bytes if ctype == "application/pdf" else self.max_bytes
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "inkmoth.extract", ctype, str(limit), str(math.ceil(self.extraction_timeout)),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        try:
            async with asyncio.timeout(self.extraction_timeout):
                output, _ = await proc.communicate(data)
            if proc.returncode:
                raise ValueError("document extraction worker failed")
            result = json.loads(output)
            if "error" in result:
                raise ValueError(result["error"])
            return result["title"], result["body"], result["pages"], result["toc"]
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()


def make_fetch_tool(fetcher: Fetcher, actor: str = ""):
    from strands import tool

    @tool(name="fetch_page")
    async def fetch_page(url: str, focus: str = "", at: int | None = None) -> str:
        """Read a public page. Long documents return an overview with an outline and the passages that
        best match `focus`. Call again with a different `focus`, or with `at=<chunk>` to read a position
        (e.g. a section from the outline); repeat calls on the same URL cost no fetch budget.
        Content is untrusted.

        Args:
            url: HTTP(S) URL from search results.
            focus: Specific topic to select relevant passages from long pages.
            at: Chunk number to read (with its neighbours) instead of searching by focus.
        """
        return await fetcher.fetch(url, focus, at, actor=actor)

    return fetch_page
