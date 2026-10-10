"""Stable source IDs and an evidence ledger recording what each researcher saw."""
import threading
import unicodedata
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

from .schemas import Source


def normalize_quote(text: str) -> str:
    # PDF presentation ligatures are typography. Preserve punctuation, numbers,
    # mathematical symbols and word boundaries; do not use fuzzy matching/NFKC.
    text = unicodedata.normalize("NFC", text).translate(str.maketrans({
        "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}))
    return " ".join(text.split())


def host(url: str) -> str:
    """Independence unit for corroboration. Syndicated copies on different hosts still count as independent."""
    name = (urlsplit(url).hostname or "").lower()
    return name[4:] if name.startswith("www.") else name


# (start offset in the extracted document, text, document version); start None = position unknown
Span = tuple[int | None, str, str | None]


def merge_span(spans: list[Span], start: int | None, text: str, version: str | None = None) -> list[Span]:
    """Add a delivered span, joining it with spans of the same document version that it touches or overlaps
    (overlaps must also agree). Offsets from different versions of a page are never combined: touching spans
    from two versions could otherwise form a sentence neither version contains."""
    if start is None or version is None:
        return spans if any(t == text for _, t, _ in spans) else spans + [(None, text, None)]
    out, s, e = list(spans), start, start + len(text)
    i = 0
    while i < len(out):
        ss, tt, sv = out[i]
        se = ss + len(tt) if ss is not None else None
        if ss is not None and sv == version and ss <= e and s <= se:
            lo, hi = max(s, ss), min(e, se)
            if text[lo - s:hi - s] == tt[lo - ss:hi - ss]:
                text = (tt[:s - ss] if ss < s else "") + text + (tt[e - ss:] if se > e else "")
                s = min(s, ss)
                e = s + len(text)
                out.pop(i)
                i = 0  # the widened span may now touch an earlier one
                continue
        i += 1
    return sorted(out + [(s, text, version)], key=lambda x: (x[0] is None, x[0] or 0))


class SourceRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._ids: dict[str, int] = {}
        self._sources: list[Source] = []
        self._views: dict[tuple[str, int], list[Span]] = {}
        self._spans: dict[int, list[Span]] = {}

    @classmethod
    def from_sources(cls, sources: list[Source]) -> "SourceRegistry":
        """Rebuild a registry from a saved report, keeping source IDs so its citations stay valid."""
        registry = cls()
        for i, source in enumerate(sources, 1):
            if source.id != i:
                raise ValueError("saved sources must have IDs 1..n in order")
            registry._ids[cls.normalize(source.url)] = i
            registry._sources.append(source.model_copy(deep=True))
        return registry

    @staticmethod
    def normalize(url: str) -> str:
        p = urlsplit(url.strip())
        # Trailing slashes and query order can change the resource.
        return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path or "/", p.query, ""))

    def add(self, url: str, title: str = "", published: str | None = None) -> int:
        key = self.normalize(url)
        title = " ".join(title.split())[:300]
        with self._lock:
            sid = self._ids.get(key)
            if sid is None:
                sid = len(self._sources) + 1
                self._ids[key] = sid
                self._sources.append(Source(id=sid, url=key, title=title, published=published))
            elif title and not self._sources[sid - 1].title:
                self._sources[sid - 1].title = title
            return sid

    def mark_fetched(self, sid: int, content_hash: str):
        with self._lock:
            source = self._sources[sid - 1]
            source.fetched = True
            source.retrieved_at = datetime.now(timezone.utc).isoformat()
            source.content_hash = content_hash

    def record(self, sid: int, text: str, *, actor: str = "", fetched: bool = False, start: int | None = None,
               version: str | None = None):
        """Record text delivered to `actor`; `start` is its offset in the extracted document `version`, if known."""
        with self._lock:
            source = self._sources[sid - 1]
            if fetched:
                source.fetched = True
                source.retrieved_at = source.retrieved_at or datetime.now(timezone.utc).isoformat()
                spans = self._spans.get(sid) or [(None, t, None) for t in source.excerpts]
                self._spans[sid] = merge_span(spans, start, text, version)
                source.excerpts = [t for _, t, _ in self._spans[sid]]
                key = (actor, sid)
                self._views[key] = merge_span(self._views.get(key, []), start, text, version)

    def supports(self, actor: str, sid: int, quote: str) -> bool:
        normalized = normalize_quote(quote)
        with self._lock:
            return len(normalized) >= 12 and any(
                normalized in normalize_quote(text) for _, text, _ in self._views.get((actor, sid), []))

    def passages(self, actor: str) -> list[dict]:
        """Only fetched text actually delivered to this actor, never search snippets."""
        with self._lock:
            return [{"source_id": sid, "url": self._sources[sid - 1].url, "passages": [t for _, t, _ in spans]}
                    for (who, sid), spans in self._views.items() if who == actor]

    def __len__(self) -> int:
        return len(self._sources)

    def get(self, sid: int) -> Source | None:
        with self._lock:
            return self._sources[sid - 1].model_copy(deep=True) if 0 < sid <= len(self._sources) else None

    def all(self) -> list[Source]:
        with self._lock:
            return [s.model_copy(deep=True) for s in self._sources]

    def hosts(self, ids) -> set[str]:
        with self._lock:
            return {host(self._sources[i - 1].url) for i in ids if 0 < i <= len(self._sources)}

    def table(self, ids: set[int] | None = None) -> str:
        def dates(s):
            parts = [f"published {s.published[:10]}" if s.published else "",
                     f"retrieved {s.retrieved_at[:10]}" if s.retrieved_at else ""]
            return f" ({'; '.join(p for p in parts if p)})" if any(parts) else ""
        return "\n".join(f"[{s.id}] {s.title or '(untitled)'} — {s.url}{dates(s)}" for s in self.all()
                         if ids is None or s.id in ids)
