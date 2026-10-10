"""Pure passage selection over extracted text: sentence-aware chunks, BM25 ranking, windows, overview."""
from __future__ import annotations

import math
import re
from bisect import bisect_left, bisect_right
from collections import Counter
from functools import lru_cache

import snowballstemmer

_PARA = re.compile(r"\n[ \t]*\n\s*")
_SENT = re.compile(r"[.!?][\"')\]]*\s+|\n")
_SPACE = re.compile(r"\s+")
_WORD = re.compile(r"\w+")
_HEADING = re.compile(r"(?m)^#{1,6}[ \t]+(.+?)[ \t#]*$")
# Lucene's default English stopword set.
STOPWORDS = frozenset("a an and are as at be but by for if in into is it no not of on or such that the their "
                      "then there these they this to was will with".split())
_stemmer = snowballstemmer.stemmer("english")


def chunk(text: str, target: int = 1000, hard: int = 1500, floor: int = 400) -> list[tuple[int, int]]:
    """Contiguous (start, end) offsets covering all of text, cut at the boundary nearest `target`:
    paragraph breaks first, then sentence ends/newlines, then whitespace, else a hard cut."""
    cuts = [[m.end() for m in p.finditer(text)] for p in (_PARA, _SENT, _SPACE)]
    out, start, n = [], 0, len(text)
    while start < n:
        if n - start <= hard:
            out.append((start, n))
            break
        end = start + hard
        for c in cuts:
            i, j = bisect_left(c, start + floor), bisect_right(c, start + hard)
            if i < j:
                end = min(c[i:j], key=lambda x: abs(x - start - target))
                break
        out.append((start, end))
        start = end
    return out


@lru_cache(maxsize=200_000)
def _stem(word: str) -> str:
    return _stemmer.stemWord(word)


def tokens(text: str) -> list[str]:
    return [_stem(w) for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


class Index:
    """Okapi BM25 with Lucene's IDF and tf normalisation (no (k1+1) factor; ranking is identical)."""

    def __init__(self, text: str, chunks: list[tuple[int, int]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(tokens(text[s:e])) for s, e in chunks]
        self.lengths = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.lengths) / len(self.lengths) if chunks else 0.0
        n = len(chunks)
        df = Counter(t for c in self.tf for t in c)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = set(tokens(query)) & self.idf.keys()
        out = []
        for tf, length in zip(self.tf, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg) if self.avg else self.k1
            out.append(sum(self.idf[t] * tf[t] / (tf[t] + norm) for t in terms if t in tf))
        return out

    def top(self, query: str, k: int) -> list[int]:
        """Best chunk numbers, ties by position; empty when no query term occurs."""
        scores = self.scores(query)
        return sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: (-scores[i], i))[:k]


def windows(hits: list[int], n_chunks: int) -> list[tuple[int, int, int]]:
    """(first, last, hit) chunk ranges: each hit ±1, touching ranges merged, best-ranked hit first."""
    out: list[list[int]] = []
    for h in hits:
        a, b = max(0, h - 1), min(n_chunks - 1, h + 1)
        for w in out:
            if a <= w[1] + 1 and w[0] <= b + 1:
                w[0], w[1] = min(w[0], a), max(w[1], b)
                break
        else:
            out.append([a, b, h])
    # A widened range can now touch a later one.
    merged: list[list[int]] = []
    for w in out:
        for m in merged:
            if w[0] <= m[1] + 1 and m[0] <= w[1] + 1:
                m[0], m[1] = min(m[0], w[0]), max(m[1], w[1])
                break
        else:
            merged.append(w)
    return [tuple(w) for w in merged]


def chunk_at(chunks: list[tuple[int, int]], offset: int) -> int:
    return max(0, bisect_right([s for s, _ in chunks], offset) - 1)


def outline(text: str, chunks: list[tuple[int, int]], pages: list[int], budget: int,
            toc: list[tuple[int, str]] = ()) -> str:
    """Markdown headings (HTML only: in PDF text a `#` line is usually a code comment) and PDF bookmarks
    as `[cN] title`; page starts only for PDFs without bookmarks."""
    entries = [] if pages else [(m.start(), f"[c{chunk_at(chunks, m.start())}] {' '.join(m.group(1).split())[:60]}")
                                for m in _HEADING.finditer(text)]
    entries += [(off, f"[c{chunk_at(chunks, off)}] {title[:60]}") for off, title in toc]
    if not toc:
        entries += [(off, f"p.{i} [c{chunk_at(chunks, off)}]") for i, off in enumerate(pages, 1)]
    items = [e for _, e in sorted(entries)]
    if not items or budget <= 0:
        return ""
    sep = " · "
    step = max(1, math.ceil(len(sep.join(items)) / budget))
    line = sep.join(items[::step])
    return line if len(line) <= budget else line[:budget].rsplit(sep, 1)[0]


def overview(sid: int, title: str, url: str, text: str, chunks: list[tuple[int, int]], pages: list[int],
             note: str, toc: list[tuple[int, str]] = (), limit: int = 800) -> str:
    head = (f"Source [{sid}]: {title or '(untitled)'}\n{url}\n"
            f"Document: {len(text):,} chars in {len(chunks)} chunks. {note} "
            "Use focus=... or at=<chunk> to read more.\n")
    line = outline(text, chunks, pages, limit - len(head) - len("Outline: \n"), toc)
    return head + (f"Outline: {line}\n" if line else "")


def render(text: str, chunks: list[tuple[int, int]], wins: list[tuple[int, int, int]],
           budget: int) -> tuple[str, list[tuple[int, int]]]:
    """Window blocks within `budget` chars, and the exact (start, end) spans delivered."""
    n, parts, spans = len(chunks), [], []
    for a, b, hit in wins:
        label = f"--- chunks {a}-{b} of {n} ---\n"
        s, e = chunks[a][0], chunks[b][1]
        if len(label) + e - s > budget:  # drop the neighbours, then clip the hit chunk itself
            label = f"--- chunk {hit} of {n} ---\n"
            s, e = chunks[hit]
            if len(label) + e - s > budget:
                label = f"--- chunk {hit} of {n} (clipped) ---\n"
                e = s + budget - len(label)
        if e - s < 200 and spans:
            break
        if e <= s:
            break
        parts.append(label + text[s:e])
        spans.append((s, e))
        budget -= len(label) + e - s + 1
    return "\n".join(parts), spans
