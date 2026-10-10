"""Disposable document extraction worker. Invoked without a shell by Fetcher."""
import io
import json
import re
import sys
from collections import Counter

MAX_TEXT_CHARS = 5_000_000  # a ~1,000-page book is ~2.5M chars
_SPLIT = re.compile(r"(\w+(?:-\w+)*)-[ \t]*\r?\n[ \t]*([a-z]\w*)")  # whole compound before the break


def dehyphenate(pages: list[str]) -> list[str]:
    """Join words broken at a line-end hyphen. Keep the hyphen when the document itself uses the hyphenated
    form more often than the joined one ("pre-training" stays, "re-lationships" becomes "relationships")."""
    words = Counter(re.findall(r"\w+(?:-\w+)*", "\n".join(pages).lower()))

    def join(m):
        a, b = m.group(1), m.group(2)
        hyphenated, joined = words[f"{a}-{b}".lower()], words[f"{a}{b}".lower()]
        # A break after an earlier hyphen ("end-to-\nend") continues a compound unless the joined form is attested.
        return f"{a}-{b}" if hyphenated > joined or ("-" in a and not joined) else a + b

    return [_SPLIT.sub(join, p) for p in pages]


def _pdf(data: bytes) -> tuple[str, str, list[int], list[tuple[int, str]]]:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    title = str((reader.metadata or {}).get("/Title", "") or "")
    texts, size = [], 0
    for page in reader.pages:
        texts.append(page.extract_text() or "")
        size += len(texts[-1]) + 2
        if size > MAX_TEXT_CHARS:
            break
    if not "".join(texts).strip():
        raise ValueError("PDF has no extractable text (scanned images?); OCR is not supported")
    texts = dehyphenate(texts)
    pages, offset = [], 0
    for t in texts:
        pages.append(offset)
        offset += len(t) + 2  # the "\n\n" join
    body = "\n\n".join(texts)
    if len(texts) < len(reader.pages):
        body += f"\n[PDF text truncated after page {len(texts)} of {len(reader.pages)}]"
    toc = []
    try:
        def walk(items, depth):
            for item in items:
                if isinstance(item, list):
                    if depth < 1:
                        walk(item, depth + 1)
                    continue
                n = reader.get_destination_page_number(item)
                if n is not None and 0 <= n < len(pages):
                    toc.append((pages[n], " ".join(str(item.title).split())[:80]))
        walk(reader.outline, 0)
    except Exception:
        toc = []  # bookmarks are a navigation aid; a broken outline must not fail extraction
    return title, body, pages, toc


def extract(data: bytes, ctype: str) -> tuple[str, str, list[int], list[tuple[int, str]]]:
    """Full text, the char offset of each PDF page start, and PDF bookmarks as (offset, title).
    Page numbers never go inside the text, so quotes across a page break still validate."""
    title, pages, toc = "", [], []
    if ctype == "application/pdf":
        title, body, pages, toc = _pdf(data)
    else:
        import trafilatura
        from trafilatura.utils import decode_file
        text = decode_file(data)
        if "html" in ctype or "<html" in text[:200].lower():
            from html import unescape
            m = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
            title = unescape(m.group(1)) if m else ""
            body = trafilatura.extract(text, output_format="markdown", include_tables=True,
                                       include_links=False, favor_recall=True) or ""
            # Never return raw HTML/scripts when extraction failed.
        else:
            body = text
        body = body[:MAX_TEXT_CHARS]
    title = " ".join(title.split())[:300]
    lead = len(body) - len(body.lstrip())
    body = body.strip()
    shift = lambda o: max(0, o - lead)
    # One entry per page, in order: the outline numbers pages by position, so empty pages must keep theirs.
    return (title, body, [shift(o) for o in pages if o - lead < len(body)],
            [(shift(o), t) for o, t in toc if o - lead < len(body)])


def main():
    ctype, max_bytes, cpu_seconds = sys.argv[1:]
    # This is a resource boundary, not a sandbox for executing document code.
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (int(cpu_seconds),) * 2)
    except (ImportError, ValueError):
        pass
    try:
        data = sys.stdin.buffer.read(int(max_bytes) + 1)
        if len(data) > int(max_bytes):
            raise ValueError("document exceeds byte limit")
        title, body, pages, toc = extract(data, ctype)
        print(json.dumps({"title": title, "body": body, "pages": pages, "toc": toc}))
    except ValueError as exc:  # our own messages, already readable
        print(json.dumps({"error": str(exc)}))
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))


if __name__ == "__main__":
    main()
