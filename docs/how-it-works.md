# How it works

```
clarify → plan → research (search, fetch, read) → extract + verify quotes → assess coverage
             ↑                                                                   │
             └──────────── follow-up sub-questions while gaps and budget remain ─┘
                                                                                 ↓
                                              synthesize → review → finish (citations, tags, sources)
```

## The loop

1. **Clarify.** A short check decides whether different reasonable readings of the question would change
   the plan (scope, time frame, audience, platform or version). If so, the CLI asks up to three questions
   once; you can answer or just press Enter. Otherwise, or when nobody can be asked (library, piped CLI,
   MCP hosts without elicitation), the run proceeds on stated assumptions, which appear under the report
   title. Answers and assumptions are passed to every later step.
2. **Plan.** The planner splits the question into focused sub-questions. The application assigns IDs and
   drops duplicates.
3. **Research.** Each sub-question gets its own researcher with `web_search` and `fetch_page`. Search results
   are leads only. `fetch_page` reads a page or PDF; long documents return an outline plus the passages that
   best match the researcher's focus (see below). Tool gathering uses at most 65% of the researcher's
   deadline so there is always time left to extract what was found.
4. **Extract and verify.** A fresh, tool-free context sees only the passages *that researcher* was shown and
   extracts atomic claims, each with a supporting quote. Every quote must occur verbatim (whitespace and PDF
   ligatures normalised, nothing fuzzy) in fetched text the same researcher saw; search snippets never count.
   Failing claims are moved to `findings.rejected` and one bounded re-extraction may repair quoting
   mistakes. Rejected claim text never reaches another prompt: it may carry injected instructions. Later
   researchers only learn which source IDs had failing quotes.
5. **Assess.** After every round, including the last, an assessor maps each explicit part of the original
   question to the facts that answer it. A part counts as answered only through real fact references. The
   assessor also sees how many independent hosts back each fact, and records contradictions as pairs of
   conflicting facts. Gaps become follow-up sub-questions while the depth and budgets allow.
6. **Synthesize and review.** The writer receives the verified claims and quotes, the coverage assessment
   and any contradictions (as claims with source IDs). A separate tool-free reviewer checks the draft against
   the same quotes for unsupported additions, dropped conditions and scope errors.
7. **Finish.** The application, not the model, produces the parts that must be right:
   - citations are checked against the sources the claims actually used; unknown ones become `[unverified]`;
   - a citation group whose sources all live on one website gets a `†`, explained in a one-line legend;
   - a *Disagreements* section is added if the writer left out the recorded contradictions;
   - the bibliography (title, URL, published and retrieved dates) is generated from the source registry.

Quote checking proves **provenance, not truth**: a quote can exist and still not support the claim, and the
reviewer is the same model and can be wrong. `status: complete` means the workflow finished without recorded
limitations, not that the report is correct.

**Corroboration** counts distinct hosts (`www.` ignored). Syndicated copies of one press release on two
different sites therefore count as two; the `†` marks evidence that is certainly thin, its absence does not
prove independence. For questions about one vendor's documentation most citations carry a `†`; that is
expected, and a quick way to see where outside confirmation exists.

**Assumptions** appear only when the question was genuinely ambiguous. They describe how the question was
read (scope, time frame, platform), never facts: in testing, letting the model state assumptions for clear
questions leaked remembered "facts" into every later step.

## Long documents

`fetch_page(url, focus, at)` downloads and extracts a document once per run and caches the text (up to
20 M characters per run, least recently used first out). The text is split into ~1,000-character chunks at
sentence boundaries and ranked with BM25 (Lucene's formula, Snowball stemming). A long document returns:

```
Source [12]: RP2040 Datasheet
https://datasheets.raspberrypi.com/rp2040/rp2040-datasheet.pdf
Document: 1,333,615 chars in 1328 chunks. Best matches for focus "DMA channel chaining". …
Outline: [c0] Colophon · [c67] Chapter 1. Introduction · [c77] Chapter 2. System Description · …
--- chunks 235-238 of 1328 ---
<passage text>
```

The outline comes from HTML headings or PDF bookmarks. The researcher can ask again with a new `focus`, or read
a position with `at=<chunk>`; neither downloads again or spends fetch budget. A call returns at most
`fetch_max_chars` (6,000 by default). Quotes that span two separately read but adjacent windows still verify,
because the evidence ledger merges touching spans; text the researcher never saw does not.

## PDFs

PDFs are parsed with [pypdf](https://pypdf.readthedocs.io/) in a separate worker process (60 s CPU, 512 MiB),
up to 50 MB and 5 M characters of text. Line-break hyphenation is undone using the document's own vocabulary,
so "re-\nlationships" becomes "relationships" while "pre-training" keeps its hyphen. Scanned PDFs without a
text layer are reported as such; there is no OCR. Servers that label PDFs `application/octet-stream` are
accepted when the file starts with `%PDF-`.

The extractor was chosen by comparing three libraries on six real PDFs (two arXiv papers, a 144-page survey,
a 642-page datasheet, a 1,151-page / 45 MB book and a scan), with arXiv's HTML versions as reading-order
ground truth:

| | pypdf | pdfminer.six | pypdfium2 |
|---|---|---|---|
| CPU: paper / datasheet / book | 0.4 / 12.9 / 28.7 s | 0.7 / 41.6 / 66 s | 0.05 / 0.7 / 1.6 s |
| Sentences verbatim, two-column BERT / RAG | 92 / 98% | 88 / 98% | 92 / 98% |
| Words glued together (worst document) | 0.35% | 0.06% | 0.06% |

PDFium is much faster but is C++ parsing files chosen by a model that read untrusted web pages, in a process
with resource limits and no OS sandbox; pypdf is memory-safe. Swapping is one function in `extract.py` if
speed matters more to you.

## Search providers that return page text

Some search APIs (Tavily) return the full extracted page with each result. inkmoth caches that text as if it
had fetched the page, so `fetch_page` on those URLs needs no download. Passage selection and quote checks work
the same, but quotes are then verified against the provider's extraction of the page.

## Resuming

The CLI checkpoints after every researcher to `reports/.checkpoints/<run_id>.json`. `--resume` keeps the
sources (and their IDs), the sub-questions that produced findings and the clarifications or assumptions,
re-runs the rest of the interrupted round, then continues normally. Budgets and timers count only the resumed
part, and the page cache starts empty.

## What it does not do

No JavaScript rendering, logins, paywalls or OCR. No guarantee against prompt injection: prompts tell the
model to treat pages as data, and quote verification limits what an injected page can make the report claim,
but a model can still be steered. See [security](security.md).
