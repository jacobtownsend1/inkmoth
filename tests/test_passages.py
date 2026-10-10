import random

import pytest

from inkmoth.passages import Index, chunk, overview, render, tokens, windows

FILLER = ("weather harbour lantern copper meadow violin orchard pebble saddle thunder velvet "
          "candle marble ribbon quarry walnut beacon falcon garnet hollow").split()
ANSWER = "The retriever was trained jointly with the generator by marginalising over retrieved documents."


def filler(rng: random.Random, chars: int) -> str:
    out, size = [], 0
    while size < chars:
        sentence = " ".join(rng.choice(FILLER) for _ in range(rng.randint(6, 18))).capitalize() + ". "
        if rng.random() < 0.1:
            sentence += "\n\n"
        out.append(sentence)
        size += len(sentence)
    return "".join(out)


@pytest.fixture(scope="module")
def long_doc():
    rng = random.Random(7)
    text = filler(rng, 140_000) + ANSWER + " " + filler(rng, 60_000)
    return text, chunk(text)


def test_chunks_cover_text_and_respect_limits(long_doc):
    text, chunks = long_doc
    assert "".join(text[s:e] for s, e in chunks) == text
    assert all(chunks[i][1] == chunks[i + 1][0] for i in range(len(chunks) - 1))
    assert max(e - s for s, e in chunks) <= 1500
    # Cuts land on sentence ends or paragraph breaks, not mid-word.
    assert all(text[e - 1].isspace() for _, e in chunks[:-1])


def test_unbroken_text_is_hard_cut():
    assert chunk("x" * 4000) == [(0, 1500), (1500, 3000), (3000, 4000)]
    assert chunk("") == []


def test_bm25_matches_lucene_reference():
    # Reference: bm25s 0.2 method="lucene", k1=1.5, b=0.75 over the same tokens (computed once, float32).
    docs = ["The retriever is trained jointly with the generator.",
            "Generators produce text; retrieval supplies passages to the generator.",
            "Dense passage retrieval uses a bi-encoder trained on question answer pairs.",
            "Unrelated text about cooking pasta and tomatoes, tomatoes, tomatoes.",
            "Training the retriever end to end improves retrieval of passages for generation."]
    text, chunks, start = "".join(docs), [], 0
    for d in docs:
        chunks.append((start, start + len(d)))
        start += len(d)
    scores = Index(text, chunks).scores("how is the retriever trained for generation")
    assert scores == pytest.approx([0.688653886, 0.431385636, 0.285527289, 0.0, 0.576232433], abs=1e-6)


def test_tokens_stem_and_drop_stopwords():
    assert tokens("The Retrievers were TRAINING on it") == ["retriev", "were", "train"]


def test_answer_at_70_percent_depth_is_in_top_window(long_doc):
    text, chunks = long_doc
    hits = Index(text, chunks).top("how was the retriever trained with the generator", 2)
    body, spans = render(text, chunks, windows(hits, len(chunks)), 6000)
    assert ANSWER in body
    assert ANSWER in text[spans[0][0]:spans[0][1]]
    assert 0.65 < text.index(ANSWER) / len(text) < 0.75


def test_no_matching_term_returns_no_hits(long_doc):
    text, chunks = long_doc
    assert Index(text, chunks).top("zeppelin", 2) == []


def test_windows_merge_touching_ranges_best_first():
    assert windows([10, 12], 100) == [(9, 13, 10)]
    assert windows([10, 13], 100) == [(9, 14, 10)]  # 9-11 and 12-14 are adjacent
    assert windows([50, 10], 100) == [(49, 51, 50), (9, 11, 10)]
    assert windows([0, 99], 100) == [(0, 1, 0), (98, 99, 99)]


def test_render_clips_to_budget_and_reports_exact_spans(long_doc):
    text, chunks = long_doc
    body, spans = render(text, chunks, [(9, 11, 10), (40, 42, 41)], 2500)
    assert len(body) <= 2500
    assert spans[0] == chunks[10]  # neighbours dropped before clipping the hit chunk
    for s, e in spans:
        assert text[s:e] in body


def test_overview_outline_from_headings_and_pages():
    text = "# Abstract\nshort\n\n## 4.2 Retriever\n" + "word " * 600 + "\n## 5 Results\nend"
    chunks = chunk(text)
    ov = overview(3, "RAG", "https://a.test/p", text, chunks, [], 'Best matches for focus "x".')
    assert ov.startswith("Source [3]: RAG\nhttps://a.test/p\nDocument: ")
    assert "[c0] Abstract" in ov and "4.2 Retriever" in ov and "p.2" not in ov
    assert len(ov) <= 800
    pdf = overview(3, "RAG", "https://a.test/p.pdf", text, chunks, [0, 1500], "n")
    assert "p.1 [c0]" in pdf and "p.2 [c" in pdf and "Abstract" not in pdf


def test_outline_is_sampled_to_fit():
    text = "".join(f"# Heading number {i}\nbody text here.\n\n" for i in range(500))
    ov = overview(1, "t", "https://a.test", text, chunk(text), [], "note")
    assert len(ov) <= 800 and "Heading number 0" in ov


def test_bookmarks_replace_page_entries_in_the_outline():
    text = "intro text. " * 200 + "register map details. " * 200
    chunks = chunk(text)
    ov = overview(1, "RP2040", "https://a.test/ds.pdf", text, chunks, [0, 2400], "n",
                  toc=[(0, "1 Introduction"), (2400, "2.3 Register map")])
    assert "[c0] 1 Introduction" in ov and "2.3 Register map" in ov and "p.2" not in ov


def test_pdf_text_hash_lines_are_not_headings():
    text = "# the necessary initialization\nx = 1\n" * 100
    ov = overview(1, "t", "https://a.test/x.pdf", text, chunk(text), [0], "n")
    assert "necessary initialization" not in ov and "p.1 [c0]" in ov
