from inkmoth.sources import SourceRegistry

def test_ids_are_stable_and_one_based():
    r = SourceRegistry()
    a = r.add("https://example.com/a", "A")
    b = r.add("https://example.com/b")
    assert (a, b) == (1, 2)
    assert r.add("https://example.com/a") == 1
    assert r.get(1).title == "A"

def test_normalization_merges_variants():
    r = SourceRegistry()
    a = r.add("https://Example.com/path/#frag")
    b = r.add("https://example.com/path/")
    c = r.add("https://example.com/path")
    assert a == b and c != a

def test_title_backfilled_and_table():
    r = SourceRegistry()
    r.add("https://example.com/x")
    r.add("https://example.com/x", "Title X")
    assert r.get(1).title == "Title X"
    assert r.table() == "[1] Title X — https://example.com/x"
    assert [s.id for s in r.all()] == [1]


def test_adjacent_reads_let_a_boundary_quote_validate():
    reg = SourceRegistry()
    sid = reg.add("https://a.test/doc")
    text = "Alpha sentence one. The retriever was trained jointly with the generator. Omega."
    cut = text.index("jointly")
    reg.record(sid, text[:cut], actor="r1", fetched=True, start=0, version="v1")
    assert not reg.supports("r1", sid, "trained jointly with the generator")
    reg.record(sid, text[cut:], actor="r1", fetched=True, start=cut, version="v1")
    assert reg.supports("r1", sid, "trained jointly with the generator")
    assert reg.get(sid).excerpts == [text]
    assert reg.passages("r1")[0]["passages"] == [text]


def test_unseen_gap_and_disagreeing_overlap_stay_separate():
    reg = SourceRegistry()
    sid = reg.add("https://a.test/doc")
    reg.record(sid, "first part of it", actor="r1", fetched=True, start=0, version="v1")
    reg.record(sid, "far later passage", actor="r1", fetched=True, start=500, version="v1")
    assert not reg.supports("r1", sid, "of it far later")
    # A re-downloaded, changed page delivers different text at an overlapping offset.
    reg.record(sid, "FIRST PART CHANGED", actor="r1", fetched=True, start=5, version="v1")
    assert reg.get(sid).excerpts == ["first part of it", "FIRST PART CHANGED", "far later passage"]


def test_overlapping_reads_merge_without_duplication():
    from inkmoth.sources import merge_span
    text = "0123456789abcdefghij"
    spans = merge_span([], 5, text[5:12], "v1")
    spans = merge_span(spans, 0, text[0:8], "v1")
    spans = merge_span(spans, 15, text[15:20], "v1")
    spans = merge_span(spans, 12, text[12:15], "v1")  # bridges the two
    assert spans == [(0, text, "v1")]
    assert merge_span([(None, "x", None)], None, "x") == [(None, "x", None)]


def test_touching_spans_from_different_versions_never_join():
    from inkmoth.sources import merge_span
    reg = SourceRegistry()
    sid = reg.add("https://a.test/doc")
    reg.record(sid, "revenue was", actor="r1", fetched=True, start=0, version="old")
    reg.record(sid, " 5 million", actor="r1", fetched=True, start=11, version="new")
    assert not reg.supports("r1", sid, "revenue was 5 million")
    assert len(merge_span([(0, "abc", "v1")], 3, "def", "v2")) == 2
