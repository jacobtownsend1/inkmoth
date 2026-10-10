import httpx, pytest, respx
from inkmoth.fetch import Fetcher
from inkmoth.sources import SourceRegistry

HTML = "<html><head><title>Doc Title</title></head><body><nav>menu</nav><article><h1>Heading</h1><p>" + "Real content sentence. " * 40 + "</p></article></body></html>"

@respx.mock
async def test_html_to_markdown_with_source_header():
    respx.get("https://site.test/doc").mock(return_value=httpx.Response(200, text=HTML, headers={"content-type": "text/html"}))
    reg = SourceRegistry()
    out = await Fetcher(reg, max_chars=5000).fetch("https://site.test/doc")
    assert out.startswith("Source [1]: Doc Title\nhttps://site.test/doc\n\n")
    assert "Real content sentence." in out and "menu" not in out
    assert reg.get(1).title == "Doc Title"

def long_text():
    import random
    from test_passages import ANSWER, filler
    rng = random.Random(7)
    return filler(rng, 140_000) + ANSWER + " " + filler(rng, 60_000)


@respx.mock
async def test_one_download_serves_every_focus_and_position(settings):
    from inkmoth.budget import RunBudget
    from test_passages import ANSWER
    route = respx.get("https://site.test/paper").mock(return_value=httpx.Response(200, text=long_text(), headers={"content-type": "text/plain"}))
    budget = RunBudget(settings)
    f = Fetcher(SourceRegistry(), max_chars=6000, budget=budget)
    first = await f.fetch("https://site.test/paper", "how was the retriever trained with the generator")
    assert first.startswith("Source [1]: (untitled)\nhttps://site.test/paper\nDocument: ")
    assert ANSWER in first and len(first) <= 6000
    await f.fetch("https://site.test/paper", "copper meadow")
    await f.fetch("https://site.test/paper", "")
    by_position = await f.fetch("https://site.test/paper", "ignored", at=5)
    assert "--- chunks 4-6 of " in by_position and "at overrides focus" in by_position
    assert route.call_count == 1 and budget.fetches == 1


@respx.mock
async def test_at_out_of_range_and_unmatched_focus():
    respx.get("https://site.test/paper").mock(return_value=httpx.Response(200, text=long_text(), headers={"content-type": "text/plain"}))
    f = Fetcher(SourceRegistry())
    out = await f.fetch("https://site.test/paper", at=10_000)
    assert out.startswith("ERROR: at must be 0..")
    out = await f.fetch("https://site.test/paper", "zeppelin")
    assert "No passage matches the focus" in out and "--- chunks 0-1 of " in out


@respx.mock
async def test_short_document_is_returned_whole():
    respx.get("https://site.test/short").mock(return_value=httpx.Response(200, text="Short and sweet.", headers={"content-type": "text/plain"}))
    reg = SourceRegistry()
    out = await Fetcher(reg).fetch("https://site.test/short", "anything")
    assert out == "Source [1]: (untitled)\nhttps://site.test/short\n\nShort and sweet."
    assert reg.get(1).excerpts == ["Short and sweet."]


@respx.mock
async def test_lru_eviction_downloads_again(monkeypatch):
    from inkmoth import fetch
    monkeypatch.setattr(fetch, "CACHE_CHARS", 250_000)
    text = long_text()
    a = respx.get("https://site.test/a").mock(return_value=httpx.Response(200, text=text, headers={"content-type": "text/plain"}))
    respx.get("https://site.test/b").mock(return_value=httpx.Response(200, text=text, headers={"content-type": "text/plain"}))
    f = Fetcher(SourceRegistry())
    await f.fetch("https://site.test/a")
    await f.fetch("https://site.test/b")  # 2 x 200k > 250k: evicts a
    await f.fetch("https://site.test/a")
    assert a.call_count == 2 and f._cache_chars <= 250_000

@respx.mock
async def test_errors_are_strings():
    respx.get("https://site.test/404").mock(return_value=httpx.Response(404))
    respx.get("https://site.test/pdf").mock(return_value=httpx.Response(200, content=b"%PDF-", headers={"content-type": "application/pdf"}))
    respx.get("https://site.test/boom").mock(side_effect=httpx.ConnectError("boom"))
    f = Fetcher(SourceRegistry())
    assert "HTTP 404" in await f.fetch("https://site.test/404")
    assert (await f.fetch("https://site.test/pdf")).startswith("ERROR:")
    assert (await f.fetch("https://site.test/boom")).startswith("ERROR:")
    assert (await f.fetch("ftp://x")).startswith("ERROR:")

@respx.mock
async def test_plain_text_passthrough():
    respx.get("https://site.test/t.txt").mock(return_value=httpx.Response(200, text="just text", headers={"content-type": "text/plain"}))
    assert "just text" in await Fetcher(SourceRegistry()).fetch("https://site.test/t.txt")


@respx.mock
async def test_private_url_refused_without_request():
    route = respx.get("http://192.168.1.100:8000/v1/models").mock(return_value=httpx.Response(200, text="secret"))
    out = await Fetcher(SourceRegistry()).fetch("http://192.168.1.100:8000/v1/models")
    assert out.startswith("ERROR:") and route.call_count == 0


@respx.mock
async def test_redirect_to_private_is_refused_at_the_hop():
    respx.get("https://site.test/go").mock(return_value=httpx.Response(302, headers={"location": "http://10.0.0.5/admin"}))
    inner = respx.get("http://10.0.0.5/admin").mock(return_value=httpx.Response(200, text="secret"))
    out = await Fetcher(SourceRegistry()).fetch("https://site.test/go")
    assert out.startswith("ERROR:") and inner.call_count == 0


@respx.mock
async def test_public_redirect_is_followed():
    respx.get("https://site.test/old").mock(return_value=httpx.Response(301, headers={"location": "https://site.test/new"}))
    respx.get("https://site.test/new").mock(return_value=httpx.Response(200, text=HTML, headers={"content-type": "text/html"}))
    reg = SourceRegistry()
    out = await Fetcher(reg).fetch("https://site.test/old")
    assert out.startswith("Source [1]: Doc Title\nhttps://site.test/new")


@respx.mock
async def test_trailing_slash_pages_not_conflated():
    a = respx.get('https://site.test/page').respond(200, text='first page', headers={'content-type':'text/plain'})
    b = respx.get('https://site.test/page/').respond(200, text='second page', headers={'content-type':'text/plain'})
    async with Fetcher(SourceRegistry()) as fetcher:
        one = await fetcher.fetch('https://site.test/page')
        two = await fetcher.fetch('https://site.test/page/')
    assert 'first page' in one and 'second page' in two and a.call_count == b.call_count == 1


@respx.mock
async def test_focus_selects_later_relevant_passages():
    text = ('Unrelated introduction. ' * 400) + ('The interconnect uses RDMA at 200 Gbps. ' * 100)
    respx.get("https://site.test/spec").mock(return_value=httpx.Response(200, text=text, headers={"content-type": "text/plain"}))
    result = await Fetcher(SourceRegistry(), max_chars=2000).fetch("https://site.test/spec", 'RDMA interconnect')
    assert 'RDMA' in result and 'Best matches for focus' in result and len(result) <= 2000


def make_pdf(pages, outline=()):
    """Tiny real PDF: one Helvetica text line per page (None = image-only page), optional bookmarks."""
    import io
    import pypdf
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    writer = pypdf.PdfWriter()
    font = DictionaryObject({NameObject('/Type'):NameObject('/Font'), NameObject('/Subtype'):NameObject('/Type1'),
                             NameObject('/BaseFont'):NameObject('/Helvetica')})
    for line in pages:
        page = writer.add_blank_page(612,792)
        if line is None:
            continue
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(b'BT /F1 12 Tf 50 700 Td (' + line.encode() + b') Tj ET')
        page[NameObject('/Contents')] = writer._add_object(stream)
    for title, page_index in outline:
        writer.add_outline_item(title, page_index)
    data = io.BytesIO()
    writer.write(data)
    return data.getvalue()


async def test_pdf_pages_and_bookmarks_travel_next_to_the_text():
    data = make_pdf(['Documented memory is 128 GB.', 'Second page text.'], outline=[('Intro', 0), ('Results', 1)])
    async with Fetcher(SourceRegistry()) as fetcher:
        _, body, pages, toc = await fetcher._extract(data, 'application/pdf')
    assert '128 GB' in body
    # Page starts are offsets next to the text, never markers inside it.
    assert pages[0] == 0 and body[pages[1]:].startswith('Second page') and '[page' not in body
    assert toc == [[0, 'Intro'], [pages[1], 'Results']]


async def test_scanned_pdf_gets_a_clear_error():
    async with Fetcher(SourceRegistry()) as fetcher:
        with pytest.raises(ValueError, match='no extractable text.*OCR'):
            await fetcher._extract(make_pdf([None, None]), 'application/pdf')


def test_dehyphenation_uses_the_documents_own_vocabulary():
    from inkmoth.extract import dehyphenate
    pages = ['Language model pre-training helps. The re-\nlationships between pre-\ntraining and fine-\ntuning matter.',
             'We study fine-tuning and pre-training.']
    out = dehyphenate(pages)
    assert 'relationships between pre-training and fine-tuning matter' in out[0]
    assert dehyphenate(['un-\nknown word']) == ['unknown word']  # unseen either way: treat as soft hyphenation


def test_pdf_text_is_capped_with_a_page_note(monkeypatch):
    from inkmoth import extract
    monkeypatch.setattr(extract, 'MAX_TEXT_CHARS', 40)
    _, body, pages, _ = extract.extract(make_pdf(['First page has some text.', 'Second page.', 'Third.']), 'application/pdf')
    assert body.endswith('[PDF text truncated after page 2 of 3]') and len(pages) == 2


@respx.mock
async def test_pdf_byte_cap_is_separate_from_html_cap():
    big = b'%PDF-1.7\n' + b'0' * 3_000_000
    respx.get('https://site.test/big.pdf').mock(return_value=httpx.Response(200, content=big, headers={'content-type': 'application/pdf'}))
    respx.get('https://site.test/big.html').mock(return_value=httpx.Response(200, content=b'<html>' + b'x' * 3_000_000, headers={'content-type': 'text/html'}))
    async with Fetcher(SourceRegistry()) as fetcher:
        _, ctype, data = await fetcher._download('https://site.test/big.pdf')
        assert ctype == 'application/pdf' and len(data) == len(big)
        with pytest.raises(ValueError, match='byte limit'):
            await fetcher._download('https://site.test/big.html')
    async with Fetcher(SourceRegistry(), max_pdf_bytes=1_000_000) as fetcher:
        with pytest.raises(ValueError, match='byte limit'):
            await fetcher._download('https://site.test/big.pdf')


@respx.mock
async def test_octet_stream_is_accepted_only_when_it_is_a_pdf():
    pdf = make_pdf(['Datasheet register map.'])
    respx.get('https://site.test/ds').mock(return_value=httpx.Response(200, content=pdf, headers={'content-type': 'application/octet-stream'}))
    respx.get('https://site.test/bin').mock(return_value=httpx.Response(200, content=b'MZ\x90\x00' * 100, headers={'content-type': 'application/octet-stream'}))
    async with Fetcher(SourceRegistry()) as fetcher:
        out = await fetcher.fetch('https://site.test/ds')
        assert 'Datasheet register map.' in out
        assert 'not a PDF' in await fetcher.fetch('https://site.test/bin')


def test_dehyphenation_keeps_multi_hyphen_compounds():
    from inkmoth.extract import dehyphenate
    out = dehyphenate(['We train end-to-\nend. A state-of-the-\nart model. Also end-to-end and state-of-the-art.'])
    assert 'train end-to-end.' in out[0] and 'A state-of-the-art model' in out[0]
    assert dehyphenate(['a well-con-\nnected graph']) == ['a well-con-nected graph']  # unattested: keep, never invent


def test_leading_empty_pdf_pages_keep_their_numbers():
    from inkmoth.extract import extract
    _, body, pages, _ = extract(make_pdf([None, 'First text page.', 'Second text page.']), 'application/pdf')
    assert len(pages) == 3 and body[pages[1]:].startswith('First text page') and body[pages[2]:].startswith('Second')


@respx.mock
async def test_redirect_target_is_served_from_the_cache():
    respx.get('https://site.test/latest.pdf').mock(return_value=httpx.Response(302, headers={'location': 'https://cdn.site.test/v2.pdf'}))
    target = respx.get('https://cdn.site.test/v2.pdf').mock(return_value=httpx.Response(200, text='Version two text.', headers={'content-type': 'text/plain'}))
    f = Fetcher(SourceRegistry())
    await f.fetch('https://site.test/latest.pdf')
    out = await f.fetch('https://cdn.site.test/v2.pdf')
    assert 'Version two text.' in out and target.call_count == 1
