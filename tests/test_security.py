import asyncio

import httpcore
import httpx
import pytest

from inkmoth import netguard
from inkmoth.fetch import Fetcher
from inkmoth.sources import SourceRegistry
from inkmoth.transport import PublicNetworkBackend, PublicTransport


class RecordingBackend(httpcore.AsyncNetworkBackend):
    def __init__(self):
        self.hosts, self.sni, self.writes = [], [], []

    async def connect_tcp(self, host, port, **kwargs):
        self.hosts.append(host)
        owner = self

        class Stream(httpcore.AsyncMockStream):
            async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
                owner.sni.append(server_hostname)
                return self

            async def write(self, buffer, timeout=None):
                owner.writes.append(buffer)
                await super().write(buffer, timeout=timeout)

        return Stream([b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK"])


async def test_dns_is_pinned_and_tls_identity_preserved(monkeypatch):
    calls = []
    def resolve(host):
        calls.append(host)
        return ['93.184.216.34'] if len(calls) == 1 else ['127.0.0.1']
    monkeypatch.setattr(netguard, 'resolve_host', resolve)
    backend = RecordingBackend()
    async with httpx.AsyncClient(transport=PublicTransport(backend), trust_env=False) as client:
        r = await client.get('https://example.com/page')
    assert r.text == 'OK'
    assert calls == ['example.com']
    assert backend.hosts == ['93.184.216.34']
    assert backend.sni == ['example.com']
    assert b'Host: example.com' in b''.join(backend.writes)


async def test_mixed_dns_answers_never_dial(monkeypatch):
    monkeypatch.setattr(netguard, 'resolve_host', lambda _: ['93.184.216.34', '127.0.0.1'])
    backend = RecordingBackend()
    with pytest.raises(ValueError, match='private'):
        await PublicNetworkBackend(backend).connect_tcp('evil.example', 80)
    assert backend.hosts == []


@pytest.mark.parametrize('url', [
    'http://100.64.0.1/', 'http://[::ffff:127.0.0.1]/', 'http://user:pass@example.com/',
    'http://example.com:9000/', 'http://example.com\\@127.0.0.1/', 'http://example.com/\nfoo',
    'http://[fe80::1%25eth0]/', 'http://224.0.0.1/',
])
async def test_nonpublic_or_ambiguous_urls_blocked(url):
    assert await netguard.block_reason(url)


async def test_empty_dns_fails_closed(monkeypatch):
    monkeypatch.setattr(netguard, 'resolve_host', lambda _: [])
    assert await netguard.block_reason('https://example.com/')


class Chunks(httpx.AsyncByteStream):
    def __init__(self):
        self.reads = 0
        self.closed = False

    async def __aiter__(self):
        for _ in range(100):
            self.reads += 1
            yield b'x' * 65536

    async def aclose(self):
        self.closed = True


async def test_oversized_stream_stops_reading_and_closes():
    stream = Chunks()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(
            200, headers={'content-type':'text/plain'}, stream=stream))) as client:
        out = await Fetcher(SourceRegistry(), max_bytes=70000, client=client).fetch('https://example.com/')
        assert 'byte limit' in out and stream.reads == 2 and stream.closed
        assert not client.is_closed  # borrowed client ownership


async def test_compressed_response_rejected_before_decompression():
    stream = Chunks()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(
            200, headers={'content-type':'text/html', 'content-encoding':'gzip'}, stream=stream))) as client:
        out = await Fetcher(SourceRegistry(), client=client).fetch('https://example.com/')
    assert 'compressed' in out and stream.reads == 0 and stream.closed


async def test_redirect_guard_overrides_borrowed_clients_auto_redirects():
    requests = []
    def respond(req):
        requests.append(str(req.url))
        return httpx.Response(302, headers={'location':'http://127.0.0.1/'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond), follow_redirects=True) as client:
        out = await Fetcher(SourceRegistry(), client=client).fetch('https://example.com/')
    assert out.startswith('ERROR:') and requests == ['https://example.com/']


async def test_extractor_cancel_kills_worker(monkeypatch):
    class Process:
        returncode = None
        killed = waited = False
        async def communicate(self, data):
            await asyncio.sleep(60)
        def kill(self):
            self.killed = True
        async def wait(self):
            self.waited = True
            self.returncode = -9
    process = Process()
    async def spawn(*args, **kwargs): return process
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
    async with Fetcher(SourceRegistry(), extraction_timeout=.01) as fetcher:
        with pytest.raises(TimeoutError):
            await fetcher._extract(b'hello', 'text/plain')
    assert process.killed and process.waited


async def test_rebinding_between_preflight_and_dial_is_blocked(monkeypatch):
    calls = 0
    def dns(host):
        nonlocal calls
        calls += 1
        return ['93.184.216.34'] if calls == 1 else ['127.0.0.1']
    monkeypatch.setattr(netguard, 'resolve_host', dns)
    backend = RecordingBackend()
    async with httpx.AsyncClient(transport=PublicTransport(backend)) as client:
        out = await Fetcher(SourceRegistry(), client=client).fetch('https://rebind.example/')
    assert out.startswith('ERROR:') and calls == 2 and backend.hosts == []


@pytest.mark.parametrize('address', ['64:ff9b::7f00:1','2002:7f00:1::','2001::1'])
async def test_ipv6_translation_not_a_route_to_internal_ipv4(address):
    assert await netguard.block_reason(f'http://[{address}]/')
