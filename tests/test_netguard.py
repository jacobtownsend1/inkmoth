import pytest
from inkmoth import netguard

@pytest.mark.parametrize("url", [
    "http://127.0.0.1/", "http://localhost:8080/x", "http://10.0.0.5/", "http://192.168.1.100:8000/v1/models",
    "http://172.16.3.4/", "http://169.254.169.254/latest/meta-data", "http://[::1]/", "http://[fd8d:2a24::1]/",
    "http://0.0.0.0/", "http://searxng/search", "http://host.docker.internal/", "http://printer.local/",
    "http://vault.internal/", "ftp://example.com/", "file:///etc/passwd",
])
async def test_blocked(url):
    assert await netguard.block_reason(url) is not None

async def test_public_allowed(monkeypatch):
    monkeypatch.setattr(netguard, "resolve_host", lambda h: ["93.184.216.34"])
    assert await netguard.block_reason("https://example.com/page") is None

async def test_name_resolving_to_private_is_blocked(monkeypatch):
    monkeypatch.setattr(netguard, "resolve_host", lambda h: ["93.184.216.34", "192.168.1.9"])
    assert "private" in await netguard.block_reason("https://evil.example/")

async def test_unresolvable_is_blocked(monkeypatch):
    def boom(h):
        raise OSError("nxdomain")
    monkeypatch.setattr(netguard, "resolve_host", boom)
    assert "resolve" in await netguard.block_reason("https://nope.example/")

def test_looks_internal_syntactic():
    assert netguard.looks_internal("http://192.168.1.1/") and netguard.looks_internal("http://localhost/")
    assert not netguard.looks_internal("https://example.com/")
