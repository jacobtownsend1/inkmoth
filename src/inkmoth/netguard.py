"""Decide whether a URL may be fetched. Agents read untrusted pages that can tell the model to fetch
internal addresses; everything on loopback, private, link-local or otherwise non-public ranges is refused,
as are local-only hostnames. Redirect hops go through the same check (see fetch.py)."""
from __future__ import annotations

import asyncio
import ipaddress
import socket
import httpx

_LOCAL_SUFFIXES = (".local", ".internal", ".localhost", ".localdomain", ".home", ".lan", ".arpa")
_LOCAL_NAMES = {"localhost", "host.docker.internal", "metadata.google.internal"}
_IPV6_TRANSLATION = tuple(ipaddress.ip_network(n) for n in ("64:ff9b::/96", "64:ff9b:1::/48", "2002::/16", "2001::/32"))


def resolve_host(host: str) -> list[str]:
    """All A/AAAA addresses for a hostname. Module-level so tests can replace it."""
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return sorted({info[4][0] for info in infos})


def _bad_ip(ip: ipaddress._BaseAddress) -> bool:
    return (not ip.is_global or ip.is_multicast or ip.is_reserved
            or getattr(ip, "is_site_local", False)
            or (ip.version == 6 and any(ip in network for network in _IPV6_TRANSLATION))
            or bool(getattr(ip, "ipv4_mapped", None) and _bad_ip(ip.ipv4_mapped)))


def validate_url(url: str) -> httpx.URL:
    """Use the same parser as the HTTP client; reject credentials and ambiguous names."""
    if any(ord(c) < 33 or ord(c) == 127 for c in url) or "\\" in url:
        raise ValueError("URL contains whitespace, controls or backslashes")
    p = httpx.URL(url)
    host = p.raw_host.decode("ascii").lower().rstrip(".")
    if p.scheme not in {"http", "https"} or not host:
        raise ValueError("only absolute HTTP(S) URLs are allowed")
    if p.userinfo or p.port not in (None, 80, 443):
        raise ValueError("URL credentials and nonstandard ports are not allowed")
    if "%" in host or host in _LOCAL_NAMES or host.endswith(_LOCAL_SUFFIXES) or ("." not in host and ":" not in host):
        raise ValueError(f"local hostname {host!r} is not allowed")
    return p


async def public_addresses(host: str) -> list[str]:
    try:
        addrs = [str(ipaddress.ip_address(host))]
    except ValueError:
        async with asyncio.timeout(5):
            addrs = await asyncio.to_thread(resolve_host, host)
    if not addrs:
        raise ValueError("DNS returned no addresses")
    for addr in addrs:
        if _bad_ip(ipaddress.ip_address(addr)):
            raise ValueError(f"host {host!r} resolves to private/non-public address {addr}")
    return addrs


def looks_internal(url: str) -> bool:
    """Cheap syntactic check (no DNS): local names, bare names, or literal non-public IPs."""
    try:
        p = validate_url(url)
        host = p.raw_host.decode("ascii").rstrip(".")
    except (ValueError, httpx.InvalidURL):
        return True  # also local names and bare hostnames, which validate_url rejects
    try:
        return _bad_ip(ipaddress.ip_address(host))
    except ValueError:
        return False  # a normal hostname; DNS check happens in block_reason


async def block_reason(url: str) -> str | None:
    """None if the URL may be fetched, else a short reason."""
    try:
        p = validate_url(url)
        await public_addresses(p.raw_host.decode("ascii").rstrip("."))
    except (OSError, ValueError, httpx.InvalidURL, TimeoutError) as e:
        return f"cannot resolve or allow URL ({e})"
    return None
