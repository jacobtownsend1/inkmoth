"""HTTPX transport using HTTPCore's public network-backend interface.

DNS is resolved and checked immediately before dialing a numeric address. HTTPCore
retains the original origin for connection pooling, Host, SNI and certificate checks.
No environment proxies, alternate TLS hostname, Unix sockets or caller auth are used.
"""
import asyncio

import httpcore
import httpx

from .netguard import public_addresses, validate_url


class PublicNetworkBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, backend=None):
        self.backend = backend or httpcore.AnyIOBackend()

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        async with asyncio.timeout(timeout or 20):
            addresses = await public_addresses(host.rstrip("."))
            last_error = None
            for address in addresses:
                try:
                    return await self.backend.connect_tcp(address, port, timeout=timeout,
                                                          local_address=local_address, socket_options=socket_options)
                except (OSError, httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                    last_error = exc
            raise httpcore.ConnectError("all validated addresses failed") from last_error

    async def sleep(self, seconds):
        await asyncio.sleep(seconds)


class _Stream(httpx.AsyncByteStream):
    def __init__(self, stream):
        self.stream = stream

    async def __aiter__(self):
        async for chunk in self.stream:
            yield chunk

    async def aclose(self):
        await self.stream.aclose()


class PublicTransport(httpx.AsyncBaseTransport):
    def __init__(self, backend=None):
        self.pool = httpcore.AsyncConnectionPool(network_backend=PublicNetworkBackend(backend),
                                                max_connections=16, max_keepalive_connections=8)

    async def handle_async_request(self, request):
        validate_url(str(request.url))
        if request.method != "GET":
            raise ValueError("public fetch transport permits GET only")
        req = httpcore.Request(
            method=request.method,
            url=httpcore.URL(scheme=request.url.raw_scheme, host=request.url.raw_host,
                             port=request.url.port, target=request.url.raw_path),
            headers=request.headers.raw, content=request.stream,
            extensions={"timeout": request.extensions.get("timeout", {})},
        )
        response = await self.pool.handle_async_request(req)
        return httpx.Response(response.status, headers=response.headers,
                              stream=_Stream(response.stream), extensions=response.extensions)

    async def aclose(self):
        await self.pool.aclose()
