"""Preflight: is the LLM up and tool-calling, does the search backend answer, which engines respond."""
from __future__ import annotations

import collections
import json
from dataclasses import dataclass

import httpx

from .config import Settings
from .search import make_backend


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


async def run_checks(settings: Settings) -> list[Check]:
    checks: list[Check] = []
    async with httpx.AsyncClient(timeout=60, trust_env=False) as c:
        auth = {"Authorization": f"Bearer {settings.llm_api_key}"}
        # 1 models
        try:
            r = await c.get(f"{settings.llm_base_url}/models", headers=auth)
            r.raise_for_status()
            ids = [m["id"] for m in r.json().get("data", [])]
            ok = settings.llm_model in ids
            checks.append(Check("LLM reachable", ok, f"{settings.llm_base_url} models={ids}"
                                + ("" if ok else f" (missing {settings.llm_model})")))
        except Exception as e:
            checks.append(Check("LLM reachable", False, f"{type(e).__name__}: {e}"))
        # 2 tool call
        try:
            body = {**settings.llm_extra_body_no_thinking, "model": settings.llm_model, "max_tokens": 200, "temperature": 0,
                    "messages": [{"role": "user", "content": "Search the web for today's weather in Paris."}],
                    "tools": [{"type": "function", "function": {
                        "name": "web_search", "description": "Search the web.",
                        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                                       "required": ["query"]}}}]}
            r = await c.post(f"{settings.llm_base_url}/chat/completions", json=body, headers=auth)
            r.raise_for_status()
            choice = r.json()["choices"][0]
            calls = choice["message"].get("tool_calls") or []
            ok = bool(calls) and calls[0]["function"]["name"] == "web_search"
            checks.append(Check("LLM tool calling", ok, json.dumps(calls[0]["function"]) if ok
                                else f"no tool call; finish={choice.get('finish_reason')}"))
        except Exception as e:
            checks.append(Check("LLM tool calling", False, f"{type(e).__name__}: {e}"))
        # 3 search backend
        name = f"Search ({settings.search_backend})"
        try:
            backend = make_backend(settings)
            try:
                resp = await backend.search("python programming language")
            finally:
                await backend.aclose()
            eng = collections.Counter(e for x in resp.results for e in x.engines)
            ok = len(resp.results) >= 3
            checks.append(Check(name, ok, f"{len(resp.results)} results"
                                + (f"; engines={dict(eng)}" if eng else "")
                                + (f"; unresponsive: {', '.join(resp.unresponsive)}" if resp.unresponsive else "")))
        except Exception as e:
            hint = " (SearXNG: is json in search.formats?)" if "403" in str(e) and settings.search_backend == "searxng" else ""
            checks.append(Check(name, False, f"{type(e).__name__}: {e}{hint}"))
    return checks
