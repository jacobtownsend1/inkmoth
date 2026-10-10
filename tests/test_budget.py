import asyncio

from inkmoth.budget import RunBudget
from inkmoth.search import SearchBudget, SearchResponse, make_web_search_tool
from inkmoth.sources import SourceRegistry


async def test_parallel_researchers_share_search_budget(settings):
    settings.max_searches = 2
    budget = RunBudget(settings)
    class Client:
        calls = 0
        async def search(self, query, **kwargs):
            self.calls += 1
            await asyncio.sleep(.01)
            return SearchResponse(query=query, results=[])
    client = Client()
    tools = [make_web_search_tool(client, SourceRegistry(), SearchBudget(6), 8, run_budget=budget)
             for _ in range(8)]
    results = await asyncio.gather(*(t.original_function(str(i)) for i,t in enumerate(tools)))
    assert client.calls == budget.searches == 2
    assert sum('budget exhausted' in r for r in results) == 6


async def test_duplicate_concurrent_searches_share_one_request(settings):
    class Client:
        calls = 0
        async def search(self, query, **kwargs):
            self.calls += 1
            await asyncio.sleep(.01)
            return SearchResponse(query=query,results=[])
    client, cache, locks = Client(), {}, {}
    tools = [make_web_search_tool(client, SourceRegistry(), SearchBudget(2), 8, cache=cache, locks=locks)
             for _ in range(3)]
    await asyncio.gather(*(t.original_function('same query') for t in tools))
    assert client.calls == 1
