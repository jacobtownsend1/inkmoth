from inkmoth.events import CollectingSink, ToolCalled, PhaseStarted

async def test_collecting_sink_keeps_order():
    sink = CollectingSink()
    await sink.emit(PhaseStarted(phase="plan"))
    await sink.emit(ToolCalled(agent="r1", tool="web_search", args={"query": "x"}))
    assert [type(e).__name__ for e in sink.events] == ["PhaseStarted", "ToolCalled"]
    assert sink.events[0].ts <= sink.events[1].ts
