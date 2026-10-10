"""One run's dependencies and ownership, independent of the model adapter."""
from uuid import uuid4

from .budget import RunBudget
from .config import Settings
from .events import EventSink, NullSink, ToolCalled
from .fetch import Fetcher
from .search import make_backend
from .sources import SourceRegistry


class RunContext:
    def __init__(self, settings: Settings, sink: EventSink | None = None,
                 registry: SourceRegistry | None = None):
        self.settings = Settings.model_validate(settings.model_dump())
        self.registry = registry if registry is not None else SourceRegistry()
        self.sink = sink if sink is not None else NullSink()
        self.run_id = uuid4().hex
        self.budget = RunBudget(self.settings)
        self.tool_calls = self.seq = 0
        self._search = self._fetch = None
        self.search_cache = {}
        self.search_locks = {}
        self.assessment = None
        self.review = None
        self.question = ""
        self.research_notes = ""  # application-generated guidance for researchers; never model or page text
        self.assumptions: list[str] = []
        self.clarifications = ""  # the user's answers to clarifying questions
        self.closed = self.started = False

    def claim(self):
        if self.started or self.closed:
            raise ValueError("RunContext is single-use; create a new context for each run")
        self.started = True

    @property
    def search(self):
        if self.closed:
            raise RuntimeError("context is closed")
        if self._search is None:
            self._search = make_backend(self.settings)
        return self._search

    @property
    def fetch(self):
        if self.closed:
            raise RuntimeError("context is closed")
        if self._fetch is None:
            s = self.settings
            self._fetch = Fetcher(self.registry, max_chars=s.fetch_max_chars, timeout=s.fetch_timeout,
                                  max_bytes=s.fetch_max_bytes, max_pdf_bytes=s.fetch_max_pdf_bytes,
                                  extraction_timeout=s.extraction_timeout,
                                  budget=self.budget)
        return self._fetch

    async def emit(self, event):
        self.seq += 1
        event.run_id, event.seq = self.run_id, self.seq
        if isinstance(event, ToolCalled):
            self.tool_calls += 1
        await self.sink.emit(event)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    async def aclose(self):
        try:
            if self._search is not None:
                await self._search.aclose()
        finally:
            try:
                if self._fetch is not None:
                    await self._fetch.aclose()
            finally:
                self.closed = True
