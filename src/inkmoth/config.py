"""Runtime settings. Every knob is an env var with prefix INKMOTH_ (or a .env line)."""
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Run-size presets. Values override env/.env defaults; explicit per-run options override presets.
PRESETS: dict[str, dict] = {
    "quick": dict(breadth=2, depth=1, researcher_searches=4, max_searches=12, max_fetches=12,
                  max_model_calls=60, max_total_tokens=400_000, run_timeout=600.0),
    "standard": dict(breadth=4, depth=2, researcher_searches=6, max_searches=48, max_fetches=48,
                     max_model_calls=160, max_total_tokens=2_000_000, run_timeout=1800.0),
    "thorough": dict(breadth=6, depth=3, researcher_searches=8, max_searches=120, max_fetches=120,
                     max_model_calls=400, max_total_tokens=6_000_000, run_timeout=3600.0),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INKMOTH_", env_file=".env", extra="ignore", validate_assignment=True)

    # endpoints
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str  # required: no model name is a safe default across OpenAI-compatible servers
    llm_api_key: str = "EMPTY"
    search_backend: Literal["searxng", "brave", "tavily"] = "searxng"
    search_api_key: str = ""  # brave / tavily
    searxng_url: str = "http://localhost:8080"
    # merged into each chat request (JSON in env); empty = plain OpenAI API, thinking_* toggles become no-ops
    llm_extra_body_thinking: dict = {}
    llm_extra_body_no_thinking: dict = {}

    # research shape
    breadth: int = Field(4, ge=1, le=20)
    depth: int = Field(2, ge=1, le=10)
    parallel: int = Field(4, ge=1, le=32)
    researcher_searches: int = Field(6, ge=1, le=50)
    researcher_turns: int = Field(12, ge=1, le=100)
    researcher_timeout: float = Field(900.0, gt=0, allow_inf_nan=False)
    phase_timeout: float = Field(600.0, gt=0, allow_inf_nan=False)
    run_timeout: float = Field(1800.0, gt=0, allow_inf_nan=False)
    max_searches: int = Field(48, ge=1)
    max_fetches: int = Field(48, ge=1)
    max_model_calls: int = Field(160, ge=1)
    max_total_tokens: int = Field(2_000_000, ge=1)  # soft cap, checked before each model request
    search_max_results: int = Field(8, ge=1, le=30)
    fetch_max_chars: int = Field(6000, ge=2000, le=100000)  # per fetch_page call; long documents are navigated
    fetch_max_bytes: int = Field(2_000_000, ge=1024, le=20_000_000)  # HTML and text
    fetch_max_pdf_bytes: int = Field(50_000_000, ge=1024, le=200_000_000)
    fetch_timeout: float = Field(30.0, gt=0, allow_inf_nan=False)
    extraction_timeout: float = Field(60.0, gt=0, allow_inf_nan=False)  # also the worker's CPU limit; ~0.65 s/MB of PDF
    checkpoint_dir: Path | None = None  # no filesystem writes unless explicitly configured
    reports_dir: Path = Path("reports")  # CLI and MCP server only

    # model behaviour
    thinking_planner: bool = True
    thinking_researcher: bool = False
    thinking_synthesizer: bool = False  # on big inputs its thinking can run past the token cap; enable per run if wanted
    review_synthesis: bool = True  # separate, tool-free semantic review; still a model judgment
    repair_evidence: bool = True  # one bounded retry when exact quote validation rejects facts
    temperature: float = Field(0.3, ge=0, le=2)
    max_tokens: int = Field(8192, ge=128)
    synth_max_tokens: int = Field(16384, ge=128)
    request_timeout: float = Field(600.0, gt=0, allow_inf_nan=False)

    @field_validator("llm_base_url", "searxng_url")
    @classmethod
    def endpoint(cls, value: str) -> str:
        from urllib.parse import urlsplit
        p = urlsplit(value)
        if p.scheme not in {"http", "https"} or not p.hostname or p.username or p.password or p.query or p.fragment:
            raise ValueError("endpoint must be an HTTP(S) URL without credentials, query or fragment")
        _ = p.port
        return value.rstrip("/")

    @model_validator(mode="after")
    def search_key(self):
        if self.search_backend != "searxng" and not self.search_api_key:
            raise ValueError(f"search_backend {self.search_backend!r} needs INKMOTH_SEARCH_API_KEY")
        return self
