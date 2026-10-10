import pytest
from inkmoth.config import Settings

@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(reports_dir=tmp_path / "reports", _env_file=None)


@pytest.fixture(autouse=True)
def _model(monkeypatch, request):
    if request.node.get_closest_marker("live") is None:  # live tests use the configured model
        monkeypatch.setenv("INKMOTH_LLM_MODEL", "test-model")


@pytest.fixture(autouse=True)
def _public_dns(monkeypatch, request):
    """Unit tests use *.test hostnames; pretend they resolve to a public address."""
    from inkmoth import netguard
    if request.node.get_closest_marker("live") is None:
        monkeypatch.setattr(netguard, "resolve_host", lambda host: ["93.184.216.34"])
