from pathlib import Path
from inkmoth.config import Settings

def test_defaults():
    s = Settings(_env_file=None)
    assert s.llm_base_url == "http://localhost:8000/v1"
    assert s.llm_model == "test-model"
    assert s.searxng_url == "http://localhost:8080"
    assert (s.breadth, s.depth, s.parallel) == (4, 2, 4)
    assert s.researcher_searches == 6
    assert s.thinking_planner and not s.thinking_synthesizer and not s.thinking_researcher
    assert s.reports_dir == Path("reports")

def test_model_is_required(monkeypatch):
    import pytest
    from pydantic import ValidationError
    monkeypatch.delenv("INKMOTH_LLM_MODEL")
    with pytest.raises(ValidationError, match="llm_model"):
        Settings(_env_file=None)


def test_env_override(monkeypatch):
    monkeypatch.setenv("INKMOTH_SEARXNG_URL", "http://10.0.0.5:9999")
    monkeypatch.setenv("INKMOTH_PARALLEL", "2")
    s = Settings(_env_file=None)
    assert s.searxng_url == "http://10.0.0.5:9999"
    assert s.parallel == 2


def test_extra_body_from_json_env(monkeypatch):
    monkeypatch.setenv("INKMOTH_LLM_EXTRA_BODY_THINKING", '{"chat_template_kwargs": {"enable_thinking": true}}')
    assert Settings(_env_file=None).llm_extra_body_thinking == {"chat_template_kwargs": {"enable_thinking": True}}


def test_standard_preset_is_the_defaults_and_presets_validate():
    from inkmoth.config import PRESETS
    s = Settings(_env_file=None)
    assert {k: getattr(s, k) for k in PRESETS["standard"]} == PRESETS["standard"]
    assert set(PRESETS["quick"]) == set(PRESETS["standard"]) == set(PRESETS["thorough"])
    for values in PRESETS.values():
        Settings(_env_file=None, **values)
