"""llm.py: tier->model mapping, provider switching, and Anthropic tool-use
parsing. All Anthropic SDK calls are mocked -- no ANTHROPIC_API_KEY needed
or read (AC1)."""
from __future__ import annotations

import anthropic
from pydantic import BaseModel

import llm


class _EchoSchema(BaseModel):
    category: str
    summary: str


class _FakeToolUseBlock:
    def __init__(self, input_dict: dict):
        self.type = "tool_use"
        self.input = input_dict


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeAnthropicResponse:
    def __init__(self, input_dict: dict, input_tokens: int = 111, output_tokens: int = 22):
        self.content = [_FakeToolUseBlock(input_dict)]
        self.usage = _FakeUsage(input_tokens, output_tokens)


class _FakeMessages:
    def __init__(self, response: _FakeAnthropicResponse):
        self._response = response
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _FakeAnthropicClient:
    def __init__(self, *args, **kwargs):
        self.messages = _FakeMessages(_FakeAnthropicResponse({"category": "billing", "summary": "test"}))


def test_model_for_default_anthropic(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_CLASSIFIER_MODEL", raising=False)
    monkeypatch.delenv("LLM_JUDGE_MODEL", raising=False)
    assert llm._model_for("classifier") == "claude-haiku-4-5"
    assert llm._model_for("judge") == "claude-sonnet-5"


def test_model_for_env_override(monkeypatch):
    monkeypatch.setenv("LLM_CLASSIFIER_MODEL", "claude-haiku-custom")
    assert llm._model_for("classifier") == "claude-haiku-custom"


def test_model_for_openai_provider_defaults(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("LLM_CLASSIFIER_MODEL", raising=False)
    assert llm._model_for("classifier") == "gpt-4o-mini"
    assert llm._model_for("judge") == "gpt-4o"


def test_model_for_unknown_tier_raises(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    try:
        llm._model_for("does-not-exist")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_complete_anthropic_happy_path(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # AC1: no key needed, fully mocked
    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropicClient)

    resp = llm.complete(
        tier="classifier",
        system="you are a classifier",
        messages=[{"role": "user", "content": "hello"}],
        schema=_EchoSchema,
    )

    assert isinstance(resp, llm.LLMResponse)
    assert resp.parsed == _EchoSchema(category="billing", summary="test")
    assert resp.model == "claude-haiku-4-5"
    assert resp.input_tokens == 111
    assert resp.output_tokens == 22
    assert resp.latency_ms >= 0


def test_complete_unsupported_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "does-not-exist")
    try:
        llm.complete(tier="classifier", system="s", messages=[], schema=_EchoSchema)
        assert False, "expected ValueError"
    except ValueError:
        pass
