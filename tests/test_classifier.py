"""classifier.py: prompt loading + classify_email(), fully mocked at the
llm.complete() boundary -- no network, no API key (AC1)."""
from __future__ import annotations

import pytest

import classifier
from llm import LLMResponse


def _fake_llm_response(category="billing", summary="test summary") -> LLMResponse:
    return LLMResponse(
        parsed=classifier.ClassificationResult(category=category, summary=summary),
        model="claude-haiku-4-5",
        input_tokens=123,
        output_tokens=17,
        latency_ms=250.0,
    )


def test_load_prompt_config_v1_loads_and_matches_shape():
    cfg = classifier.load_prompt_config("v1")
    assert cfg.version == "v1"
    assert cfg.model == "claude-haiku-4-5"
    assert len(cfg.few_shot_examples) >= 1
    assert cfg.output_schema["category"].startswith("enum[")


def test_load_prompt_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        classifier.load_prompt_config("v999-does-not-exist")


def test_load_prompt_config_version_mismatch_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(classifier, "PROMPTS_DIR", tmp_path)
    (tmp_path / "v2.yaml").write_text(
        "version: v1\nmodel: claude-haiku-4-5\ncreated_at: '2026-01-01'\n"
        "system_prompt: x\nfew_shot_examples: []\noutput_schema: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected 'v2'"):
        classifier.load_prompt_config("v2")


def test_classify_email_returns_parsed_result_with_llm_response(monkeypatch):
    fake_resp = _fake_llm_response(category="technical", summary="a technical issue")
    monkeypatch.setattr(classifier, "complete", lambda **kwargs: fake_resp)

    cfg = classifier.load_prompt_config("v1")
    result = classifier.classify_email("my API integration is broken", cfg)

    assert result.category == "technical"
    assert result.summary == "a technical issue"
    assert result.llm_response is fake_resp


def test_classify_email_empty_text_raises():
    cfg = classifier.load_prompt_config("v1")
    with pytest.raises(ValueError):
        classifier.classify_email("   ", cfg)


def test_classify_email_public_shape_excludes_llm_response(monkeypatch):
    """ClassificationResult's public contract stays {category, summary} exactly
    as specced -- llm_response is a private attribute, not a schema field."""
    fake_resp = _fake_llm_response()
    monkeypatch.setattr(classifier, "complete", lambda **kwargs: fake_resp)
    cfg = classifier.load_prompt_config("v1")
    result = classifier.classify_email("some email", cfg)
    assert set(result.model_dump().keys()) == {"category", "summary"}
