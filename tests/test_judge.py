"""evalkit/judge.py: LLM-as-judge scoring, mocked at the llm.complete()
boundary -- no network, no API key (AC1)."""
from __future__ import annotations

from evalkit import judge
from llm import LLMResponse


def test_score_summary_returns_score_and_response(monkeypatch):
    fake_resp = LLMResponse(
        parsed=judge.JudgeVerdict(score=4, reasoning="close enough"),
        model="claude-sonnet-5",
        input_tokens=200,
        output_tokens=15,
        latency_ms=300.0,
    )
    captured = {}

    def fake_complete(**kwargs):
        captured.update(kwargs)
        return fake_resp

    monkeypatch.setattr(judge, "complete", fake_complete)

    score, resp = judge.score_summary("original email", "expected summary", "actual summary")

    assert score == 4
    assert resp is fake_resp
    assert captured["tier"] == "judge"
    assert captured["schema"] is judge.JudgeVerdict
    assert "expected summary" in captured["messages"][0]["content"]
    assert "actual summary" in captured["messages"][0]["content"]


def test_judge_verdict_score_bounds():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        judge.JudgeVerdict(score=6, reasoning="out of range")
    with pytest.raises(ValidationError):
        judge.JudgeVerdict(score=0, reasoning="out of range")
