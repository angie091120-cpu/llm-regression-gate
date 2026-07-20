"""evalkit/run_eval.py: end-to-end eval loop with classify_email() and
score_summary() both mocked -- no network, no API key (AC1)."""
from __future__ import annotations

import json

import pytest

import classifier
from evalkit import run_eval as run_eval_module
from llm import LLMResponse

CASES = [
    {
        "id": "case-001",
        "language": "en",
        "input_text": "my invoice is wrong",
        "draft_category": "billing",
        "draft_summary": "draft",
        "draft_difficulty": "easy",
        "expected_category": "billing",
        "expected_summary": "Customer flags a billing invoice error.",
        "expected_difficulty": "easy",
        "notes": "n/a",
        "label_status": "confirmed",
        "labeled_by": "boss",
        "labeled_at": "2026-07-20T00:00:00Z",
    },
    {
        "id": "case-002",
        "language": "en",
        "input_text": "I can't log in",
        "draft_category": "account",
        "draft_summary": "draft",
        "draft_difficulty": "easy",
        "expected_category": "account",
        "expected_summary": "Customer can't log in.",
        "expected_difficulty": "easy",
        "notes": "n/a",
        "label_status": "confirmed",
        "labeled_by": "boss",
        "labeled_at": "2026-07-20T00:00:00Z",
    },
    {
        "id": "case-003",
        "language": "en",
        "input_text": "not yet reviewed",
        "draft_category": "general",
        "draft_summary": "draft",
        "draft_difficulty": "easy",
        "expected_category": None,
        "expected_summary": None,
        "expected_difficulty": None,
        "notes": "n/a",
        "label_status": "draft",
        "labeled_by": None,
        "labeled_at": None,
    },
]


def _write_dataset(tmp_path, cases=CASES):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({"dataset_version": "v1", "cases": cases}), encoding="utf-8")
    return path


def _make_classify_stub(outcomes: dict[str, tuple[str, str]]):
    """outcomes: input_text substring -> (category, summary)."""

    def fake_classify_email(email_text, prompt_config):
        for key, (category, summary) in outcomes.items():
            if key in email_text:
                result = classifier.ClassificationResult(category=category, summary=summary)
                result._llm_response = LLMResponse(
                    parsed=result, model="claude-haiku-4-5", input_tokens=100, output_tokens=10, latency_ms=200.0
                )
                return result
        raise AssertionError(f"no stub outcome for {email_text!r}")

    return fake_classify_email


def _make_judge_stub(scores: dict[str, int]):
    def fake_score_summary(input_text, expected_summary, actual_summary):
        for key, score in scores.items():
            if key in input_text:
                resp = LLMResponse(
                    parsed=None, model="claude-sonnet-5", input_tokens=50, output_tokens=5, latency_ms=80.0
                )
                return score, resp
        raise AssertionError(f"no judge stub for {input_text!r}")

    return fake_score_summary


def test_run_eval_computes_pass_rate_and_writes_ledger(tmp_path, monkeypatch):
    dataset_path = _write_dataset(tmp_path)
    ledger_path = tmp_path / "cost_ledger.json"

    monkeypatch.setattr(
        run_eval_module,
        "classify_email",
        _make_classify_stub(
            {
                "invoice": ("billing", "correct billing summary"),
                "log in": ("account", "correct account summary"),
            }
        ),
    )
    monkeypatch.setattr(
        run_eval_module,
        "score_summary",
        _make_judge_stub({"invoice": 5, "log in": 2}),  # case-002 fails the judge threshold
    )

    report = run_eval_module.run_eval("v1", dataset_path, cost_ledger_path=ledger_path)

    # case-003 is draft -> excluded; only case-001/002 evaluated
    assert len(report["cases"]) == 2
    ids = {c["id"] for c in report["cases"]}
    assert ids == {"case-001", "case-002"}

    by_id = {c["id"]: c for c in report["cases"]}
    assert by_id["case-001"]["passed"] is True
    assert by_id["case-001"]["category_match"] is True
    assert by_id["case-002"]["passed"] is False  # judge_score 2 < threshold 3
    assert by_id["case-002"]["category_match"] is True

    assert report["pass_rate"] == pytest.approx(0.5)
    assert report["per_category_accuracy"]["billing"] == pytest.approx(1.0)
    assert report["per_category_accuracy"]["account"] == pytest.approx(0.0)
    assert "cases" in report and "pass_rate" in report and "model" in report

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["cumulative_usd"] > 0


def test_run_eval_raises_when_no_confirmed_cases(tmp_path):
    dataset_path = _write_dataset(tmp_path, cases=[CASES[2]])  # only the draft one
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match="no label_status='confirmed' cases"):
            run_eval_module.run_eval("v1", dataset_path)


def test_run_eval_limit_flag(tmp_path, monkeypatch):
    dataset_path = _write_dataset(tmp_path)
    monkeypatch.setattr(
        run_eval_module,
        "classify_email",
        _make_classify_stub({"invoice": ("billing", "x"), "log in": ("account", "y")}),
    )
    monkeypatch.setattr(run_eval_module, "score_summary", _make_judge_stub({"invoice": 5, "log in": 5}))
    report = run_eval_module.run_eval("v1", dataset_path, limit=1, cost_ledger_path=tmp_path / "ledger.json")
    assert len(report["cases"]) == 1
