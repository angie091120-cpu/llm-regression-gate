"""evalkit/dataset.py: confirmed-only filtering, loud warnings, fail-loud on
malformed confirmed cases (see SPEC.md §3.1(c) step 3)."""
from __future__ import annotations

import json

import pytest

from evalkit.dataset import load_all_cases, load_confirmed_cases

CONFIRMED_CASE = {
    "id": "case-001",
    "language": "en",
    "input_text": "hello",
    "draft_category": "billing",
    "draft_summary": "draft summary",
    "draft_difficulty": "easy",
    "expected_category": "billing",
    "expected_summary": "confirmed summary",
    "expected_difficulty": "easy",
    "notes": "n/a",
    "label_status": "confirmed",
    "labeled_by": "boss",
    "labeled_at": "2026-07-20T00:00:00Z",
}

DRAFT_CASE = {
    "id": "case-002",
    "language": "zh-tw",
    "input_text": "你好",
    "draft_category": "technical",
    "draft_summary": "draft summary 2",
    "draft_difficulty": "edge",
    "expected_category": None,
    "expected_summary": None,
    "expected_difficulty": None,
    "notes": "n/a",
    "label_status": "draft",
    "labeled_by": None,
    "labeled_at": None,
}


def _write_dataset(tmp_path, cases, wrapped=True):
    path = tmp_path / "dataset.json"
    payload = {"dataset_version": "v1", "cases": cases} if wrapped else cases
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_all_cases_wrapped_format(tmp_path):
    path = _write_dataset(tmp_path, [CONFIRMED_CASE, DRAFT_CASE], wrapped=True)
    cases = load_all_cases(path)
    assert len(cases) == 2
    assert {c.id for c in cases} == {"case-001", "case-002"}


def test_load_all_cases_plain_list_format(tmp_path):
    path = _write_dataset(tmp_path, [CONFIRMED_CASE], wrapped=False)
    cases = load_all_cases(path)
    assert len(cases) == 1


def test_load_confirmed_cases_filters_and_warns(tmp_path):
    path = _write_dataset(tmp_path, [CONFIRMED_CASE, DRAFT_CASE])
    with pytest.warns(UserWarning, match="case-002"):
        confirmed = load_confirmed_cases(path)
    assert len(confirmed) == 1
    assert confirmed[0].id == "case-001"


def test_load_confirmed_cases_no_draft_cases_no_warning(tmp_path):
    path = _write_dataset(tmp_path, [CONFIRMED_CASE])
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        confirmed = load_confirmed_cases(path)
    assert len(confirmed) == 1


def test_load_confirmed_cases_fails_loud_on_incomplete_confirmed_case(tmp_path):
    broken = dict(CONFIRMED_CASE)
    broken["expected_summary"] = None  # confirmed but incomplete -- must never evaluate against this
    path = _write_dataset(tmp_path, [broken])
    with pytest.raises(ValueError, match="expected_category/expected_summary"):
        load_confirmed_cases(path)


def test_current_golden_dataset_has_zero_confirmed_cases():
    """As of W1, golden_dataset.json is entirely draft -- confirming labels is
    the boss's manual step (SPEC.md §3.1(c) step 2). This test documents that
    invariant and will start failing the moment real labels land, which is
    exactly the signal we want."""
    from pathlib import Path

    dataset_path = Path(__file__).parent.parent / "golden_dataset.json"
    with pytest.warns(UserWarning):
        confirmed = load_confirmed_cases(dataset_path)
    assert confirmed == []
