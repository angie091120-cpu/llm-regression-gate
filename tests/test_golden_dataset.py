"""Structural integrity checks on the shipped golden_dataset.json draft
(SPEC.md §3.1(c)). These guard the W1 deliverable itself, independent of any
LLM call: schema validity, id uniqueness, size bounds, and the language
quota (the one hard percentage split in the spec)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from evalkit.dataset import GoldenCase

DATASET_PATH = Path(__file__).parent.parent / "golden_dataset.json"


@pytest.fixture(scope="module")
def dataset() -> dict:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def test_dataset_file_has_expected_envelope(dataset):
    assert dataset["dataset_version"] == "v1"
    assert isinstance(dataset["cases"], list)


def test_case_count_within_ac2_bounds(dataset):
    # AC2 requires >= 60; spec caps the intended range at 80 (§7 assumption 1)
    assert 60 <= len(dataset["cases"]) <= 80


def test_every_case_validates_against_schema(dataset):
    for raw in dataset["cases"]:
        GoldenCase(**raw)  # raises if a required field is missing/mistyped


def test_case_ids_are_unique(dataset):
    ids = [c["id"] for c in dataset["cases"]]
    assert len(ids) == len(set(ids))


def test_language_quota_matches_spec_50_40_10(dataset):
    total = len(dataset["cases"])
    counts = Counter(c["language"] for c in dataset["cases"])
    assert counts["zh-tw"] / total == pytest.approx(0.50, abs=0.01)
    assert counts["en"] / total == pytest.approx(0.40, abs=0.01)
    assert counts["mixed"] / total == pytest.approx(0.10, abs=0.01)


def test_all_cases_are_draft_and_unlabeled(dataset):
    """W1 invariant: labeling is the boss's manual step, not the agent's."""
    for c in dataset["cases"]:
        assert c["label_status"] == "draft"
        assert c["expected_category"] is None
        assert c["expected_summary"] is None


def test_every_case_has_notes(dataset):
    for c in dataset["cases"]:
        assert c["notes"], f"case {c['id']} missing notes"


def test_categories_are_valid_enum_values(dataset):
    valid = {"billing", "technical", "account", "general"}
    for c in dataset["cases"]:
        assert c["draft_category"] in valid, f"case {c['id']} has invalid draft_category"
