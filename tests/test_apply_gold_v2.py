"""Writing a sealed gold v2 into the dataset, and refusing to when it is not.

The step itself has not been run: A2 and A3 are not back, so no gold v2 exists
and `golden_dataset.json` is still at v1.1. These tests are what stands in for
having run it -- every fixture is synthetic and lives in tmp_path, and the real
dataset is never opened.

`labeled_at` was the fourth situation the 2026-09-19 entries left undecided and
was ruled the same day: the cases a panel decided take the date gold v2 was
sealed, and the cases that kept their shipped label keep the date that label
already carried, because nothing about them was decided again.
"""
from __future__ import annotations

import json

import pytest

from experiments import apply_gold_v2
from tests.test_annotation_import import make_dataset

ORIGINAL_DATE = "2026-07-20"
SEAL_DATE = "2026-09-24"


@pytest.fixture
def dataset(tmp_path):
    path = make_dataset(tmp_path, n=4)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for case in payload["cases"]:
        case["labeled_at"] = ORIGINAL_DATE
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_gold(tmp_path, routes, labels=None, sealed=True, sealed_on=SEAL_DATE):
    labels = labels or {}
    payload = {
        "gold_version": "v2",
        "regime": "main",
        "regime_basis": "all three sheets have arrived",
        "sealed": sealed,
        "sealed_on": sealed_on,
        "adjudication_pending_case_ids": [] if sealed else ["case-000"],
        "labels": [
            {
                "case_id": case_id,
                "label": labels.get(case_id, "billing"),
                "labeled_by": route,
            }
            for case_id, route in sorted(routes.items())
        ],
    }
    path = tmp_path / "gold_v2.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def cases_of(dataset):
    return {case["id"]: case for case in json.loads(dataset.read_text(encoding="utf-8"))["cases"]}


def test_a_dry_run_writes_nothing(dataset, tmp_path):
    gold = write_gold(tmp_path, {f"case-{i:03d}": "majority" for i in range(4)})
    before = dataset.read_text(encoding="utf-8")
    assert apply_gold_v2.main(["--gold", str(gold), "--dataset", str(dataset)]) == 0
    assert dataset.read_text(encoding="utf-8") == before


def test_labeled_at_moves_only_on_the_cases_a_panel_decided(dataset, tmp_path):
    gold = write_gold(
        tmp_path,
        {
            "case-000": "majority",
            "case-001": "adjudicated",
            "case-002": "fallback_v1.1",
            "case-003": "majority",
        },
        labels={"case-000": "technical", "case-001": "account",
                "case-002": "account", "case-003": "general"},
    )
    assert apply_gold_v2.main([
        "--gold", str(gold), "--dataset", str(dataset), "--write",
    ]) == 0
    cases = cases_of(dataset)
    assert cases["case-000"]["labeled_at"] == SEAL_DATE
    assert cases["case-001"]["labeled_at"] == SEAL_DATE
    assert cases["case-003"]["labeled_at"] == SEAL_DATE
    assert cases["case-002"]["labeled_at"] == ORIGINAL_DATE
    # And the three fields the dataset paragraph names moved on every case.
    assert cases["case-000"]["expected_category"] == "technical"
    assert cases["case-002"]["labeled_by"] == "fallback_v1.1"
    assert json.loads(dataset.read_text(encoding="utf-8"))["dataset_version"] == "v2.0"


def test_nothing_but_the_four_fields_moves(dataset, tmp_path):
    before = cases_of(dataset)
    gold = write_gold(tmp_path, {f"case-{i:03d}": "majority" for i in range(4)})
    apply_gold_v2.main(["--gold", str(gold), "--dataset", str(dataset), "--write"])
    after = cases_of(dataset)
    moveable = {"expected_category", "labeled_by", "labeled_at"}
    for case_id, case in after.items():
        untouched = {k: v for k, v in case.items() if k not in moveable}
        assert untouched == {k: v for k, v in before[case_id].items() if k not in moveable}


def test_an_unsealed_gold_is_refused(dataset, tmp_path):
    gold = write_gold(tmp_path, {f"case-{i:03d}": "majority" for i in range(4)}, sealed=False)
    before = dataset.read_text(encoding="utf-8")
    assert apply_gold_v2.main(["--gold", str(gold), "--dataset", str(dataset), "--write"]) == 1
    assert dataset.read_text(encoding="utf-8") == before


def test_a_sealed_gold_with_no_seal_date_is_refused(dataset, tmp_path):
    """There would be nothing to write labeled_at from, and guessing a date is
    the thing the ruling replaced."""
    gold = write_gold(tmp_path, {f"case-{i:03d}": "majority" for i in range(4)}, sealed_on=None)
    before = dataset.read_text(encoding="utf-8")
    assert apply_gold_v2.main(["--gold", str(gold), "--dataset", str(dataset), "--write"]) == 1
    assert dataset.read_text(encoding="utf-8") == before


def test_an_unknown_decision_route_is_refused(dataset, tmp_path):
    gold = write_gold(tmp_path, {"case-000": "majority", "case-001": "majority",
                                 "case-002": "author_said_so", "case-003": "majority"})
    before = dataset.read_text(encoding="utf-8")
    assert apply_gold_v2.main(["--gold", str(gold), "--dataset", str(dataset), "--write"]) == 1
    assert dataset.read_text(encoding="utf-8") == before


def test_a_case_with_no_gold_v2_label_is_refused(dataset, tmp_path):
    gold = write_gold(tmp_path, {f"case-{i:03d}": "majority" for i in range(3)})
    before = dataset.read_text(encoding="utf-8")
    assert apply_gold_v2.main(["--gold", str(gold), "--dataset", str(dataset), "--write"]) == 1
    assert dataset.read_text(encoding="utf-8") == before
