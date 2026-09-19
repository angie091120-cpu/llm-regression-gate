"""Every branch of the gold v2 decision rule, on synthetic returned sheets.

A2 and A3 are not back yet, so there is no real three-annotator data to run
these rules on -- and running them on A1's real sheet is not an option:
docs/PREREGISTRATION.md section 9 fixes the rules before any sheet is compared
with anything, and trying the rules out on the one sheet that has arrived is a
comparison. Every fixture here is invented.

The branches, from the section 9 entry dated 2026-09-19:

    all three sheets in     majority of three / all three differ -> author
                            two valid votes agreeing / two differing
                            fewer than two valid votes -> not covered
    fallback 1              the two humans agree / they split and v1.1 breaks
                            the tie / all three differ -> author / only A1's
                            vote left / only the other human's vote -> not
                            covered
    fallback 2              nothing is written
    before the deadline     nothing is written
    a late sheet            does not reopen a gold produced under a fallback
"""
from __future__ import annotations

import csv
import json

import pytest

from experiments import gold_v2, import_annotations
from tests.test_annotation_import import make_dataset

SHIPPED = ["billing", "technical", "account", "general", "billing", "technical"]


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    path = make_dataset(tmp_path, n=len(SHIPPED))
    payload = json.loads(path.read_text(encoding="utf-8"))
    for case, label in zip(payload["cases"], SHIPPED):
        case["expected_category"] = label
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    monkeypatch.setattr(gold_v2, "DATASET", path)
    monkeypatch.setattr(import_annotations, "DATASET", path)
    return path


@pytest.fixture
def annotation_dir(tmp_path):
    path = tmp_path / "annotations"
    path.mkdir()
    return path


def put(annotation_dir, annotator, labels):
    """labels: {case index: category}. A case left out is a case that
    annotator did not label."""
    records = [
        {"case_id": f"case-{i:03d}", "label": label, "notes": ""}
        for i, label in sorted(labels.items())
    ]
    payload = {
        "annotator_id": annotator,
        "annotator": import_annotations.ANNOTATORS[annotator]["rater"],
        "sheet_sha256": "0" * 64,
        "sheet_dataset_version": "v1",
        "n": len(records),
        "missing_case_ids": [f"case-{i:03d}" for i in range(len(SHIPPED)) if i not in labels],
        "labels": records,
    }
    (annotation_dir / f"{annotator.lower()}_labels.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run(annotation_dir, tmp_path, as_of, rulings=None, out=None, queue=None):
    out = out or tmp_path / "gold_v2.json"
    queue = queue or tmp_path / "queue.csv"
    argv = [
        "--as-of", as_of, "--annotation-dir", str(annotation_dir),
        "--out", str(out), "--queue-out", str(queue),
    ]
    if rulings:
        argv += ["--rulings", str(rulings)]
    return gold_v2.main(argv), out, queue


def routes(out):
    payload = json.loads(out.read_text(encoding="utf-8"))
    return {entry["case_id"]: (entry["label"], entry["labeled_by"]) for entry in payload["labels"]}


# -- all three sheets in ----------------------------------------------------
def test_three_votes_take_the_majority_and_send_a_three_way_split_to_the_author(
    dataset, annotation_dir, tmp_path
):
    put(annotation_dir, "A1", {0: "billing", 1: "technical", 2: "billing",
                               3: "general", 4: "account", 5: "technical"})
    put(annotation_dir, "A2", {0: "billing", 1: "technical", 2: "technical",
                               3: "general", 4: "account", 5: "technical"})
    put(annotation_dir, "A3", {0: "technical", 1: "technical", 2: "account",
                               3: "general", 4: "billing", 5: "technical"})
    code, out, _ = run(annotation_dir, tmp_path, "2026-09-24")
    assert code == 0
    decided = routes(out)
    assert decided["case-000"] == ("billing", "majority")      # 2 of 3
    assert decided["case-001"] == ("technical", "majority")    # unanimous
    assert decided["case-002"] == (None, "adjudicated")        # all three differ
    assert decided["case-004"] == ("account", "majority")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["regime"] == "main"
    assert payload["sealed"] is False
    assert payload["adjudication_pending_case_ids"] == ["case-002"]


def test_two_valid_votes_agreeing_decide_and_two_differing_go_to_the_author(
    dataset, annotation_dir, tmp_path
):
    """Section 9's two-vote rule: that sheet omitted the case, or its entry
    failed validation and no correction came back."""
    put(annotation_dir, "A1", {0: "billing", 1: "account", 2: "general",
                               3: "general", 4: "billing", 5: "technical"})
    put(annotation_dir, "A2", {0: "billing", 1: "technical", 2: "general",
                               3: "general", 4: "billing", 5: "technical"})
    put(annotation_dir, "A3", {2: "general", 3: "general", 4: "billing", 5: "technical"})
    code, out, _ = run(annotation_dir, tmp_path, "2026-09-24")
    assert code == 0
    decided = routes(out)
    assert decided["case-000"] == ("billing", "majority")    # two agreeing
    assert decided["case-001"] == (None, "adjudicated")      # two differing


def test_fewer_than_two_valid_votes_is_a_protocol_gap_and_writes_nothing(
    dataset, annotation_dir, tmp_path
):
    put(annotation_dir, "A1", {i: SHIPPED[i] for i in range(len(SHIPPED))})
    put(annotation_dir, "A2", {i: SHIPPED[i] for i in range(1, len(SHIPPED))})
    put(annotation_dir, "A3", {i: SHIPPED[i] for i in range(1, len(SHIPPED))})
    code, out, _ = run(annotation_dir, tmp_path, "2026-09-24")
    assert code == 4
    assert not out.exists()


# -- the adjudication sheet -------------------------------------------------
def test_the_adjudication_sheet_carries_the_email_and_the_human_labels_and_nothing_else(
    dataset, annotation_dir, tmp_path
):
    put(annotation_dir, "A1", {0: "billing", 1: "technical", 2: "billing",
                               3: "general", 4: "account", 5: "technical"})
    put(annotation_dir, "A2", {0: "technical", 1: "technical", 2: "technical",
                               3: "general", 4: "account", 5: "technical"})
    put(annotation_dir, "A3", {0: "account", 1: "technical", 2: "account",
                               3: "general", 4: "billing", 5: "technical"})
    code, _, queue = run(annotation_dir, tmp_path, "2026-09-24")
    assert code == 0
    rows = list(csv.DictReader(queue.read_text(encoding="utf-8").splitlines(True)))
    assert [row["case_id"] for row in rows] == ["case-000", "case-002"]
    assert set(rows[0]) == {"case_id", "email_body", "label_A1", "label_A2", "label_A3"}
    # The shipped v1.1 label for case-000 is "billing" and A1 also wrote
    # "billing", so its presence in the sheet must come from A1's column and
    # from no second place: nothing outside those five columns exists.
    assert rows[0]["label_A1"] == "billing"
    assert "draft" not in queue.read_text(encoding="utf-8")


def test_rulings_seal_the_gold_and_a_ruling_for_an_unqueued_case_is_refused(
    dataset, annotation_dir, tmp_path
):
    put(annotation_dir, "A1", {0: "billing", 1: "technical", 2: "billing",
                               3: "general", 4: "account", 5: "technical"})
    put(annotation_dir, "A2", {0: "technical", 1: "technical", 2: "technical",
                               3: "general", 4: "account", 5: "technical"})
    put(annotation_dir, "A3", {0: "account", 1: "technical", 2: "account",
                               3: "general", 4: "billing", 5: "technical"})
    run(annotation_dir, tmp_path, "2026-09-24")

    wrong = tmp_path / "wrong.csv"
    wrong.write_text("case_id,ruling\ncase-001,billing\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        run(annotation_dir, tmp_path, "2026-09-24", rulings=wrong)

    rulings = tmp_path / "rulings.csv"
    rulings.write_text("case_id,ruling\ncase-000,billing\ncase-002,account\n", encoding="utf-8")
    code, out, _ = run(annotation_dir, tmp_path, "2026-09-24", rulings=rulings)
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["sealed"] is True
    assert routes(out)["case-000"] == ("billing", "adjudicated")
    assert routes(out)["case-002"] == ("account", "adjudicated")


# -- fallback 1 -------------------------------------------------------------
def test_fallback_1_uses_the_shipped_label_only_to_break_a_two_human_split(
    dataset, annotation_dir, tmp_path
):
    put(annotation_dir, "A1", {0: "billing", 1: "billing", 2: "billing",
                               3: "general", 4: "billing", 5: "technical"})
    put(annotation_dir, "A2", {0: "billing", 1: "technical", 2: "technical",
                               3: "general", 4: "billing", 5: "technical"})
    code, out, _ = run(annotation_dir, tmp_path, "2026-09-27")
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["regime"] == "fallback_1"
    decided = routes(out)
    assert decided["case-000"] == ("billing", "majority")        # the two agree
    # case-001: A1 billing, A2 technical, shipped technical -> the shipped
    # label is the third vote and the case is individually identifiable.
    assert decided["case-001"] == ("technical", "fallback_v1.1")
    # case-002: A1 billing, A2 technical, shipped account -> all three differ.
    assert decided["case-002"] == (None, "adjudicated")


def test_fallback_1_keeps_the_shipped_label_where_only_a1_voted(
    dataset, annotation_dir, tmp_path
):
    put(annotation_dir, "A1", {i: "billing" for i in range(len(SHIPPED))})
    put(annotation_dir, "A2", {i: "billing" for i in range(1, len(SHIPPED))})
    code, out, _ = run(annotation_dir, tmp_path, "2026-09-27")
    assert code == 0
    assert routes(out)["case-000"] == (SHIPPED[0], "fallback_v1.1")


def test_fallback_1_with_only_the_other_humans_vote_is_a_protocol_gap(
    dataset, annotation_dir, tmp_path
):
    """Section 9 names the case where fallback 1 leaves only A1's vote and no
    other, so the mirror image is not decided anywhere."""
    put(annotation_dir, "A1", {i: "billing" for i in range(1, len(SHIPPED))})
    put(annotation_dir, "A2", {i: "billing" for i in range(len(SHIPPED))})
    code, out, _ = run(annotation_dir, tmp_path, "2026-09-27")
    assert code == 4
    assert not out.exists()


# -- fallback 2, waiting, and late arrivals ---------------------------------
def test_fallback_2_writes_nothing(dataset, annotation_dir, tmp_path):
    put(annotation_dir, "A1", {i: "billing" for i in range(len(SHIPPED))})
    code, out, _ = run(annotation_dir, tmp_path, "2026-09-27")
    assert code == 3
    assert not out.exists()


def test_before_the_deadline_an_incomplete_roster_writes_nothing(
    dataset, annotation_dir, tmp_path
):
    put(annotation_dir, "A1", {i: "billing" for i in range(len(SHIPPED))})
    put(annotation_dir, "A2", {i: "billing" for i in range(len(SHIPPED))})
    code, out, _ = run(annotation_dir, tmp_path, "2026-09-26")
    assert code == 3
    assert not out.exists()


def test_a_sheet_arriving_after_a_fallback_does_not_reopen_gold_v2(
    dataset, annotation_dir, tmp_path
):
    put(annotation_dir, "A1", {i: "billing" for i in range(len(SHIPPED))})
    put(annotation_dir, "A2", {i: "billing" for i in range(len(SHIPPED))})
    code, out, _ = run(annotation_dir, tmp_path, "2026-09-27")
    assert code == 0
    before = out.read_text(encoding="utf-8")

    put(annotation_dir, "A3", {i: "technical" for i in range(len(SHIPPED))})
    with pytest.raises(SystemExit, match="does not reopen gold v2"):
        run(annotation_dir, tmp_path, "2026-09-28")
    assert out.read_text(encoding="utf-8") == before


def test_a_roster_without_a1_is_a_protocol_gap(dataset, annotation_dir, tmp_path):
    put(annotation_dir, "A2", {i: "billing" for i in range(len(SHIPPED))})
    put(annotation_dir, "A3", {i: "billing" for i in range(len(SHIPPED))})
    code, out, _ = run(annotation_dir, tmp_path, "2026-09-27")
    assert code == 4
    assert not out.exists()
