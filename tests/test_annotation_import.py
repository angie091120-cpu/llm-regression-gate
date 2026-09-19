"""Ingestion of a returned annotator sheet and of the summary review sheet.

Both importers exist to refuse bad input loudly. These tests are mostly about
the refusals, because an importer that accepts a damaged sheet costs a wrong
label in gold v2 and nobody finds out.

Nothing here reads the real `golden_dataset.json` or the real returned sheets:
every case is synthetic and lives in tmp_path.
"""
from __future__ import annotations

import csv
import io
import json

import pytest

from experiments import blank_sheet_notes, import_annotations, import_summary_review

CATEGORIES = ("billing", "technical", "account", "general")


def make_dataset(tmp_path, n=4, version="v1.1"):
    cases = []
    for i in range(n):
        cases.append({
            "id": f"case-{i:03d}",
            "language": "en",
            "input_text": f"email body {i}",
            "draft_category": CATEGORIES[i % len(CATEGORIES)],
            "draft_summary": f"draft summary {i}",
            "draft_difficulty": "easy",
            "expected_category": CATEGORIES[i % len(CATEGORIES)],
            "expected_summary": f"summary {i}",
            "expected_difficulty": "easy",
            "notes": None,
            "label_status": "confirmed",
            "labeled_by": "human-1",
            "labeled_at": "2026-07-20",
        })
    path = tmp_path / "golden_dataset.json"
    path.write_text(
        json.dumps({"dataset_version": version, "cases": cases}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def write_csv(path, header, rows):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
    path.write_bytes(buffer.getvalue().encode("utf-8-sig"))
    return path


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    path = make_dataset(tmp_path)
    monkeypatch.setattr(import_annotations, "DATASET", path)
    monkeypatch.setattr(import_summary_review, "DATASET", path)
    monkeypatch.setattr(import_annotations, "EXPECTED_ROWS", 4)
    monkeypatch.setattr(import_summary_review, "EXPECTED_ROWS", 4)
    return path


def category_sheet(tmp_path, labels, bodies=None, name="sheet.csv", notes=None):
    bodies = bodies or {}
    notes = notes or {}
    rows = [
        [
            f"case-{i:03d}",
            bodies.get(f"case-{i:03d}", f"email body {i}"),
            label,
            notes.get(f"case-{i:03d}", ""),
        ]
        for i, label in enumerate(labels)
    ]
    return write_csv(tmp_path / name, ["case_id", "email_body", "your_label", "notes"], rows)


def test_a_clean_sheet_is_ingested_under_its_annotator(dataset, tmp_path):
    sheet = category_sheet(tmp_path, ["billing", "technical", "account", "general"])
    out = tmp_path / "a2_labels.json"
    assert import_annotations.main([
        "--annotator", "A2", "--sheet", str(sheet), "--out", str(out),
        "--annotated-on", "2026-09-23",
    ]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["annotator_id"] == "A2"
    assert payload["annotator"] == import_annotations.ANNOTATORS["A2"]["rater"]
    assert payload["n"] == 4
    assert payload["missing_case_ids"] == []
    assert [record["label"] for record in payload["labels"]] == [
        "billing", "technical", "account", "general"
    ]


def test_capitalisation_and_padding_are_normalised(dataset, tmp_path):
    sheet = category_sheet(tmp_path, [" Billing ", "technical", "account", "general"])
    out = tmp_path / "a1_labels.json"
    assert import_annotations.main(["--annotator", "A1", "--sheet", str(sheet), "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["labels"][0]["label"] == "billing"


def test_a_label_outside_the_four_is_rejected_and_nothing_is_written(dataset, tmp_path):
    sheet = category_sheet(tmp_path, ["billing", "urgent", "account", "general"])
    out = tmp_path / "a1_labels.json"
    assert import_annotations.main(["--annotator", "A1", "--sheet", str(sheet), "--out", str(out)]) == 1
    assert not out.exists()


def test_an_edited_email_body_is_rejected(dataset, tmp_path):
    sheet = category_sheet(
        tmp_path, ["billing", "technical", "account", "general"],
        bodies={"case-002": "somebody retyped this row"},
    )
    out = tmp_path / "a1_labels.json"
    assert import_annotations.main(["--annotator", "A1", "--sheet", str(sheet), "--out", str(out)]) == 1
    assert not out.exists()


def test_a_missing_case_is_rejected_by_default_and_recorded_under_the_flag(dataset, tmp_path):
    rows = [
        ["case-000", "email body 0", "billing", ""],
        ["case-001", "email body 1", "technical", ""],
        ["case-003", "email body 3", "general", ""],
    ]
    sheet = write_csv(tmp_path / "short.csv", ["case_id", "email_body", "your_label", "notes"], rows)
    out = tmp_path / "a3_labels.json"
    assert import_annotations.main(["--annotator", "A3", "--sheet", str(sheet), "--out", str(out)]) == 1
    assert not out.exists()

    assert import_annotations.main([
        "--annotator", "A3", "--sheet", str(sheet), "--out", str(out), "--allow-missing-cases",
    ]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["missing_case_ids"] == ["case-002"]
    assert payload["n"] == 3


def test_an_unknown_annotator_is_not_accepted(dataset, tmp_path):
    sheet = category_sheet(tmp_path, ["billing", "technical", "account", "general"])
    with pytest.raises(SystemExit):
        import_annotations.main(["--annotator", "A4", "--sheet", str(sheet)])


def test_the_ingested_labels_record_whether_there_was_a_note_and_not_the_note(
    dataset, tmp_path
):
    """A2 and A3 are outside the project and were promised anonymity and a
    published category answer, not a published notes column."""
    sheet = category_sheet(
        tmp_path, ["billing", "technical", "account", "general"],
        notes={"case-001": "hesitated between billing and account"},
    )
    out = tmp_path / "a2_labels.json"
    assert import_annotations.main([
        "--annotator", "A2", "--sheet", str(sheet), "--out", str(out),
    ]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload["labels"][0]) == {"case_id", "label", "has_notes"}
    assert [record["has_notes"] for record in payload["labels"]] == [False, True, False, False]
    assert "hesitated" not in out.read_text(encoding="utf-8")


# -- the notes-blanked copy that enters version control --------------------
def test_blanking_empties_the_notes_column_and_leaves_the_answers_alone(tmp_path):
    sheet = category_sheet(
        tmp_path, ["billing", "technical", "account", "general"],
        notes={"case-000": "unsure", "case-003": "could be technical"},
    )
    out = tmp_path / "copy.csv"
    assert blank_sheet_notes.main(["--sheet", str(sheet), "--out", str(out)]) == 0
    rows = list(csv.DictReader(out.read_text(encoding="utf-8-sig").splitlines(True)))
    assert [row["notes"] for row in rows] == ["", "", "", ""]
    assert [row["your_label"] for row in rows] == ["billing", "technical", "account", "general"]
    assert [row["case_id"] for row in rows] == [f"case-{i:03d}" for i in range(4)]
    assert [row["email_body"] for row in rows] == [f"email body {i}" for i in range(4)]
    assert "unsure" not in out.read_text(encoding="utf-8-sig")


def test_blanking_a_sheet_with_no_notes_reproduces_it_byte_for_byte(tmp_path):
    """Which is why A1's single registered hash serves as both of its hashes:
    the file committed for A1 already is its own notes-blanked copy."""
    sheet = category_sheet(tmp_path, ["billing", "technical", "account", "general"])
    out = tmp_path / "copy.csv"
    assert blank_sheet_notes.main(["--sheet", str(sheet), "--out", str(out)]) == 0
    assert out.read_bytes() == sheet.read_bytes()


def test_blanking_refuses_a_sheet_carrying_a_fifth_column(tmp_path):
    """Re-serialising would drop it, and a tool whose whole contract is "one
    column changes" cannot be the thing that quietly removes another."""
    sheet = write_csv(
        tmp_path / "extra.csv",
        ["case_id", "email_body", "your_label", "notes", "confidence"],
        [["case-000", "email body 0", "billing", "", "high"]],
    )
    out = tmp_path / "copy.csv"
    assert blank_sheet_notes.main(["--sheet", str(sheet), "--out", str(out)]) == 1
    assert not out.exists()


def test_blanking_refuses_a_sheet_whose_columns_are_reordered(tmp_path):
    """The copy is written in handout order, so accepting a reordered sheet
    would move columns as well as empty one."""
    sheet = write_csv(
        tmp_path / "reordered.csv",
        ["case_id", "your_label", "email_body", "notes"],
        [["case-000", "billing", "email body 0", ""]],
    )
    out = tmp_path / "copy.csv"
    assert blank_sheet_notes.main(["--sheet", str(sheet), "--out", str(out)]) == 1
    assert not out.exists()


def test_blanking_refuses_a_sheet_missing_a_column(tmp_path):
    sheet = write_csv(
        tmp_path / "short.csv",
        ["case_id", "email_body", "your_label"],
        [["case-000", "email body 0", "billing"]],
    )
    out = tmp_path / "copy.csv"
    assert blank_sheet_notes.main(["--sheet", str(sheet), "--out", str(out)]) == 1
    assert not out.exists()


def test_blanking_refuses_to_overwrite_without_force(tmp_path):
    sheet = category_sheet(tmp_path, ["billing", "technical", "account", "general"])
    out = tmp_path / "copy.csv"
    out.write_bytes(b"do not clobber")
    assert blank_sheet_notes.main(["--sheet", str(sheet), "--out", str(out)]) == 1
    assert out.read_bytes() == b"do not clobber"
    assert blank_sheet_notes.main(["--sheet", str(sheet), "--out", str(out), "--force"]) == 0
    assert out.read_bytes() != b"do not clobber"


# -- the summary review sheet ----------------------------------------------
SUMMARY_HEADER = ["case_id", "email_body", "reference_summary", "verdict(ok/edit)", "your_edit", "notes"]


def summary_sheet(tmp_path, verdicts, edits=None, summaries=None, name="review.csv"):
    edits = edits or {}
    summaries = summaries or {}
    rows = []
    for i, verdict in enumerate(verdicts):
        case_id = f"case-{i:03d}"
        rows.append([
            case_id,
            f"email body {i}",
            summaries.get(case_id, f"summary {i}"),
            verdict,
            edits.get(case_id, ""),
            "",
        ])
    return write_csv(tmp_path / name, SUMMARY_HEADER, rows)


def test_summary_review_reports_counts_and_the_edited_ids(dataset, tmp_path):
    sheet = summary_sheet(tmp_path, ["ok", "edit", "ok", "ok"], edits={"case-001": "a better summary"})
    out = tmp_path / "review.json"
    replacements = tmp_path / "replacements.json"
    assert import_summary_review.main([
        "--reviewer", "A1", "--sheet", str(sheet), "--out", str(out),
        "--replacements-out", str(replacements), "--reviewed-on", "2026-09-19",
    ]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert (payload["n_ok"], payload["n_edit"]) == (3, 1)
    assert payload["edit_case_ids"] == ["case-001"]
    # The replacement text is held apart from the verdicts, and apart from the
    # dataset: section 9 forbids overwriting expected_summary here. The
    # reviewer's free-text notes stay out of the committed JSON too -- only
    # whether a case carries one.
    assert "replacement" not in json.dumps(payload["verdicts"])
    assert set(payload["verdicts"][0]) == {"case_id", "verdict", "has_notes"}
    stored = json.loads(replacements.read_text(encoding="utf-8"))
    assert stored["replacements"] == [{"case_id": "case-001", "replacement": "a better summary"}]
    assert json.loads(dataset.read_text(encoding="utf-8"))["cases"][1]["expected_summary"] == "summary 1"


def test_summary_review_rejects_an_edit_with_no_replacement(dataset, tmp_path):
    sheet = summary_sheet(tmp_path, ["ok", "edit", "ok", "ok"])
    out = tmp_path / "review.json"
    assert import_summary_review.main([
        "--reviewer", "A1", "--sheet", str(sheet), "--out", str(out),
        "--replacements-out", str(tmp_path / "r.json"),
    ]) == 1
    assert not out.exists()


def test_summary_review_rejects_ok_with_a_replacement_filled_in(dataset, tmp_path):
    sheet = summary_sheet(tmp_path, ["ok", "ok", "ok", "ok"], edits={"case-002": "stray text"})
    out = tmp_path / "review.json"
    assert import_summary_review.main([
        "--reviewer", "A1", "--sheet", str(sheet), "--out", str(out),
        "--replacements-out", str(tmp_path / "r.json"),
    ]) == 1
    assert not out.exists()


def test_summary_review_rejects_a_sheet_whose_reference_summary_was_edited(dataset, tmp_path):
    sheet = summary_sheet(
        tmp_path, ["ok", "ok", "ok", "ok"], summaries={"case-003": "not the shipped summary"}
    )
    out = tmp_path / "review.json"
    assert import_summary_review.main([
        "--reviewer", "A1", "--sheet", str(sheet), "--out", str(out),
        "--replacements-out", str(tmp_path / "r.json"),
    ]) == 1
    assert not out.exists()
