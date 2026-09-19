"""E4's panel rows, and which note lands on which pairing.

The three-rater rows of `e4_kappa.csv` cannot be exercised on real data yet:
A2 and A3 are due 2026-09-23, and until all three sheets are in the rows are
absent by design. Everything here is three synthetic `*_labels.json` files in
tmp_path, one of them deliberately short a case, because "Fleiss drops that
case and Krippendorff keeps it" is the reason section 9 asks for both.
"""
from __future__ import annotations

import json

import pytest

from experiments import analyze
from tests.test_annotation_import import make_dataset

HEADER = {name: i for i, name in enumerate(analyze.E4_KAPPA_HEADER)}
N_CASES = 12
# Twelve cases over the four categories. Twelve rather than a handful because
# `e4_panel_rows` bootstraps: on a five-case panel a resample in which every
# case lands in one category is common enough to hit, and Fleiss' kappa is
# genuinely undefined there and says so rather than returning a number. The
# real panel is 70 cases.
AGREED = {
    0: "billing", 1: "technical", 2: "account", 3: "general",
    4: "billing", 5: "technical", 6: "account", 7: "general",
    8: "billing", 9: "technical", 10: "account", 11: "general",
}


def put(annotation_dir, annotator, labels):
    """labels: {case index: category}; an index left out is a case that
    annotator did not label."""
    payload = {
        "annotator_id": annotator,
        "annotator": analyze.ANNOTATORS[annotator]["rater"],
        "sheet_sha256": "0" * 64,
        "missing_case_ids": [f"case-{i:03d}" for i in range(N_CASES) if i not in labels],
        "labels": [
            {"case_id": f"case-{i:03d}", "label": label, "notes": ""}
            for i, label in sorted(labels.items())
        ],
    }
    (annotation_dir / f"{annotator.lower()}_labels.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path, n=N_CASES)
    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()
    monkeypatch.setattr(analyze, "E4_ANNOTATION_DIR", annotation_dir)
    monkeypatch.setattr(analyze, "GOLD_V2_PATH", tmp_path / "no_gold_v2.json")
    return dataset, annotation_dir


def model_rows(model, labels):
    return [
        {
            "exp_id": analyze.E4_EXP_ID, "tier": analyze.E4_TIER, "request_model": model,
            "ok": True, "case_id": f"case-{i:03d}", "parsed": {"category": label},
        }
        for i, label in sorted(labels.items())
    ]


def rows_by_method(rows):
    return {row[HEADER["method"]]: row for row in rows}


# -- the three panel rows ---------------------------------------------------
def test_all_three_panel_rows_appear_once_every_human_sheet_is_in(env):
    dataset, annotation_dir = env
    put(annotation_dir, "A1", dict(AGREED))
    put(annotation_dir, "A2", {**AGREED, 3: "billing"})
    put(annotation_dir, "A3", {**AGREED, 1: "account"})
    labels, _ = analyze.e4_rater_labels([], dataset)
    rows = analyze.e4_panel_rows(labels, seed=20260920, n_boot=50)
    assert set(rows_by_method(rows)) == {
        "fleiss_kappa", "three_way_exact_agreement", "krippendorff_alpha_nominal"
    }
    for row in rows:
        assert row[HEADER["rater_b"]] == analyze.E4_PANEL_RATER_B
        assert row[HEADER["rater_a"]].startswith("panel: ")
        assert row[HEADER["n_boot"]] == 50 and row[HEADER["seed"]] == 20260920


def test_two_human_sheets_produce_no_panel_rows(env):
    """A three-rater coefficient computed on two raters would be a different
    quantity wearing the name section 9 fixed."""
    dataset, annotation_dir = env
    put(annotation_dir, "A1", {i: "billing" for i in range(N_CASES)})
    put(annotation_dir, "A2", {i: "billing" for i in range(N_CASES)})
    labels, _ = analyze.e4_rater_labels([], dataset)
    assert analyze.e4_panel_rows(labels, seed=20260920, n_boot=50) == []


def test_a_case_one_annotator_skipped_is_dropped_by_fleiss_and_kept_by_alpha(env):
    dataset, annotation_dir = env
    put(annotation_dir, "A1", dict(AGREED))
    put(annotation_dir, "A2", dict(AGREED))
    put(annotation_dir, "A3", {i: label for i, label in AGREED.items() if i != 0})
    labels, _ = analyze.e4_rater_labels([], dataset)
    rows = rows_by_method(analyze.e4_panel_rows(labels, seed=20260920, n_boot=50))

    fleiss = rows["fleiss_kappa"]
    assert (fleiss[HEADER["n"]], fleiss[HEADER["n_excluded"]]) == (N_CASES - 1, 1)
    rate = rows["three_way_exact_agreement"]
    assert (rate[HEADER["n"]], rate[HEADER["n_excluded"]]) == (N_CASES - 1, 1)

    # case-000 still has two of the three values, so it is pairable and alpha
    # keeps it. n counts units used, n_excluded counts units that were not
    # pairable at all.
    alpha = rows["krippendorff_alpha_nominal"]
    assert (alpha[HEADER["n"]], alpha[HEADER["n_excluded"]]) == (N_CASES, 0)


def test_the_alpha_rows_po_and_pe_read_as_the_kappa_column_says(env):
    """po and pe are 1 - observed and 1 - expected disagreement, chosen so that
    the shared `kappa` column still reads (po - pe) / (1 - pe)."""
    dataset, annotation_dir = env
    put(annotation_dir, "A1", dict(AGREED))
    put(annotation_dir, "A2", {**AGREED, 2: "billing", 4: "account"})
    put(annotation_dir, "A3", {**AGREED, 0: "technical", 5: "general"})
    labels, _ = analyze.e4_rater_labels([], dataset)
    alpha = rows_by_method(analyze.e4_panel_rows(labels, seed=20260920, n_boot=50))[
        "krippendorff_alpha_nominal"
    ]
    po, pe, value = alpha[HEADER["po"]], alpha[HEADER["pe"]], alpha[HEADER["kappa"]]
    assert (po - pe) / (1.0 - pe) == pytest.approx(value, abs=1e-12)
    # No Landis & Koch band: that convention was published for kappa.
    assert alpha[HEADER["landis_koch"]] is None


def test_the_exact_agreement_row_puts_its_rate_in_po_and_leaves_kappa_empty(env):
    dataset, annotation_dir = env
    # Ten of twelve unanimous, across four categories.
    put(annotation_dir, "A1", dict(AGREED))
    put(annotation_dir, "A2", {**AGREED, 10: "technical"})
    put(annotation_dir, "A3", {**AGREED, 11: "account"})
    labels, _ = analyze.e4_rater_labels([], dataset)
    rate = rows_by_method(analyze.e4_panel_rows(labels, seed=20260920, n_boot=50))[
        "three_way_exact_agreement"
    ]
    assert rate[HEADER["po"]] == pytest.approx(10 / 12)
    assert rate[HEADER["pe"]] is None
    assert rate[HEADER["kappa"]] is None


# -- which note lands where -------------------------------------------------
def test_only_the_model_pairings_say_a_model_against_labels_a_model_drafted(env):
    """The provenance of `human-1 (gold)` is true of every pairing it takes
    part in. "A model against labels a model drafted" is true only where the
    other rater is a model -- on A1's row it would describe A1 as a model."""
    dataset, annotation_dir = env
    put(annotation_dir, "A1", dict(AGREED))
    raw = model_rows("claude-haiku-4-5", {i: "billing" for i in range(N_CASES)})
    _, rows = analyze.table_e4_kappa(raw, dataset, seed=20260920, n_boot=50)
    by_pair = {(row[HEADER["rater_a"]], row[HEADER["rater_b"]]): row[HEADER["note"]] for row in rows}

    model_note = by_pair[(analyze.E4_GOLD_RATER, "model:claude-haiku-4-5")]
    assert analyze.E4_GOLD_PROVENANCE_NOTE in model_note
    assert analyze.E4_GOLD_VS_MODEL_NOTE in model_note

    human_note = by_pair[(analyze.E4_GOLD_RATER, analyze.ANNOTATORS["A1"]["rater"])]
    assert analyze.E4_GOLD_PROVENANCE_NOTE in human_note
    assert analyze.E4_GOLD_VS_MODEL_NOTE not in human_note
    assert "a model drafted" not in human_note

    a1_vs_model = by_pair[(analyze.ANNOTATORS["A1"]["rater"], "model:claude-haiku-4-5")]
    assert analyze.E4_GOLD_PROVENANCE_NOTE not in a1_vs_model
