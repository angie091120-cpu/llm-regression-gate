"""The gold v2 sensitivity analysis: re-grading, family labelling, differences.

docs/PREREGISTRATION.md section 9 (2026-09-19) puts three requirements on this
step, and each has a test here: the recomputation is arithmetic over judge
scores already recorded and obtains no new score; every gold v2 row carries the
`sensitivity_gold_v2` family and no multiplicity correction; and the two
versions are published beside each other with the differences listed table by
table.
"""
from __future__ import annotations

import json

from experiments import analyze


def observation(case_id, expected, predicted, judge_score, classifier_ok=True):
    obs = {
        "exp_id": "e0_noise", "run_id": "r1", "repeat_idx": 0, "case_id": case_id,
        "language": "en", "difficulty": "easy", "expected_category": expected,
        "prompt_version": "v1", "classifier_ok": classifier_ok, "judge_ok": True,
        "predicted_category": predicted, "predicted_summary": "s",
        "judged_summary_source": "reference", "judge_score": judge_score,
        "classifier_input_tokens": 1, "classifier_output_tokens": 1,
        "judge_input_tokens": 1, "judge_output_tokens": 1,
        "classifier_latency_ms": 1, "judge_latency_ms": 1, "cost_usd": 0.0,
        "classifier_response_model": "m", "judge_response_model": "m",
    }
    obs["category_match"] = None if not classifier_ok else int(predicted == expected)
    obs["passed"] = (
        None if (obs["category_match"] is None or judge_score is None)
        else int(bool(obs["category_match"]) and judge_score >= 3)
    )
    return obs


def test_regrading_moves_category_match_and_passed_and_leaves_judge_score_alone():
    before = [
        observation("case-001", "billing", "billing", 5),
        observation("case-002", "general", "technical", 4),
        observation("case-003", "account", "account", 2),
    ]
    assert [o["category_match"] for o in before] == [1, 0, 1]
    assert [o["passed"] for o in before] == [1, 0, 0]

    after = analyze.regrade_observations(
        before, {"case-001": "technical", "case-002": "technical", "case-003": "account"}
    )
    assert [o["category_match"] for o in after] == [0, 1, 1]
    # case-002 now matches and its recorded judge score was already 4, so it
    # passes without a new call being made.
    assert [o["passed"] for o in after] == [0, 1, 0]
    assert [o["judge_score"] for o in after] == [5, 4, 2]
    # The input list is not mutated: the v1.1 tables are built from it after.
    assert [o["expected_category"] for o in before] == ["billing", "general", "account"]


def test_regrading_leaves_a_failed_classifier_call_unscored():
    before = [observation("case-001", "billing", None, None, classifier_ok=False)]
    after = analyze.regrade_observations(before, {"case-001": "technical"})
    assert after[0]["category_match"] is None
    assert after[0]["passed"] is None


def test_every_gold_v2_row_takes_the_sensitivity_family_and_no_adjustment():
    header = ["metric", "p_raw", "p_holm", "p_bh", "family", "note"]
    rows = [
        ["category_match", 0.01, 0.04, 0.02, "confirmatory_holm", "an existing note"],
        ["category_match", 0.20, 0.20, 0.20, "sensitivity_drop_leaked_case", ""],
    ]
    out = analyze.relabel_gold_v2(header, rows)
    assert [row[4] for row in out] == [analyze.GOLD_V2_FAMILY, analyze.GOLD_V2_FAMILY]
    assert [row[2] for row in out] == [None, None]
    assert [row[3] for row in out] == [None, None]
    assert [row[1] for row in out] == [0.01, 0.20]
    assert "read" in out[0][5] and "p_raw" in out[0][5]
    # The family the row had under v1.1 survives in the note, because the
    # leak-sensitivity rows and the confirmatory rows are still different rows.
    assert "confirmatory_holm" in out[0][5]
    assert "sensitivity_drop_leaked_case" in out[1][5]
    assert "an existing note" in out[0][5]
    assert rows[0][4] == "confirmatory_holm"


def test_an_unsealed_gold_v2_is_not_read(tmp_path):
    path = tmp_path / "gold_v2.json"
    path.write_text(json.dumps({
        "sealed": False,
        "labels": [{"case_id": "case-001", "label": None, "labeled_by": "adjudicated"}],
    }), encoding="utf-8")
    assert analyze.load_gold_v2_labels(path) is None


def test_a_sealed_gold_v2_with_a_missing_label_is_not_read(tmp_path):
    path = tmp_path / "gold_v2.json"
    path.write_text(json.dumps({
        "sealed": True,
        "labels": [
            {"case_id": "case-001", "label": "billing", "labeled_by": "majority"},
            {"case_id": "case-002", "label": None, "labeled_by": "adjudicated"},
        ],
    }), encoding="utf-8")
    assert analyze.load_gold_v2_labels(path) is None


def test_a_sealed_gold_v2_is_read(tmp_path):
    path = tmp_path / "gold_v2.json"
    path.write_text(json.dumps({
        "sealed": True,
        "labels": [{"case_id": "case-001", "label": "billing", "labeled_by": "majority"}],
    }), encoding="utf-8")
    assert analyze.load_gold_v2_labels(path) == {"case-001": "billing"}


def test_the_difference_table_names_the_cell_that_moved():
    header = ["exp_id", "run_id", "prompt_version", "repeat_idx", "metric", "n", "k",
              "estimate", "ci_low", "ci_high", "method"]
    v1 = [["e0_noise", "r1", "v1", 0, "category_match", 70, 64, 0.914286, 0.0, 1.0, "wilson"]]
    v2 = [["e0_noise", "r1", "v1", 0, "category_match", 70, 62, 0.885714, 0.0, 1.0, "wilson"]]
    _, rows = analyze.table_gold_v2_diff([("rates_by_run.csv", (header, v1), (header, v2))])
    moved = [(row[2], row[3], row[4]) for row in rows]
    assert ("k", "64", "62") in moved
    assert ("estimate", "0.914286", "0.885714") in moved
    assert all(row[2] != "n" for row in rows)


def test_the_difference_table_says_so_when_a_table_did_not_move():
    header = ["exp_id", "run_id", "prompt_version", "repeat_idx", "metric", "n", "k",
              "estimate", "ci_low", "ci_high", "method"]
    rows_in = [["e0_noise", "r1", "v1", 0, "category_match", 70, 64, 0.914286, 0.0, 1.0, "wilson"]]
    _, rows = analyze.table_gold_v2_diff([("rates_by_run.csv", (header, rows_in), (header, list(rows_in)))])
    assert len(rows) == 1
    assert rows[0][5] == "identical under gold v2"


def test_the_difference_table_reports_a_row_that_exists_in_only_one_version():
    header = ["exp_id", "run_id", "prompt_version", "repeat_idx", "metric", "n", "k",
              "estimate", "ci_low", "ci_high", "method"]
    base = ["e0_noise", "r1", "v1", 0, "category_match", 70, 64, 0.914286, 0.0, 1.0, "wilson"]
    extra = ["e0_noise", "r1", "v1", 1, "category_match", 70, 64, 0.914286, 0.0, 1.0, "wilson"]
    _, rows = analyze.table_gold_v2_diff([("rates_by_run.csv", (header, [base]), (header, [base, extra]))])
    assert [row[4] for row in rows if row[2] == "(whole row)"] == ["present"]
