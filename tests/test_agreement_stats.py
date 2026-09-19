"""The two agreement estimators added for the three-annotator panel.

`checks/experiments_acceptance.sh` C5 already runs `python -m experiments.stats`,
which asserts these same printed examples among thirty-odd others. They are
here as well because the acceptance script is not what runs on a pull request:
a refactor that breaks Fleiss' kappa should fail the production suite, not wait
for someone to run the acceptance gate by hand.

Sources, and why these two: docs/PREREGISTRATION.md section 9 (entry dated
2026-09-19) asks for Fleiss' kappa and Krippendorff's alpha (nominal) across
the three human annotators. Every estimator in `experiments/stats.py` is pinned
to a published worked example rather than to another implementation, and these
follow that rule -- Randolph (2005) ERIC ED490661 for Fleiss, Krippendorff
(2011) for alpha. The constants carry the page numbers.
"""
from __future__ import annotations

import pytest

from experiments.stats import (
    FLEISS_RANDOLPH_EXAMPLE,
    KRIPPENDORFF_2011_EXAMPLE,
    fleiss_kappa,
    krippendorff_alpha,
)


def _expand(counts):
    """(yes, no) counts per case -> one label list per case."""
    return [["yes"] * yes + ["no"] * no for yes, no in counts]


def test_fleiss_reproduces_randolph_table_1():
    example = FLEISS_RANDOLPH_EXAMPLE
    result = fleiss_kappa(_expand(example["table1"]))
    assert result["po"] == pytest.approx(example["table1_po"], abs=example["printed_tolerance"])
    assert result["pe"] == pytest.approx(example["table1_pe"], abs=example["printed_tolerance"])
    assert result["kappa"] == pytest.approx(example["table1_kappa_exact"], abs=1e-12)


def test_fleiss_reproduces_randolph_table_2_and_is_not_the_free_marginal_value():
    """The discriminating case. Both tables have the same observed agreement and
    differ only in marginal symmetry: fixed marginals give -0.2, and the
    free-marginal coefficient the same paper prints beside it gives +0.33."""
    example = FLEISS_RANDOLPH_EXAMPLE
    result = fleiss_kappa(_expand(example["table2"]))
    assert result["po"] == pytest.approx(example["table2_po"], abs=example["printed_tolerance"])
    assert result["kappa"] == pytest.approx(example["table2_kappa"], abs=1e-12)
    assert result["kappa"] != pytest.approx(example["free_marginal_both_tables"], abs=0.02)


def test_fleiss_is_undefined_when_every_item_is_one_category_and_says_so():
    """Expected agreement is then 1 and kappa is 0/0. It matters in a bootstrap
    over a small panel, where a resample can land every case in one category:
    the estimator raises instead of returning a number the caller would print."""
    with pytest.raises(ValueError, match="expected agreement is 1.0"):
        fleiss_kappa([["a", "a", "a"], ["a", "a", "a"]])


def test_fleiss_refuses_an_unequal_number_of_raters():
    """Several generalisations exist for unequal n and this study has validated
    none of them, so the caller drops the incomplete items and says how many."""
    with pytest.raises(ValueError, match="same number of raters"):
        fleiss_kappa([["a", "b"], ["a", "b", "b"]])


def test_krippendorff_reproduces_the_2011_example_and_its_margins():
    example = KRIPPENDORFF_2011_EXAMPLE
    units = [list(column) for column in zip(*example["observers"])]
    result = krippendorff_alpha(units)
    assert result["alpha"] == pytest.approx(
        example["alpha_nominal"], abs=example["printed_tolerance"]
    )
    assert result["n_pairable"] == example["n_pairable"]
    assert result["category_counts"] == example["category_counts"]
    assert result["n_units_dropped"] == example["n_units_dropped"]


def test_krippendorff_keeps_a_unit_one_observer_skipped():
    """The reason section 9 asks for alpha as well as Fleiss' kappa: a case one
    annotator missed costs that annotator's vote, not the whole case."""
    both = krippendorff_alpha([["a", "a", "a"], ["b", "b", "b"], ["a", "a", None]])
    assert both["n_units_dropped"] == 0
    assert both["n_pairable"] == 8


def test_krippendorff_drops_a_unit_no_two_observers_labelled():
    result = krippendorff_alpha([["a", "a"], ["b", "b"], ["a", None]])
    assert result["n_units_dropped"] == 1
    assert result["n_pairable"] == 4
