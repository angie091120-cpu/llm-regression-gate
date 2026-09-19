"""Statistics for the pilot experiments -- standard library only.

Every estimator in this module is implemented against the Python standard
library (`math.comb`, `statistics.NormalDist`, `random`). scipy and
statsmodels live in the analysis venv `~/.venvs/lrg-exp` (Python 3.14.4:
scipy 1.18.1, numpy 2.5.3, statsmodels 0.15.0, matplotlib 3.11.2) and are used
only as an optional cross-check, never as the source of a published number.
Two reasons:

1. The estimators keep working on an interpreter that has no scipy build --
   the repo's own `.venv` is one, and a cp314 wheel that exists today may not
   exist for the next release.
2. A reader of the write-up can follow every formula to a line of code here
   instead of to a third-party release.

Self-check against published worked examples:

    python -m experiments.stats            # standard library only
    python -m experiments.stats --cross-check   # adds the scipy comparisons
                                                # under the analysis venv. On an
                                                # interpreter without scipy the
                                                # scipy row is FAIL and the exit
                                                # code is 1 -- never a silent skip.
    python -m experiments.stats --coverage      # Monte-Carlo coverage of the
                                                # paired risk-difference interval

The command line is strict: an unrecognised flag exits 2 rather than falling
through to the default self-check.

Fleiss' kappa and Krippendorff's alpha (nominal) arrived on 2026-09-19 for the
three-annotator panel of docs/PREREGISTRATION.md section 9, each pinned to a
printed worked example the same way every other estimator here is --
`FLEISS_RANDOLPH_EXAMPLE` and `KRIPPENDORFF_2011_EXAMPLE` name the source and
the page.
"""
from __future__ import annotations

import functools
import math
import random
from statistics import NormalDist
from typing import Callable, Iterable, Sequence

__all__ = [
    "wilson_ci",
    "clopper_pearson_ci",
    "binom_pmf",
    "binom_cdf",
    "exact_binomial_test",
    "fisher_exact_2x2",
    "hypergeom_pmf",
    "TEA_TASTING_EXAMPLE",
    "mcnemar_exact",
    "bootstrap_ci",
    "holm_adjust",
    "bh_adjust",
    "cohen_kappa",
    "newcombe_paired_diff_ci",
    "NEWCOMBE_VALIDATION",
    "NEWCOMBE_TABLE3_EXAMPLE",
    "paired_power_simulation",
    "fleiss_kappa",
    "krippendorff_alpha",
    "AGREEMENT_VALIDATION",
    "FLEISS_RANDOLPH_EXAMPLE",
    "KRIPPENDORFF_2011_EXAMPLE",
]

_TOL = 1e-12

# Worked example transcribed from Newcombe (1998), Statistics in Medicine
# 17(22):2635-2650, Table III -- the row carrying the paper's dagger footnote
# ("From reference 10, p. 122", i.e. Armitage & Berry). Used by `_selftest()`.
#
# The paper's cell letters (its section 2 and Table I) are e, f, g, h:
#
#     e  positive on both occasions
#     f  positive on the first occasion only
#     g  positive on the second occasion only
#     h  negative on both
#
# with theta-hat = (f - g) / n. This module writes the same table as
# (a, b, c, d) in candidate-vs-baseline order, so the translation is
#
#     a = e,  b = g,  c = f,  d = h
#
# and rd = (c - b) / n = (f - g) / n = theta-hat. Reading f and g the wrong way
# round negates the interval without changing its width, which is the failure
# this example is here to catch: the self-check asserts the signed endpoints
# and asserts that the transposed table does *not* satisfy them.
#
# `method10_ci` is the shipped default (phi continuity correction on);
# `method8_ci` is the same table's uncorrected row, which the paper also
# prints, so `continuity=False` is pinned to published values as well.
NEWCOMBE_TABLE3_EXAMPLE = {
    "source": "Newcombe (1998), Stat Med 17(22):2635-2650, Table III",
    "e": 20,
    "f": 12,
    "g": 2,
    "h": 16,
    "n": 50,
    "theta_hat": 0.2000,
    "method10_ci": (0.0562, 0.3292),
    "method8_ci": (0.0618, 0.3242),
    "tolerance": 1e-4,  # the paper prints 4 decimal places
}

# Carried into every table and figure that shows a Newcombe interval, so the
# validation status travels with the number instead of living in a doc.
# See docs/PREREGISTRATION.md section 9 (2026-09-14 and 2026-09-15 entries).
NEWCOMBE_VALIDATION = (
    "Newcombe (1998) method 10, square-and-add with phi continuity correction; "
    "reproduces the paper's Table III worked example (e=20, f=12, g=2, h=16: "
    "0.0562 to 0.3292) to the 4 dp it is printed at, and also checked by "
    "structural invariants and Monte-Carlo coverage (experiments/stats.py "
    "--coverage)"
)


# Worked example for Fleiss' kappa, from Randolph, J. J. (2005), "Free-Marginal
# Multirater Kappa (multirater kappa-free): An Alternative to Fleiss'
# Fixed-Marginal Multirater Kappa", Joensuu Learning and Instruction Symposium
# 2005 (ERIC ED490661). The paper's Tables 1 and 2 are on its page 17 and the
# arithmetic is worked line by line on pages 7 and 8.
#
# Why this paper and not Fleiss (1971), Psychological Bulletin 76(5):378-382,
# which is the source of the coefficient: no copy of that paper was reachable
# from this machine on 2026-09-19 (it is not open access; OpenAlex confirms the
# citation and lists no OA location), so its printed table could not be
# transcribed from the paper itself. Transcribing it from a third-party
# reimplementation would be a check of that reimplementation, not of the
# printed value, so it is not done here. Randolph prints Fleiss' quantities for
# two tables and is open access.
#
# Each table is (yes_count, no_count) per case: N = 4 cases, n = 3 raters,
# k = 2 categories. The pair is the point. Both tables have the same observed
# agreement, .67, and differ only in marginal symmetry, so:
#
#   Table 1 (symmetric marginals, 6 yes / 6 no)   Fleiss kappa =  1/3
#   Table 2 (skewed marginals,   10 yes / 2 no)   Fleiss kappa = -0.2
#
# and the paper states that the free-marginal coefficient is .33 for *both*.
# An implementation that builds expected agreement from free marginals reads
# -0.2 as +0.33, so the Table 2 row catches the one substitution most likely to
# pass every other sanity check. `table1_kappa_printed` is the paper's own
# .34: it divides its already-rounded .67 by 1 - .50, and 2/3 exact gives
# 1/3. Both are asserted, the printed one at the precision it is printed at,
# so the example is reproduced without the rounding being copied into the code.
FLEISS_RANDOLPH_EXAMPLE = {
    "source": "Randolph (2005), ERIC ED490661, Tables 1 and 2 (p. 17), worked on pp. 7-8",
    "table1": ((3, 0), (2, 1), (1, 2), (0, 3)),
    "table1_po": 0.67,
    "table1_pe": 0.50,
    "table1_kappa_printed": 0.34,  # from the paper's rounded po, see above
    "table1_kappa_exact": 1.0 / 3.0,
    "table2": ((3, 0), (2, 1), (2, 1), (3, 0)),
    "table2_po": 0.67,
    "table2_kappa": -0.2,
    "free_marginal_both_tables": 0.33,  # the value a free-marginal formula returns
    "printed_tolerance": 5e-3,  # the paper prints two decimals
}

# Worked example for Krippendorff's alpha, from Krippendorff, K. (2011),
# "Computing Krippendorff's Alpha-Reliability", University of Pennsylvania
# Annenberg School for Communication. The 4-observers-by-12-units reliability
# data matrix is printed on the paper's page 4 and alpha-nominal = .743 on page
# 5, re-derived on pages 8 and 10 by two further computational routes.
#
# `None` is the paper's ".", a value an observer did not assign. Seven of the
# 48 cells are missing, unit 12 is left with one value and is therefore not
# pairable, and the paper's own working carries n = 40 pairable values with
# category counts 9, 13, 10, 5, 3 -- all of which the self-check asserts
# alongside the coefficient, so a missing-value rule that silently drops or
# imputes the wrong cells fails on the margins before it fails on alpha.
KRIPPENDORFF_2011_EXAMPLE = {
    "source": "Krippendorff (2011), Computing Krippendorff's Alpha-Reliability, pp. 4-5",
    "observers": (
        ("1", "2", "3", "3", "2", "1", "4", "1", "2", None, None, None),
        ("1", "2", "3", "3", "2", "2", "4", "1", "2", "5", None, "3"),
        (None, "3", "3", "3", "2", "3", "4", "2", "2", "5", "1", None),
        ("1", "2", "3", "3", "2", "4", "4", "1", "2", "5", "1", None),
    ),
    "alpha_nominal": 0.743,
    "n_pairable": 40,
    "category_counts": {"1": 9, "2": 13, "3": 10, "4": 5, "5": 3},
    "n_units_dropped": 1,
    "printed_tolerance": 5e-4,  # the paper prints three decimals
}

AGREEMENT_VALIDATION = (
    "Fleiss kappa reproduces Randolph (2005) Tables 1 and 2 (ERIC ED490661), including the "
    "-0.2 the free-marginal coefficient reads as +0.33; Krippendorff alpha (nominal) reproduces "
    "the 4-observers-by-12-units example of Krippendorff (2011), alpha = .743, together with its "
    "printed margins. Both checks run in experiments/stats.py"
)


def _z(alpha: float) -> float:
    """Two-sided critical value of the standard normal."""
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


# --------------------------------------------------------------------------
# Binomial proportion intervals
# --------------------------------------------------------------------------
def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the Wald interval at the sample sizes in this study
    (n=70, and n as small as 7 in the language strata): Wald degenerates to
    a zero-width interval at k=0 or k=n and undercovers badly below n~40.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError(f"k={k} out of range for n={n}")
    z = _z(alpha)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def binom_pmf(k: int, n: int, p: float) -> float:
    if k < 0 or k > n:
        return 0.0
    if p <= 0.0:
        return 1.0 if k == 0 else 0.0
    if p >= 1.0:
        return 1.0 if k == n else 0.0
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(binom_pmf(i, n, p) for i in range(0, k + 1))


def _binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(binom_pmf(i, n, p) for i in range(k, n + 1))


def _bisect(f: Callable[[float], float], lo: float, hi: float, iters: int = 200) -> float:
    """Root of a monotone f on [lo, hi] by bisection (no scipy dependency)."""
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) binomial interval, solved by bisection on the
    binomial CDF rather than via an incomplete-beta quantile, so it needs no
    scipy. Reported alongside Wilson wherever a conservative bound matters."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError(f"k={k} out of range for n={n}")
    lower = 0.0 if k == 0 else _bisect(lambda p: _binom_sf(k, n, p) - alpha / 2.0, 0.0, 1.0)
    upper = 1.0 if k == n else _bisect(lambda p: binom_cdf(k, n, p) - alpha / 2.0, 0.0, 1.0)
    return (lower, upper)


def exact_binomial_test(k: int, n: int, p0: float = 0.5) -> float:
    """Two-sided exact binomial p-value, 'minimum likelihood' method:
    sum the probabilities of every outcome no more likely than the observed
    one. This is the method R's binom.test and scipy's binomtest use, so the
    number is comparable to anything a reader reproduces elsewhere."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError(f"k={k} out of range for n={n}")
    observed = binom_pmf(k, n, p0)
    total = 0.0
    for i in range(n + 1):
        pmf = binom_pmf(i, n, p0)
        if pmf <= observed * (1 + 1e-7):
            total += pmf
    return min(1.0, total)


# --------------------------------------------------------------------------
# Unpaired 2x2: Fisher exact
# --------------------------------------------------------------------------
# Fisher (1935), The Design of Experiments, chapter 2 -- the lady tasting tea.
# Eight cups, four of each preparation, the taster names four and gets three
# right: the 2x2 is (a, b, c, d) = (3, 1, 1, 3). Every probability in that
# table is a ratio of binomial coefficients over C(8,4) = 70, so both p-values
# are exact rationals rather than constants rounded off a page:
#
#   P(a=4) = 1/70   P(a=3) = 16/70   P(a=2) = 36/70   P(a=1) = 16/70   P(a=0) = 1/70
#
# one-sided (a >= 3):                          17/70 = 0.2428571...
# two-sided, every outcome no more likely
# than the observed 16/70:      1 + 16 + 16 + 1 = 34/70 = 0.4857142...
#
# _selftest() recomputes both from math.comb along a path that never calls the
# function below, so a rewrite of it cannot agree with itself and pass.
TEA_TASTING_EXAMPLE = {
    "source": "Fisher (1935), The Design of Experiments, ch. 2 (lady tasting tea)",
    "table": (3, 1, 1, 3),
    "p_two_sided": 34 / 70,
    "p_greater": 17 / 70,
}


def hypergeom_pmf(k: int, n_total: int, n_success: int, n_draw: int) -> float:
    """P(X = k) for X ~ Hypergeometric(n_total, n_success, n_draw): exact
    integer arithmetic up to a single final division."""
    if k < 0 or k > n_draw or k > n_success or (n_draw - k) > (n_total - n_success):
        return 0.0
    return (
        math.comb(n_success, k)
        * math.comb(n_total - n_success, n_draw - k)
        / math.comb(n_total, n_draw)
    )


def fisher_exact_2x2(a: int, b: int, c: int, d: int, alternative: str = "two-sided") -> dict:
    """Fisher exact test on

            success   failure
        g1     a         b
        g2     c         d

    Conditioning on both margins makes the count in cell `a` hypergeometric,
    so the p-value is a finite sum of exact rationals: no approximation, and
    no minimum expected-count condition to violate. That is why E5 compares
    strata with this rather than chi-square -- two of its strata are n = 7 and
    n = 8 (mixed language, edge difficulty), sizes at which the chi-square
    approximation does not apply at all.

    Two-sided sums the probabilities of every table no more likely than the
    observed one, the convention R fisher.test and scipy.stats.fisher_exact
    use, so the number is comparable to anything a reader reproduces
    elsewhere. The 1e-7 relative slack in that comparison is there because two
    tables can be equiprobable up to floating-point noise; without it, a table
    exactly as likely as the observed one is kept or dropped on the last bit.
    """
    if min(a, b, c, d) < 0:
        raise ValueError("cell counts must be non-negative")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"unknown alternative={alternative!r}")
    n = a + b + c + d
    if n == 0:
        raise ValueError("Fisher exact on an empty table")
    row1, col1 = a + b, a + c
    lo = max(0, col1 - (c + d))
    hi = min(row1, col1)
    support = {x: hypergeom_pmf(x, n, row1, col1) for x in range(lo, hi + 1)}
    observed = support[a]
    if alternative == "greater":
        p = sum(pr for x, pr in support.items() if x >= a)
    elif alternative == "less":
        p = sum(pr for x, pr in support.items() if x <= a)
    else:
        p = sum(pr for pr in support.values() if pr <= observed * (1 + 1e-7))
    rate1 = a / row1 if row1 else None
    rate2 = c / (c + d) if (c + d) else None
    return {
        "a": a, "b": b, "c": c, "d": d, "n": n,
        "rate_group1": rate1,
        "rate_group2": rate2,
        "difference": (rate1 - rate2) if (rate1 is not None and rate2 is not None) else None,
        "p_value": min(1.0, p),
        "alternative": alternative,
        "method": "fisher_exact_conditional",
    }


# --------------------------------------------------------------------------
# Paired binary outcomes
# --------------------------------------------------------------------------
def mcnemar_exact(b: int, c: int) -> dict:
    """Exact McNemar test on the discordant pairs.

    b = cases the baseline got right and the candidate got wrong
    c = cases the candidate got right and the baseline got wrong

    Uses the exact binomial (not the chi-square approximation): with 70
    paired cases the discordant count is routinely under 10, where the
    chi-square version is anticonservative.
    """
    if b < 0 or c < 0:
        raise ValueError("b and c must be non-negative")
    n_disc = b + c
    if n_disc == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0, "method": "mcnemar_exact"}
    return {
        "b": b,
        "c": c,
        "n_discordant": n_disc,
        "p_value": exact_binomial_test(b, n_disc, 0.5),
        "method": "mcnemar_exact",
    }


def phi_paired(a: int, b: int, c: int, d: int, continuity: bool = True) -> float:
    """Correlation between the two paired proportions, as Newcombe's method 10
    uses it. Zero when any margin is empty; clamped to [-1, 1].

    `continuity=True` subtracts n/2 from (ad - bc) when that product difference
    is positive and floors the result at zero. Shrinking phi towards zero
    widens the interval, so the corrected form is the conservative one; the
    uncorrected form is available for comparison.
    """
    n = a + b + c + d
    if n <= 0:
        raise ValueError("empty table")
    denom_sq = (a + b) * (c + d) * (a + c) * (b + d)
    if denom_sq <= 0:
        return 0.0
    num = a * d - b * c
    if continuity and num > 0:
        num = max(num - n / 2.0, 0.0)
    return max(-1.0, min(1.0, num / math.sqrt(denom_sq)))


def newcombe_paired_diff_ci(
    a: int, b: int, c: int, d: int, alpha: float = 0.05, continuity: bool = True
) -> dict:
    """Newcombe (1998) method 10 interval for the difference of two paired
    proportions, Stat Med 17:2635-2650.

    Cell convention, matching `mcnemar_exact` and `experiments/analyze.py`:

        a  baseline correct, candidate correct
        b  baseline correct, candidate wrong
        c  baseline wrong,   candidate correct
        d  baseline wrong,   candidate wrong

    The estimated quantity is `rd = p_candidate - p_baseline = (c - b) / n`,
    so a degradation is negative.

    Method 10 is "square-and-add": take the Wilson score interval for each
    marginal proportion and combine the two distances from their point
    estimates, with a correction for the correlation between them.

        p1 = (a + c) / n   candidate rate,  Wilson interval (l1, u1)
        p2 = (a + b) / n   baseline  rate,  Wilson interval (l2, u2)
        d  = p1 - p2
        L  = d - sqrt( (p1-l1)^2 - 2*phi*(p1-l1)*(u2-p2) + (u2-p2)^2 )
        U  = d + sqrt( (u1-p1)^2 - 2*phi*(u1-p1)*(p2-l2) + (p2-l2)^2 )

    with `phi` from `phi_paired()`. Both radicands are non-negative for any
    phi <= 1, because x^2 - 2*phi*x*y + y^2 >= (x - y)^2 when x, y >= 0.

    **Validation status.** As of 2026-09-15 this reproduces the paper's own
    printed worked example, Table III with e=20, f=12, g=2, h=16 (n=50):
    method 10 gives 0.056156 to 0.329207 against the printed 0.0562 to 0.3292,
    and `continuity=False` gives 0.061805 to 0.324162 against the printed
    method 8 row 0.0618 to 0.3242. Both are asserted by `_selftest()` from
    `NEWCOMBE_TABLE3_EXAMPLE`, which also documents the e/f/g/h to a/b/c/d
    translation and pins the sign. That sits on top of the three checks the
    function shipped with on 2026-09-14: the formula written out line by line
    above, the structural invariants in `_selftest()` (symmetry at b == c,
    containment of the point estimate, the continuity correction widening
    rather than narrowing, degenerate tables), and the Monte-Carlo coverage
    study under `--coverage`, whose result is printed rather than hidden.
    See docs/PREREGISTRATION.md section 9, entries 2026-09-14 and 2026-09-15;
    `note` on the returned dict carries the status into every table that uses
    the interval.
    """
    n = a + b + c + d
    if min(a, b, c, d) < 0:
        raise ValueError("cell counts must be non-negative")
    if n <= 0:
        raise ValueError("empty table")
    p1 = (a + c) / n
    p2 = (a + b) / n
    l1, u1 = wilson_ci(a + c, n, alpha)
    l2, u2 = wilson_ci(a + b, n, alpha)
    phi = phi_paired(a, b, c, d, continuity=continuity)
    rd = p1 - p2
    lo_term = math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi_term = math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return {
        "rd": rd,
        "ci_low": max(-1.0, rd - lo_term),
        "ci_high": min(1.0, rd + hi_term),
        "phi": phi,
        "p_baseline": p2,
        "p_candidate": p1,
        "n": n,
        "alpha": alpha,
        "continuity": continuity,
        "method": "newcombe_1998_method10",
        "note": NEWCOMBE_VALIDATION,
    }


# --------------------------------------------------------------------------
# Power
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _mcnemar_p(b: int, c: int) -> float:
    """Cached exact McNemar p-value. The power simulation asks for the same
    (b, c) thousands of times; the cache makes B = 2000 at three sample sizes
    cheap enough to run inside the acceptance check."""
    return mcnemar_exact(b, c)["p_value"]


def paired_power_simulation(
    pairs: Sequence[tuple[int, int]],
    sizes: Sequence[int],
    seed: int,
    label: str = "",
    n_sim: int = 2000,
    alphas: Sequence[float] = (0.05,),
) -> list[dict]:
    """Resampling power for the exact McNemar test, conditional on the
    observed effect.

    `pairs` is one (baseline, candidate) outcome per *case* -- the case is the
    independent unit, which is why repeated measurements of the same case are
    collapsed before they get here (docs/PREREGISTRATION.md section 10). For
    each target sample size, `n_sim` datasets of that size are drawn from the
    observed pairs with replacement and tested; power is the share that
    rejects at `alpha`.

    This is conditional power: it answers "if the true effect is the one this
    study happened to observe, how often would a study of size n detect it",
    and it inherits every peculiarity of the 70 observed cases. It is not a
    design power calculation and cannot be used as one -- the sample size was
    fixed by the dataset before any of this ran.

    Deterministic: the generator is seeded from (seed, label, n), so a result
    does not depend on the order in which sizes or versions are evaluated.
    """
    pairs = [(int(x), int(y)) for x, y in pairs]
    if not pairs:
        raise ValueError("power simulation over zero pairs")
    if n_sim <= 0:
        raise ValueError("n_sim must be positive")
    observed_b = sum(1 for x, y in pairs if x == 1 and y == 0)
    observed_c = sum(1 for x, y in pairs if x == 0 and y == 1)
    out: list[dict] = []
    for n in sizes:
        if n <= 0:
            raise ValueError("sample size must be positive")
        rng = random.Random(f"{seed}|{label}|{n}")
        draws = []
        for _ in range(n_sim):
            b = c = 0
            for _ in range(n):
                x, y = pairs[rng.randrange(len(pairs))]
                if x == 1 and y == 0:
                    b += 1
                elif x == 0 and y == 1:
                    c += 1
            draws.append((b, c))
        for alpha in alphas:
            reject = sum(1 for b, c in draws if _mcnemar_p(b, c) < alpha)
            out.append({
                "label": label,
                "n": n,
                "alpha": alpha,
                "power": reject / n_sim,
                "n_sim": n_sim,
                "seed": seed,
                "n_observed_pairs": len(pairs),
                "observed_b": observed_b,
                "observed_c": observed_c,
                "mean_b": sum(b for b, _ in draws) / n_sim,
                "mean_c": sum(c for _, c in draws) / n_sim,
                "method": "case_resampling_conditional_power_mcnemar_exact",
            })
    return out


# --------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------
def bootstrap_ci(
    units: Sequence[float] | Sequence[Sequence[float]],
    statistic: Callable[[list], float],
    seed: int,
    n_boot: int = 10000,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap over independent units.

    `units` is one entry per *case*, not per observation: when a case is
    measured five times, pass its five outcomes as one unit. Resampling
    whole cases keeps the repeated measurements of the same email together,
    which is the dependence structure this design actually has.

    Deterministic for a fixed `seed` (random.Random, not numpy), so the same
    seed reproduces the same interval on any machine with the same Python.
    """
    units = list(units)
    n = len(units)
    if n == 0:
        raise ValueError("bootstrap over zero units")
    rng = random.Random(seed)
    point = statistic(units)
    draws: list[float] = []
    for _ in range(n_boot):
        sample = [units[rng.randrange(n)] for _ in range(n)]
        draws.append(statistic(sample))
    draws.sort()
    lo_idx = int(math.floor((alpha / 2.0) * n_boot))
    hi_idx = min(n_boot - 1, int(math.ceil((1 - alpha / 2.0) * n_boot)) - 1)
    return {
        "estimate": point,
        "ci_low": draws[lo_idx],
        "ci_high": draws[hi_idx],
        "n_units": n,
        "n_boot": n_boot,
        "seed": seed,
        "method": "case_cluster_bootstrap_percentile",
    }


# --------------------------------------------------------------------------
# Multiplicity
# --------------------------------------------------------------------------
def holm_adjust(pvals: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, original input order preserved.
    Used for the confirmatory family only (the four E1 degradation versions
    against the primary metric); see docs/PREREGISTRATION.md."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        adjusted[idx] = running
    return adjusted


def bh_adjust(pvals: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (FDR), input order preserved.
    Used for the exploratory families, which stay labelled exploratory in
    every table regardless of the result."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i], reverse=True)
    adjusted = [0.0] * m
    running = 1.0
    for pos, idx in enumerate(order):
        rank = m - pos
        val = min(1.0, pvals[idx] * m / rank)
        running = min(running, val)
        adjusted[idx] = running
    return adjusted


# --------------------------------------------------------------------------
# Agreement
# --------------------------------------------------------------------------
def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> dict:
    """Cohen's kappa for two raters on the same items, nominal categories."""
    if len(labels_a) != len(labels_b):
        raise ValueError("rater label lists must have the same length")
    n = len(labels_a)
    if n == 0:
        raise ValueError("kappa over zero items")
    categories = sorted(set(labels_a) | set(labels_b))
    agree = sum(1 for x, y in zip(labels_a, labels_b) if x == y)
    po = agree / n
    pe = 0.0
    for cat in categories:
        pa = sum(1 for x in labels_a if x == cat) / n
        pb = sum(1 for y in labels_b if y == cat) / n
        pe += pa * pb
    if abs(1.0 - pe) < _TOL:
        raise ValueError("kappa undefined: expected agreement is 1.0 (all items in one category)")
    return {
        "kappa": (po - pe) / (1.0 - pe),
        "po": po,
        "pe": pe,
        "n": n,
        "n_categories": len(categories),
        "method": "cohen_kappa",
    }


def fleiss_kappa(ratings: Iterable[Sequence[str]]) -> dict:
    """Fleiss' kappa for m raters on the same items, nominal categories.

    `ratings` is one sequence per item, each holding that item's labels, one
    per rater. Every item must carry the same number of labels; an item with
    fewer raises rather than being folded in under one of the several
    unequal-n generalisations, none of which this study has validated. The
    caller decides what to do with an item a rater skipped -- E4 drops it from
    the panel and records the count, which is what `table_e4_kappa` already
    does for a model rater's failed call.

    Fixed marginals: the expected agreement is built from the pooled category
    proportions, which is what makes this Fleiss' coefficient rather than a
    free-marginal one (Randolph 2005). The two differ on exactly the data this
    study has -- four categories the annotators are not told to fill evenly --
    so `FLEISS_RANDOLPH_EXAMPLE` pins the difference, not just the formula.

    Returns the same shape as `cohen_kappa`: po is Fleiss' P-bar (mean
    within-item agreement), pe is P-bar-sub-e.
    """
    items = [list(item) for item in ratings]
    if not items:
        raise ValueError("fleiss_kappa over zero items")
    sizes = {len(item) for item in items}
    if len(sizes) != 1:
        raise ValueError(
            "fleiss_kappa needs the same number of raters on every item; got sizes "
            f"{sorted(sizes)}. Drop the incomplete items and say how many were dropped"
        )
    n_raters = sizes.pop()
    if n_raters < 2:
        raise ValueError(f"fleiss_kappa needs at least 2 raters per item, got {n_raters}")
    categories = sorted({label for item in items for label in item})
    n_items = len(items)

    agreement_sum = 0.0
    for item in items:
        counts = [sum(1 for label in item if label == cat) for cat in categories]
        agreement_sum += (sum(c * c for c in counts) - n_raters) / (n_raters * (n_raters - 1))
    po = agreement_sum / n_items

    total = n_items * n_raters
    pe = 0.0
    for cat in categories:
        p = sum(1 for item in items for label in item if label == cat) / total
        pe += p * p
    if abs(1.0 - pe) < _TOL:
        raise ValueError("fleiss_kappa undefined: expected agreement is 1.0 (all items in one category)")
    return {
        "kappa": (po - pe) / (1.0 - pe),
        "po": po,
        "pe": pe,
        "n_items": n_items,
        "n_raters": n_raters,
        "n_categories": len(categories),
        "method": "fleiss_kappa",
    }


def krippendorff_alpha(ratings: Iterable[Sequence[str | None]]) -> dict:
    """Krippendorff's alpha, nominal metric, missing values allowed.

    `ratings` is one sequence per unit of analysis, holding that unit's values
    with `None` (or an absent entry) where an observer did not label it. Units
    left with fewer than two values are not pairable and drop out of both
    disagreements, which is Krippendorff's own treatment and the reason this
    coefficient is in section 9 beside Fleiss' kappa: a sheet that comes back
    with a case missing costs that case, not the whole estimate.

    Computed from the coincidence matrix, in exact rational arithmetic
    (`fractions.Fraction`) so the result does not depend on the order the
    units arrive in. Two algebraically identical forms are evaluated and
    compared -- the disagreement ratio `1 - Do/De` and the closed form the
    source prints -- because a single path can agree with itself while being
    wrong.
    """
    from fractions import Fraction

    units = [[v for v in unit if v is not None] for unit in ratings]
    pairable = [unit for unit in units if len(unit) >= 2]
    if not pairable:
        raise ValueError("krippendorff_alpha: no unit carries two or more values")

    categories = sorted({v for unit in pairable for v in unit})
    index = {cat: i for i, cat in enumerate(categories)}
    k = len(categories)
    coincidences = [[Fraction(0) for _ in range(k)] for _ in range(k)]
    for unit in pairable:
        m = len(unit)
        counts = [0] * k
        for value in unit:
            counts[index[value]] += 1
        for c in range(k):
            if not counts[c]:
                continue
            for j in range(k):
                if not counts[j]:
                    continue
                pairs = counts[c] * (counts[j] - 1) if c == j else counts[c] * counts[j]
                if pairs:
                    coincidences[c][j] += Fraction(pairs, m - 1)

    margins = [sum(row) for row in coincidences]
    n = sum(margins)
    if n < 2:
        raise ValueError("krippendorff_alpha: fewer than two pairable values")
    observed_matches = sum(coincidences[c][c] for c in range(k))
    mismatch = n - observed_matches
    expected_mismatch = Fraction(n * n - sum(m * m for m in margins), n - 1)
    if expected_mismatch == 0:
        raise ValueError(
            "krippendorff_alpha undefined: expected disagreement is 0 "
            "(every pairable value is in one category)"
        )
    do = mismatch / n
    de = expected_mismatch / n
    alpha = 1 - do / de

    closed_form_num = (n - 1) * observed_matches - sum(m * (m - 1) for m in margins)
    closed_form_den = n * (n - 1) - sum(m * (m - 1) for m in margins)
    if closed_form_den == 0 or Fraction(closed_form_num, closed_form_den) != alpha:
        raise AssertionError(
            "krippendorff_alpha: the disagreement ratio and the closed form disagree "
            f"({alpha} vs {closed_form_num}/{closed_form_den}) -- this is a bug in this function"
        )

    return {
        "alpha": float(alpha),
        "observed_disagreement": float(do),
        "expected_disagreement": float(de),
        "category_counts": {cat: int(margins[i]) for cat, i in index.items()},
        "n_pairable": int(n),
        "n_units_pairable": len(pairable),
        "n_units_dropped": len(units) - len(pairable),
        "n_categories": k,
        "method": "krippendorff_alpha_nominal",
    }


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------
def _close(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(a - b) <= tol


def paired_coverage_simulation(
    cell_probs: tuple[float, float, float, float],
    n: int,
    seed: int,
    n_sim: int = 2000,
    alpha: float = 0.05,
    continuity: bool = True,
) -> dict:
    """Monte-Carlo coverage of `newcombe_paired_diff_ci`.

    `cell_probs` is (p_a, p_b, p_c, p_d) over the four paired outcomes, so the
    true risk difference is p_c - p_b. Draws `n_sim` tables of `n` pairs and
    reports how often the interval covers that true value. A correct 95%
    interval for a discrete outcome lands at or a little above 0.95; a badly
    wrong formula misses by far more than Monte-Carlo error.
    """
    pa, pb, pc, pd = cell_probs
    total = pa + pb + pc + pd
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"cell probabilities must sum to 1, got {total}")
    true_rd = pc - pb
    rng = random.Random(f"{seed}|coverage|{n}|{cell_probs}|{continuity}")
    cum = [pa, pa + pb, pa + pb + pc]
    covered = 0
    widths = 0.0
    for _ in range(n_sim):
        counts = [0, 0, 0, 0]
        for _ in range(n):
            u = rng.random()
            idx = 0 if u < cum[0] else 1 if u < cum[1] else 2 if u < cum[2] else 3
            counts[idx] += 1
        res = newcombe_paired_diff_ci(*counts, alpha=alpha, continuity=continuity)
        widths += res["ci_high"] - res["ci_low"]
        if res["ci_low"] <= true_rd <= res["ci_high"]:
            covered += 1
    return {
        "cell_probs": cell_probs,
        "true_rd": true_rd,
        "n": n,
        "n_sim": n_sim,
        "alpha": alpha,
        "continuity": continuity,
        "coverage": covered / n_sim,
        "mean_width": widths / n_sim,
        "seed": seed,
    }


def _coverage_report(n_sim: int = 20000) -> int:
    """`python -m experiments.stats --coverage`: the validation evidence for
    the paired interval, printed in full so the numbers can be quoted."""
    scenarios = [
        ("null, strong agreement", (0.85, 0.05, 0.05, 0.05)),
        ("small degradation", (0.75, 0.15, 0.05, 0.05)),
        ("large degradation", (0.60, 0.25, 0.05, 0.10)),
        ("weak agreement", (0.45, 0.20, 0.20, 0.15)),
        ("near ceiling", (0.90, 0.06, 0.02, 0.02)),
    ]
    print(f"Newcombe method 10 coverage, nominal 95%, n_sim={n_sim} per cell\n")
    print(f"{'scenario':<24}{'true rd':>9}{'n':>5}{'cc cover':>10}{'raw cover':>11}{'cc width':>10}")
    worst_cc = worst_raw = 1.0
    for name, probs in scenarios:
        for n in (30, 70):
            cc = paired_coverage_simulation(probs, n, seed=20260920, n_sim=n_sim, continuity=True)
            raw = paired_coverage_simulation(probs, n, seed=20260920, n_sim=n_sim, continuity=False)
            worst_cc = min(worst_cc, cc["coverage"])
            worst_raw = min(worst_raw, raw["coverage"])
            print(f"{name:<24}{cc['true_rd']:>9.2f}{n:>5}{cc['coverage']:>10.4f}"
                  f"{raw['coverage']:>11.4f}{cc['mean_width']:>10.4f}")
    print("\ncc = phi continuity correction on (the shipped default), raw = off")
    print(f"worst coverage, shipped default: {worst_cc:.4f}   uncorrected: {worst_raw:.4f}"
          f"   (nominal 0.95)")
    return 0 if worst_cc >= 0.94 else 1


def _selftest(cross_check: bool = False) -> int:
    checks: list[tuple[str, bool, str]] = []

    lo, hi = wilson_ci(8, 10)
    checks.append(("wilson_ci(8,10) == (0.4902, 0.9433)", _close(lo, 0.4902) and _close(hi, 0.9433), f"{lo:.4f},{hi:.4f}"))

    lo, hi = clopper_pearson_ci(8, 10)
    checks.append(("clopper_pearson_ci(8,10) == (0.4439, 0.9747)", _close(lo, 0.44392) and _close(hi, 0.97472), f"{lo:.5f},{hi:.5f}"))

    lo, hi = wilson_ci(0, 7)
    checks.append(("wilson_ci(0,7) lower bound is 0 and upper < 1", lo == 0.0 and hi < 1.0, f"{lo:.4f},{hi:.4f}"))

    p = exact_binomial_test(3, 10, 0.5)
    checks.append(("exact_binomial_test(3,10,0.5) == 0.34375", _close(p, 0.34375, 1e-9), f"{p:.6f}"))

    p = exact_binomial_test(5, 10, 0.5)
    checks.append(("exact_binomial_test(5,10,0.5) == 1.0", _close(p, 1.0, 1e-9), f"{p:.6f}"))

    res = mcnemar_exact(10, 2)
    checks.append(("mcnemar_exact(10,2).p == 0.038574", _close(res["p_value"], 2 * 79 / 4096, 1e-9), f"{res['p_value']:.6f}"))

    res = mcnemar_exact(0, 0)
    checks.append(("mcnemar_exact(0,0).p == 1.0", res["p_value"] == 1.0, str(res["p_value"])))

    adj = holm_adjust([0.01, 0.02, 0.03, 0.04])
    checks.append(("holm_adjust([.01,.02,.03,.04]) == [.04,.06,.06,.06]", all(_close(a, b, 1e-9) for a, b in zip(adj, [0.04, 0.06, 0.06, 0.06])), str([round(a, 4) for a in adj])))

    adj = bh_adjust([0.01, 0.02, 0.03, 0.04])
    checks.append(("bh_adjust([.01,.02,.03,.04]) == [.04,.04,.04,.04]", all(_close(a, 0.04, 1e-9) for a in adj), str([round(a, 4) for a in adj])))

    k = cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"])
    checks.append(("cohen_kappa with po==pe gives 0", _close(k["kappa"], 0.0, 1e-9), f"{k['kappa']:.6f}"))

    # 50 units, mean statistic: fine-grained enough that two seeds land on
    # different percentile endpoints (a 5-unit binary vector would tie at the
    # same boundary values for any seed and prove nothing).
    units = [[i / 49.0] for i in range(50)]
    mean_stat = lambda u: sum(x[0] for x in u) / len(u)  # noqa: E731
    b1 = bootstrap_ci(units, mean_stat, seed=20260920, n_boot=2000)
    b2 = bootstrap_ci(units, mean_stat, seed=20260920, n_boot=2000)
    checks.append(("bootstrap_ci is seed-deterministic", b1 == b2, f"{b1['ci_low']:.4f}-{b1['ci_high']:.4f}"))

    b3 = bootstrap_ci(units, mean_stat, seed=1, n_boot=2000)
    checks.append(("bootstrap_ci changes with the seed", b1["ci_low"] != b3["ci_low"] or b1["ci_high"] != b3["ci_high"], f"seed1={b3['ci_low']:.4f}-{b3['ci_high']:.4f}"))

    b4 = bootstrap_ci(units, mean_stat, seed=20260920, n_boot=2000, alpha=0.20)
    checks.append(("bootstrap_ci honours alpha (80% CI narrower than 95%)", (b4["ci_high"] - b4["ci_low"]) < (b1["ci_high"] - b1["ci_low"]), f"80%={b4['ci_low']:.4f}-{b4['ci_high']:.4f}"))

    # ---- Fleiss kappa, Randolph (2005) Tables 1 and 2 ---------------------
    # The paper prints counts, this module takes labels, so the counts are
    # expanded here rather than the function being taught a second input shape.
    fl = FLEISS_RANDOLPH_EXAMPLE
    ftol = fl["printed_tolerance"]

    def _expand(counts) -> list[list[str]]:
        return [["yes"] * yes + ["no"] * no for yes, no in counts]

    t1 = fleiss_kappa(_expand(fl["table1"]))
    checks.append((
        f"fleiss: {fl['source']} Table 1 -> po {fl['table1_po']}, pe {fl['table1_pe']}",
        _close(t1["po"], fl["table1_po"], ftol)
        and _close(t1["pe"], fl["table1_pe"], ftol)
        and _close(t1["kappa"], fl["table1_kappa_exact"], 1e-12),
        f"po={t1['po']:.6f} pe={t1['pe']:.6f} kappa={t1['kappa']:.6f}",
    ))
    checks.append((
        "fleiss: Table 1's printed .34 is recovered from the paper's own rounded po",
        _close((round(t1["po"], 2) - fl["table1_pe"]) / (1 - fl["table1_pe"]),
               fl["table1_kappa_printed"], ftol),
        f"({round(t1['po'], 2)} - {fl['table1_pe']}) / {1 - fl['table1_pe']}",
    ))

    t2 = fleiss_kappa(_expand(fl["table2"]))
    checks.append((
        f"fleiss: the same paper's Table 2 -> kappa {fl['table2_kappa']} on skewed marginals",
        _close(t2["po"], fl["table2_po"], ftol) and _close(t2["kappa"], fl["table2_kappa"], 1e-12),
        f"po={t2['po']:.6f} kappa={t2['kappa']:.6f}",
    ))
    # The discriminating row: free marginals give +0.33 on Table 2, which is
    # the wrong coefficient reading as a respectable positive agreement.
    checks.append((
        "fleiss: Table 2 is not the free-marginal .33 the same paper prints beside it",
        not _close(t2["kappa"], fl["free_marginal_both_tables"], 0.02),
        f"{t2['kappa']:.6f} vs free-marginal {fl['free_marginal_both_tables']}",
    ))
    checks.append((
        "fleiss: unanimous labels give kappa 1",
        _close(fleiss_kappa([["a"] * 3, ["b"] * 3, ["a"] * 3])["kappa"], 1.0, 1e-12),
        "",
    ))
    try:
        fleiss_kappa([["a", "b"], ["a", "b", "b"]])
        ragged_ok = False
    except ValueError:
        ragged_ok = True
    checks.append((
        "fleiss: unequal raters per item raises instead of guessing a generalisation",
        ragged_ok, "",
    ))

    # ---- Krippendorff alpha (nominal), Krippendorff (2011) ----------------
    kx = KRIPPENDORFF_2011_EXAMPLE
    ktol = kx["printed_tolerance"]
    units = [list(col) for col in zip(*kx["observers"])]
    ka = krippendorff_alpha(units)
    checks.append((
        f"krippendorff: {kx['source']} -> alpha_nominal {kx['alpha_nominal']}",
        _close(ka["alpha"], kx["alpha_nominal"], ktol),
        f"alpha={ka['alpha']:.6f}",
    ))
    checks.append((
        "krippendorff: the same example's printed margins (40 pairable values, 1 unpairable unit, "
        "category counts 9/13/10/5/3)",
        ka["n_pairable"] == kx["n_pairable"]
        and ka["n_units_dropped"] == kx["n_units_dropped"]
        and ka["category_counts"] == kx["category_counts"],
        f"n_pairable={ka['n_pairable']} dropped={ka['n_units_dropped']} counts={ka['category_counts']}",
    ))
    # Order independence is a property of the exact-rational coincidence
    # matrix, and the reason the margins above are asserted separately: a
    # missing-value rule can be wrong in a way that still sums to 40.
    shuffled = list(reversed([list(reversed(u)) for u in units]))
    checks.append((
        "krippendorff: alpha does not depend on the order of units or of observers",
        krippendorff_alpha(shuffled)["alpha"] == ka["alpha"],
        f"{krippendorff_alpha(shuffled)['alpha']:.12f}",
    ))
    checks.append((
        "krippendorff: perfect agreement gives alpha 1",
        _close(krippendorff_alpha([["a", "a"], ["b", "b"], ["a", "a"]])["alpha"], 1.0, 1e-12),
        "",
    ))
    checks.append((
        "krippendorff: a unit no two observers labelled is dropped, not scored",
        krippendorff_alpha([["a", "a"], ["b", "b"], ["a", None]])["n_units_dropped"] == 1,
        "",
    ))

    # ---- Fisher exact 2x2 -------------------------------------------------
    tea = TEA_TASTING_EXAMPLE
    a0, b0, c0, d0 = tea["table"]
    n0, row0, col0 = a0 + b0 + c0 + d0, a0 + b0, a0 + c0
    pmf = [math.comb(row0, x) * math.comb(c0 + d0, col0 - x) / math.comb(n0, col0)
           for x in range(0, min(row0, col0) + 1)]
    want_two = sum(pr for pr in pmf if pr <= pmf[a0] * (1 + 1e-7))
    want_gt = sum(pmf[x] for x in range(a0, len(pmf)))
    tea_got = fisher_exact_2x2(a0, b0, c0, d0)
    checks.append((
        "fisher: " + tea["source"] + " two-sided == 34/70",
        _close(tea_got["p_value"], tea["p_two_sided"], 1e-12) and _close(want_two, tea["p_two_sided"], 1e-12),
        f"{tea_got['p_value']:.9f}",
    ))
    tea_gt = fisher_exact_2x2(a0, b0, c0, d0, alternative="greater")
    checks.append((
        "fisher: the same table one-sided (greater) == 17/70",
        _close(tea_gt["p_value"], tea["p_greater"], 1e-12) and _close(want_gt, tea["p_greater"], 1e-12),
        f"{tea_gt['p_value']:.9f}",
    ))

    # Structural invariants. The first is the one E5 leans on: 7 out of 7
    # against 57/63 must not come back significant because the stratum is
    # small -- a small stratum is an interval problem, not a finding.
    small = fisher_exact_2x2(7, 0, 57, 6)
    checks.append(("fisher: 7/7 vs 57/63 is not significant (small-stratum sanity)",
                   small["p_value"] > 0.5, f"p={small['p_value']:.4f}"))
    swapped = fisher_exact_2x2(57, 6, 7, 0)
    checks.append(("fisher: transposing the two groups leaves the two-sided p unchanged",
                   _close(small["p_value"], swapped["p_value"], 1e-12), f"{swapped['p_value']:.6f}"))
    edgy = fisher_exact_2x2(4, 4, 60, 2)
    checks.append(("fisher: 4/8 vs 60/62 is significant and its difference is negative",
                   edgy["p_value"] < 0.01 and edgy["difference"] < 0,
                   f"p={edgy['p_value']:.6f} diff={edgy['difference']:.4f}"))
    balanced = fisher_exact_2x2(5, 5, 5, 5)
    checks.append(("fisher: an exactly balanced table gives p == 1",
                   _close(balanced["p_value"], 1.0, 1e-12), f"{balanced['p_value']:.6f}"))
    try:
        fisher_exact_2x2(1, 1, 1, 1, alternative="two_sided")
        alt_ok = False
    except ValueError:
        alt_ok = True
    checks.append(("fisher: an unknown alternative raises instead of defaulting", alt_ok, ""))

    # ---- Newcombe method 10, paired risk difference -----------------------
    # The paper's own printed worked example leads, because it is the only
    # check here that can catch a wrong constant rather than a wrong shape.
    # NEWCOMBE_TABLE3_EXAMPLE documents the e/f/g/h -> a/b/c/d translation.
    ex = NEWCOMBE_TABLE3_EXAMPLE
    tol = ex["tolerance"]
    want_lo, want_hi = ex["method10_ci"]
    paper = newcombe_paired_diff_ci(ex["e"], ex["g"], ex["f"], ex["h"])
    checks.append((f"newcombe: method 10 reproduces {ex['source']} -> ({want_lo}, {want_hi})",
                   _close(paper["rd"], ex["theta_hat"], tol)
                   and _close(paper["ci_low"], want_lo, tol)
                   and _close(paper["ci_high"], want_hi, tol),
                   f"rd={paper['rd']:.6f} ci=({paper['ci_low']:.6f},{paper['ci_high']:.6f}) phi={paper['phi']:.6f}"))

    want8_lo, want8_hi = ex["method8_ci"]
    paper8 = newcombe_paired_diff_ci(ex["e"], ex["g"], ex["f"], ex["h"], continuity=False)
    checks.append((f"newcombe: continuity=False reproduces the same table's method 8 row -> ({want8_lo}, {want8_hi})",
                   _close(paper8["ci_low"], want8_lo, tol) and _close(paper8["ci_high"], want8_hi, tol),
                   f"ci=({paper8['ci_low']:.6f},{paper8['ci_high']:.6f}) phi={paper8['phi']:.6f}"))

    # f and g transposed: same width, opposite sign. Were this ever to satisfy
    # the printed endpoints, the translation above would be reading the paper
    # backwards and every published rd would carry the wrong sign.
    flipped = newcombe_paired_diff_ci(ex["e"], ex["f"], ex["g"], ex["h"])
    checks.append(("newcombe: transposing f and g negates the interval and fails the printed values",
                   _close(flipped["rd"], -ex["theta_hat"], tol)
                   and _close(flipped["ci_low"], -want_hi, tol)
                   and _close(flipped["ci_high"], -want_lo, tol)
                   and not (_close(flipped["ci_low"], want_lo, tol)
                            and _close(flipped["ci_high"], want_hi, tol)),
                   f"rd={flipped['rd']:.6f} ci=({flipped['ci_low']:.6f},{flipped['ci_high']:.6f})"))

    # E0's own numbers: 64 cases right under both repeats, 6 wrong under both,
    # no discordant pair. The interval has to be centred on zero and narrow.
    zero = newcombe_paired_diff_ci(64, 0, 0, 6)
    checks.append(("newcombe: zero-difference table gives rd=0 and a symmetric interval",
                   _close(zero["rd"], 0.0, 1e-12) and _close(zero["ci_low"], -zero["ci_high"], 1e-12)
                   and zero["ci_low"] < 0 < zero["ci_high"],
                   f"rd=0 ci=({zero['ci_low']:.4f},{zero['ci_high']:.4f}) phi={zero['phi']:.4f}"))

    sym = newcombe_paired_diff_ci(40, 8, 8, 14)
    checks.append(("newcombe: b == c gives a symmetric interval around 0",
                   _close(sym["rd"], 0.0, 1e-12) and _close(sym["ci_low"], -sym["ci_high"], 1e-12),
                   f"({sym['ci_low']:.4f},{sym['ci_high']:.4f})"))

    asym = newcombe_paired_diff_ci(50, 12, 2, 6)
    checks.append(("newcombe: point estimate lies inside its own interval",
                   asym["ci_low"] <= asym["rd"] <= asym["ci_high"] and asym["rd"] < 0,
                   f"rd={asym['rd']:.4f} ci=({asym['ci_low']:.4f},{asym['ci_high']:.4f})"))

    raw = newcombe_paired_diff_ci(50, 12, 2, 6, continuity=False)
    checks.append(("newcombe: the phi continuity correction widens, never narrows",
                   (asym["ci_high"] - asym["ci_low"]) >= (raw["ci_high"] - raw["ci_low"]) - 1e-12
                   and asym["phi"] <= raw["phi"] + 1e-12,
                   f"cc width {asym['ci_high'] - asym['ci_low']:.4f} vs raw {raw['ci_high'] - raw['ci_low']:.4f}"))

    perfect = newcombe_paired_diff_ci(70, 0, 0, 0)
    checks.append(("newcombe: degenerate table (every pair a hit) stays inside [-1, 1]",
                   -1.0 <= perfect["ci_low"] <= 0 <= perfect["ci_high"] <= 1.0 and perfect["phi"] == 0.0,
                   f"({perfect['ci_low']:.4f},{perfect['ci_high']:.4f})"))

    cov_null = paired_coverage_simulation((0.85, 0.05, 0.05, 0.05), 70, seed=20260920, n_sim=1000)
    checks.append(("newcombe: 95% coverage at rd=0, n=70 (Monte-Carlo, 1000 draws)",
                   0.93 <= cov_null["coverage"] <= 1.0, f"{cov_null['coverage']:.4f}"))

    cov_eff = paired_coverage_simulation((0.75, 0.15, 0.05, 0.05), 70, seed=20260920, n_sim=1000)
    checks.append(("newcombe: 95% coverage at rd=-0.10, n=70 (Monte-Carlo, 1000 draws)",
                   0.93 <= cov_eff["coverage"] <= 1.0, f"{cov_eff['coverage']:.4f}"))

    # ---- conditional power simulation -------------------------------------
    null_pairs = [(1, 1)] * 64 + [(0, 0)] * 6
    pw_null = paired_power_simulation(null_pairs, (30, 50, 70), seed=20260920, label="selftest_null", n_sim=200)
    checks.append(("power: a zero-difference pairing has power 0 at every n",
                   all(r["power"] == 0.0 for r in pw_null),
                   ",".join(f"n={r['n']}:{r['power']:.2f}" for r in pw_null)))

    eff_pairs = [(1, 1)] * 52 + [(1, 0)] * 12 + [(0, 1)] * 1 + [(0, 0)] * 5
    pw_eff = paired_power_simulation(eff_pairs, (30, 50, 70), seed=20260920, label="selftest_eff", n_sim=200)
    powers = [r["power"] for r in pw_eff]
    checks.append(("power: rises with n on a 12-vs-1 discordant pairing",
                   powers == sorted(powers) and powers[0] < powers[-1],
                   ",".join(f"n={r['n']}:{r['power']:.2f}" for r in pw_eff)))

    pw_again = paired_power_simulation(eff_pairs, (70,), seed=20260920, label="selftest_eff", n_sim=200)
    pw_other = paired_power_simulation(eff_pairs, (70,), seed=20260921, label="selftest_eff", n_sim=200)
    checks.append(("power: seed-deterministic, and a different seed is a different draw",
                   pw_again[0]["power"] == powers[-1] and pw_other[0]["mean_b"] != pw_again[0]["mean_b"],
                   f"same={pw_again[0]['power']:.3f} mean_b {pw_again[0]['mean_b']:.3f} vs {pw_other[0]['mean_b']:.3f}"))

    if cross_check:
        try:
            from scipy import stats as sp  # type: ignore
        except ImportError as exc:
            checks.append(("scipy cross-check", False, f"scipy unavailable: {exc}"))
        else:
            sp_p = sp.binomtest(3, 10, 0.5).pvalue
            checks.append(("scipy binomtest agrees", _close(sp_p, exact_binomial_test(3, 10, 0.5), 1e-12), f"{sp_p:.8f}"))
            sp_ci = sp.binomtest(8, 10, 0.5).proportion_ci(method="exact")
            mine = clopper_pearson_ci(8, 10)
            checks.append(("scipy Clopper-Pearson agrees", _close(sp_ci.low, mine[0], 1e-6) and _close(sp_ci.high, mine[1], 1e-6), f"{sp_ci.low:.6f},{sp_ci.high:.6f}"))
            for table in ((3, 1, 1, 3), (7, 0, 57, 6), (4, 4, 60, 2), (10, 9, 54, 0)):
                sp_f = sp.fisher_exact([[table[0], table[1]], [table[2], table[3]]])[1]
                mine_f = fisher_exact_2x2(*table)["p_value"]
                checks.append((f"scipy fisher_exact agrees on {table}",
                               _close(sp_f, mine_f, 1e-12), f"{sp_f:.9f}"))
            sp_wil = sp.binomtest(8, 10, 0.5).proportion_ci(method="wilson")
            mine_w = wilson_ci(8, 10)
            checks.append(("scipy Wilson agrees", _close(sp_wil.low, mine_w[0], 1e-6) and _close(sp_wil.high, mine_w[1], 1e-6), f"{sp_wil.low:.6f},{sp_wil.high:.6f}"))

        # statsmodels is the only second implementation of Fleiss' kappa on this
        # machine. It is a cross-check and not the validation: the validation is
        # the printed Randolph tables above, which run with no third party
        # installed. Listed separately so a missing statsmodels fails its own row
        # rather than the scipy one.
        try:
            from statsmodels.stats.inter_rater import fleiss_kappa as sm_fleiss  # type: ignore
        except ImportError as exc:
            checks.append(("statsmodels cross-check", False, f"statsmodels unavailable: {exc}"))
        else:
            for name, counts in (("Randolph table 1", FLEISS_RANDOLPH_EXAMPLE["table1"]),
                                 ("Randolph table 2", FLEISS_RANDOLPH_EXAMPLE["table2"]),
                                 ("a 4-category table", ((2, 1, 0, 0), (0, 0, 3, 0), (1, 1, 1, 0), (0, 2, 0, 1)))):
                sm_value = float(sm_fleiss([list(row) for row in counts]))
                expanded: list[list[str]] = []
                for row in counts:
                    item: list[str] = []
                    for i, c in enumerate(row):
                        item.extend([str(i)] * c)
                    expanded.append(item)
                mine_fl = fleiss_kappa(expanded)["kappa"]
                checks.append((f"statsmodels fleiss_kappa agrees on {name}",
                               _close(sm_value, mine_fl, 1e-12), f"{sm_value:.9f}"))

    failures = 0
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"[{status}] {name}" + (f"  -> {detail}" if detail else ""))
    print(f"\n{len(checks) - failures}/{len(checks)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    import argparse

    # Parsed strictly, and abbreviations are off. The previous version tested
    # `"--cross-check" in sys.argv`, so `--corss-check` and `--self-check` both
    # ran the plain self-check and exited 0 -- a typo could not be told apart
    # from a pass.
    parser = argparse.ArgumentParser(
        prog="python -m experiments.stats",
        description="Self-check the standard-library estimators in this module.",
        allow_abbrev=False,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--cross-check",
        action="store_true",
        help="also compare three estimators against scipy (needs the analysis venv)",
    )
    mode.add_argument(
        "--coverage",
        action="store_true",
        help="Monte-Carlo coverage study of the paired risk-difference interval",
    )
    parser.add_argument(
        "--n-sim",
        type=int,
        default=20000,
        help="replicates per cell, --coverage only (default: 20000)",
    )
    args = parser.parse_args()
    if args.n_sim != parser.get_default("n_sim") and not args.coverage:
        parser.error("--n-sim applies to --coverage only")
    if args.coverage:
        raise SystemExit(_coverage_report(n_sim=args.n_sim))
    raise SystemExit(_selftest(cross_check=args.cross_check))
