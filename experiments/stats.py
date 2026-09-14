"""Statistics for the pilot experiments -- standard library only.

Every estimator in this module is implemented against the Python standard
library (`math.comb`, `statistics.NormalDist`, `random`). scipy/statsmodels
are installed in the analysis venv and are used only as an optional
cross-check (`python -m experiments.stats --cross-check`), never as the
source of a published number. Two reasons:

1. The analysis has to stay runnable on a machine where a cp314 wheel for
   scipy does not exist.
2. A reader of the write-up can follow every formula to a line of code here
   instead of to a third-party release.

Self-check against published worked examples:

    python -m experiments.stats            # asserts known values, prints OK
    python -m experiments.stats --cross-check   # additionally compares scipy

Not implemented yet (deliberately left as errors rather than approximations,
so nothing unvalidated can leak into a table):
  - Newcombe method 10 CI for a paired risk difference   -> newcombe_paired_diff_ci
  - Fleiss kappa / Krippendorff alpha                     -> fleiss_kappa, krippendorff_alpha
"""
from __future__ import annotations

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
    "mcnemar_exact",
    "bootstrap_ci",
    "holm_adjust",
    "bh_adjust",
    "cohen_kappa",
    "newcombe_paired_diff_ci",
    "fleiss_kappa",
    "krippendorff_alpha",
]

_TOL = 1e-12


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


def newcombe_paired_diff_ci(a: int, b: int, c: int, d: int, alpha: float = 0.05) -> tuple[float, float]:
    """TODO(E1, due 2026-09-21): Newcombe (1998) method 10 interval for the
    difference of two paired proportions.

    Left unimplemented on purpose. A wrong CI here would be invisible in the
    output table and would end up in a graduate-admission document, so the
    estimator ships only after it is checked against the worked example in
    Newcombe (1998), Stat Med 17:2635-2650. Until then analyze.py reports
    the McNemar exact p-value and the point estimate of the risk difference
    without an interval.
    """
    raise NotImplementedError(
        "newcombe_paired_diff_ci is not implemented yet (see docstring); "
        "use mcnemar_exact + the point estimate until it is validated"
    )


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


def fleiss_kappa(ratings: Iterable[Sequence[str]]) -> float:
    """TODO(E0, due 2026-09-19): Fleiss kappa across the five repeated runs."""
    raise NotImplementedError("fleiss_kappa is not implemented yet (E0 analysis, 2026-09-19)")


def krippendorff_alpha(ratings: Iterable[Sequence[str]]) -> float:
    """TODO(E0/E4, due 2026-09-24): Krippendorff alpha with missing values."""
    raise NotImplementedError("krippendorff_alpha is not implemented yet (E0/E4 analysis)")


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------
def _close(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(a - b) <= tol


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

    for name in ("newcombe_paired_diff_ci", "fleiss_kappa", "krippendorff_alpha"):
        fn = globals()[name]
        try:
            fn(1, 1, 1, 1) if name == "newcombe_paired_diff_ci" else fn([])
            ok = False
        except NotImplementedError:
            ok = True
        checks.append((f"{name} raises NotImplementedError (not a silent approximation)", ok, ""))

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
            sp_wil = sp.binomtest(8, 10, 0.5).proportion_ci(method="wilson")
            mine_w = wilson_ci(8, 10)
            checks.append(("scipy Wilson agrees", _close(sp_wil.low, mine_w[0], 1e-6) and _close(sp_wil.high, mine_w[1], 1e-6), f"{sp_wil.low:.6f},{sp_wil.high:.6f}"))

    failures = 0
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"[{status}] {name}" + (f"  -> {detail}" if detail else ""))
    print(f"\n{len(checks) - failures}/{len(checks)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_selftest(cross_check="--cross-check" in sys.argv))
