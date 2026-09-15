"""Analysis: raw JSONL in, CSV tables + PNG figures + MANIFEST.json out.

Reads only from experiments/results/raw/ and never calls an API, so it can be
re-run any number of times for free. Deterministic for a fixed --seed: the
same raw files produce byte-identical CSVs, which is what
checks/experiments_acceptance.sh asserts.

    python -m experiments.analyze --seed 20260920

What is implemented, self-checked against published worked examples in
`python -m experiments.stats` and compared with scipy wherever scipy has an
equivalent (`--cross-check`): Wilson interval, Clopper-Pearson exact interval,
exact McNemar, case-level cluster bootstrap, Holm and BH adjustment, Newcombe
method 10 CI for a paired risk difference, and conditional power by case
resampling. Run-to-run spread, per-case instability and judge re-score
stability are descriptive counts plus a sample standard deviation -- no
interval is attached to them.

The last two were listed here as unimplemented until 2026-09-15. The power
simulation shipped on 2026-09-14 with its own self-checks (a zero-difference
pairing has power 0 at every n, power rises with n on a 12-vs-1 discordant
pairing, and the same seed reproduces a draw while a different seed does
not). The Newcombe interval shipped on 2026-09-14 validated by structural
invariants and a Monte-Carlo coverage study, and its remaining gap was closed
on 2026-09-15 against the printed worked example in Newcombe (1998) Table III.
Both dates are entries in docs/PREREGISTRATION.md section 9, and every
Newcombe row carries experiments.stats.NEWCOMBE_VALIDATION in its note column.

What is still not implemented, and is therefore absent from the output rather
than approximated: Fleiss kappa, Krippendorff alpha, and the logistic model
with case-clustered standard errors. See the table in experiments/README.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median, stdev
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evalkit.dataset import load_all_cases  # noqa: E402
from evalkit.run_eval import DEFAULT_PASS_THRESHOLD  # noqa: E402
from experiments import cost_from_usage, price_key  # noqa: E402
from experiments.stats import (  # noqa: E402
    NEWCOMBE_VALIDATION,
    bh_adjust,
    bootstrap_ci,
    clopper_pearson_ci,
    holm_adjust,
    mcnemar_exact,
    newcombe_paired_diff_ci,
    paired_power_simulation,
    wilson_ci,
)

DEFAULT_SEED = 20260920
DEFAULT_N_BOOT = 10000
FLOAT_FMT = "{:.6f}"
BASELINE_PROMPT_VERSION = "v1"
JUDGE_PASS_SOURCE = "frozen_classifier_output"
NOT_IMPLEMENTED_NOTE = (
    "fleiss_kappa and krippendorff_alpha are not implemented; "
    "left empty rather than approximated (experiments/README.md)"
)

# ---- E1, frozen by docs/PREREGISTRATION.md section 10 ---------------------
# The paired baseline is E0's first three repeats, not a fresh v1 run, and the
# unit of analysis is the case (repeats collapsed by majority vote), not the
# (case, repeat) observation. Both are pre-registered decisions; changing
# either one here without changing section 10 breaks the pre-registration.
E1_BASELINE_EXP_ID = "e0_noise"
E1_BASELINE_REPEATS = (0, 1, 2)
E1_VERSIONS = ("v2a", "v2b", "v2c", "v2d")
E1_PRIMARY_METRIC = "category_match"
E1_POWER_SIZES = (30, 50, 70)
E1_POWER_N_SIM = 2000
# alpha = 0.05 is the nominal per-test level; 0.0125 is what Holm charges the
# first-rejected test in a family of four, i.e. the worst case for this design.
E1_POWER_ALPHAS = ((0.05, "nominal"), (0.0125, "holm_worst_case"))
E1_EMPTY_NOTE = "no E1 raw data under experiments/results/raw/ yet; header written, no rows"


def gate_thresholds() -> tuple[float, float]:
    """The production gate thresholds, read out of `evalkit/diff.py` rather
    than re-declared here -- a threshold change over there must not leave a
    figure in this package annotated with a stale line. Raises instead of
    guessing if the argparse defaults stop being readable."""
    src = (REPO_ROOT / "evalkit" / "diff.py").read_text(encoding="utf-8")
    warn = re.search(r'"WARN_THRESHOLD",\s*([0-9.]+)', src)
    crit = re.search(r'"CRITICAL_THRESHOLD",\s*([0-9.]+)', src)
    if not warn or not crit:
        raise SystemExit(
            "could not read the default thresholds from evalkit/diff.py -- "
            "the argparse defaults changed shape; fix gate_thresholds()"
        )
    return float(warn.group(1)), float(crit.group(1))


def fmt(value) -> str:
    """One float formatter for every table, so two runs cannot differ in the
    last digit of a repr."""
    if value is None:
        return ""
    if isinstance(value, float):
        return FLOAT_FMT.format(value)
    return str(value)


def rel_to_repo(path: Path) -> str:
    """Repo-relative path when possible, absolute otherwise -- the acceptance
    check runs the analysis into a temp directory outside the repo."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def row_cost(row: dict) -> float:
    """Cost of one raw call, recomputed here from the `usage` object the API
    returned and the pinned price table in `experiments/__init__.py`.

    The runner also writes a `cost_usd` field, and this function -- not that
    field -- is what every published cost number comes from. The two differ on
    calls that returned 200, burned tokens and then failed validation: the
    runner recorded $0 for those until 2026-09-14, and the E0 raw files are
    never rewritten (a raw file is evidence, not a working copy). MANIFEST.json
    carries both totals and their difference so the correction is visible
    rather than applied silently.
    """
    if row.get("cache_hit"):
        return 0.0
    usage = row.get("usage") or {}
    input_tokens = usage.get("input_tokens") or row.get("input_tokens") or 0
    output_tokens = usage.get("output_tokens") or row.get("output_tokens") or 0
    return cost_from_usage(
        price_key(row.get("response_model"), row.get("request_model") or ""),
        int(input_tokens),
        int(output_tokens),
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([fmt(v) for v in row])


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_raw(raw_dir: Path, exp_ids: list[str] | None) -> tuple[list[dict], list[Path]]:
    files = sorted(p for p in raw_dir.rglob("*.jsonl") if p.is_file())
    if exp_ids:
        files = [p for p in files if p.parent.name in exp_ids]
    rows: list[dict] = []
    for path in files:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno} is not valid JSON: {exc}") from exc
            row["_source_file"] = str(path.relative_to(REPO_ROOT))
            rows.append(row)
    return rows, files


def build_observations(rows: list[dict], dataset_path: Path) -> list[dict]:
    """Join the classifier row and the judge row of one (run, repeat, case)
    into one scored observation.

    A case whose classifier call failed produces no observation and is
    counted as a failure instead -- it is never silently scored as a miss,
    because an API error and a wrong answer are different things.
    """
    cases = {c.id: c for c in load_all_cases(dataset_path)}
    grouped: dict[tuple, dict] = defaultdict(dict)
    for row in rows:
        key = (row["exp_id"], row["run_id"], row["repeat_idx"], row["case_id"])
        grouped[key][row["tier"]] = row

    observations: list[dict] = []
    for key in sorted(grouped, key=lambda k: (k[0], k[1], k[2], k[3])):
        exp_id, run_id, repeat_idx, case_id = key
        clf = grouped[key].get("classifier")
        judge = grouped[key].get("judge")
        case = cases.get(case_id)
        if case is None:
            raise SystemExit(f"raw data references case_id={case_id!r} that is not in {dataset_path}")
        obs = {
            "exp_id": exp_id,
            "run_id": run_id,
            "repeat_idx": repeat_idx,
            "case_id": case_id,
            "language": case.language,
            "difficulty": case.expected_difficulty or case.draft_difficulty or "unknown",
            "expected_category": case.expected_category,
            "prompt_version": (clf or judge or {}).get("prompt_version"),
            "classifier_ok": bool(clf and clf.get("ok")),
            "judge_ok": bool(judge and judge.get("ok")),
            "predicted_category": (clf or {}).get("parsed", {}).get("category") if clf and clf.get("ok") else None,
            "predicted_summary": (clf or {}).get("parsed", {}).get("summary") if clf and clf.get("ok") else None,
            "judged_summary_source": (judge or {}).get("judged_summary_source"),
            "judge_score": (judge or {}).get("parsed", {}).get("score") if judge and judge.get("ok") else None,
            "classifier_input_tokens": (clf or {}).get("input_tokens"),
            "classifier_output_tokens": (clf or {}).get("output_tokens"),
            "judge_input_tokens": (judge or {}).get("input_tokens"),
            "judge_output_tokens": (judge or {}).get("output_tokens"),
            "classifier_latency_ms": (clf or {}).get("latency_ms"),
            "judge_latency_ms": (judge or {}).get("latency_ms"),
            "cost_usd": round(
                (row_cost(clf) if clf else 0.0) + (row_cost(judge) if judge else 0.0), 8
            ),
            "classifier_response_model": (clf or {}).get("response_model"),
            "judge_response_model": (judge or {}).get("response_model"),
        }
        obs["category_match"] = (
            None if not obs["classifier_ok"] else int(obs["predicted_category"] == obs["expected_category"])
        )
        obs["passed"] = (
            None
            if (obs["category_match"] is None or obs["judge_score"] is None)
            else int(bool(obs["category_match"]) and obs["judge_score"] >= DEFAULT_PASS_THRESHOLD)
        )
        observations.append(obs)
    return observations


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------
METRICS = ("category_match", "passed")


def _wrap(text: str, width: int = 90) -> str:
    """Figure footnotes carry n, CI method and seed, so they are long enough to
    run off the canvas if left on one line."""
    import textwrap

    return "\n".join(textwrap.wrap(text, width=width))


def table_per_case(observations: list[dict]) -> tuple[list[str], list[list]]:
    header = [
        "exp_id", "run_id", "prompt_version", "repeat_idx", "case_id", "language", "difficulty",
        "expected_category", "predicted_category", "category_match", "judge_score", "passed",
        "classifier_ok", "judge_ok", "judged_summary_source",
        "classifier_input_tokens", "classifier_output_tokens", "judge_input_tokens", "judge_output_tokens",
        "classifier_latency_ms", "judge_latency_ms", "cost_usd",
        "classifier_response_model", "judge_response_model",
    ]
    rows = [[obs.get(col) for col in header] for obs in observations]
    return header, rows


def _rate_rows(label: dict, values: list[int]) -> list[list]:
    n = len(values)
    k = sum(values)
    out = []
    for method, ci in (("wilson", wilson_ci(k, n)), ("clopper_pearson", clopper_pearson_ci(k, n))):
        out.append(
            [label["exp_id"], label["run_id"], label["prompt_version"], label["repeat_idx"], label["metric"],
             n, k, k / n, ci[0], ci[1], method]
        )
    return out


def table_rates(observations: list[dict]) -> tuple[list[str], list[list]]:
    header = ["exp_id", "run_id", "prompt_version", "repeat_idx", "metric", "n", "k", "estimate", "ci_low", "ci_high", "method"]
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for obs in observations:
        for metric in METRICS:
            if obs[metric] is None:
                continue
            buckets[(obs["exp_id"], obs["run_id"], obs["prompt_version"], obs["repeat_idx"], metric)].append(obs[metric])
            buckets[(obs["exp_id"], obs["run_id"], obs["prompt_version"], "pooled", metric)].append(obs[metric])
    rows: list[list] = []
    for key in sorted(buckets, key=lambda k: (k[0], k[1], str(k[2]), str(k[3]), k[4])):
        exp_id, run_id, pv, repeat_idx, metric = key
        rows.extend(_rate_rows({"exp_id": exp_id, "run_id": run_id, "prompt_version": pv, "repeat_idx": repeat_idx, "metric": metric}, buckets[key]))
    return header, rows


def table_strata(observations: list[dict]) -> tuple[list[str], list[list]]:
    header = ["exp_id", "run_id", "prompt_version", "stratum_kind", "stratum", "metric", "n", "k", "estimate", "ci_low", "ci_high", "method"]
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for obs in observations:
        for metric in METRICS:
            if obs[metric] is None:
                continue
            for kind in ("language", "difficulty"):
                buckets[(obs["exp_id"], obs["run_id"], obs["prompt_version"], kind, obs[kind], metric)].append(obs[metric])
    rows: list[list] = []
    for key in sorted(buckets, key=lambda k: tuple(str(x) for x in k)):
        exp_id, run_id, pv, kind, stratum, metric = key
        values = buckets[key]
        n, k = len(values), sum(values)
        lo, hi = wilson_ci(k, n)
        rows.append([exp_id, run_id, pv, kind, stratum, metric, n, k, k / n, lo, hi, "wilson"])
    return header, rows


def table_bootstrap(observations: list[dict], seed: int, n_boot: int) -> tuple[list[str], list[list]]:
    header = ["exp_id", "run_id", "prompt_version", "metric", "n_cases", "n_observations", "estimate", "ci_low", "ci_high", "method", "seed", "n_boot"]
    grouped: dict[tuple, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for obs in observations:
        for metric in METRICS:
            if obs[metric] is None:
                continue
            grouped[(obs["exp_id"], obs["run_id"], obs["prompt_version"], metric)][obs["case_id"]].append(obs[metric])
    rows: list[list] = []
    for key in sorted(grouped, key=lambda k: (k[0], k[1], str(k[2]), k[3])):
        clusters = [grouped[key][cid] for cid in sorted(grouped[key])]
        res = bootstrap_ci(
            clusters,
            lambda units: sum(sum(u) for u in units) / sum(len(u) for u in units),
            seed=seed,
            n_boot=n_boot,
        )
        rows.append([
            key[0], key[1], key[2], key[3], res["n_units"], sum(len(c) for c in clusters),
            res["estimate"], res["ci_low"], res["ci_high"], res["method"], seed, n_boot,
        ])
    return header, rows


def table_mcnemar(observations: list[dict]) -> tuple[list[str], list[list]]:
    """Paired comparison of each non-baseline prompt version against v1,
    pairing every (case, repeat) observation. Empty until a v2* run exists.

    **This is the sensitivity view, not the confirmatory test.** Three repeats
    of one case are not three independent pairs, so treating 210 observations
    as 210 pairs makes the exact McNemar p-value anticonservative. The
    confirmatory family collapses repeats to one value per case first and
    lives in `e1_main.csv` (docs/PREREGISTRATION.md section 10). Keeping this
    table lets a reader see how much the choice of unit moved the answer.
    """
    header = [
        "metric", "baseline_version", "candidate_version", "n_pairs",
        "both_correct", "baseline_only_correct", "candidate_only_correct", "both_wrong",
        "risk_difference", "p_value", "p_value_holm", "method", "family", "ci_note",
    ]
    by_version: dict[str, dict[tuple, int]] = defaultdict(dict)
    for obs in observations:
        for metric in METRICS:
            if obs[metric] is None:
                continue
            by_version[obs["prompt_version"]][(metric, obs["case_id"], obs["repeat_idx"])] = obs[metric]

    versions = sorted(v for v in by_version if v != BASELINE_PROMPT_VERSION)
    if BASELINE_PROMPT_VERSION not in by_version or not versions:
        return header, []

    raw_rows: list[list] = []
    for metric in METRICS:
        for candidate in versions:
            base = by_version[BASELINE_PROMPT_VERSION]
            cand = by_version[candidate]
            keys = sorted(k for k in base if k[0] == metric and k in cand)
            if not keys:
                continue
            a = sum(1 for k in keys if base[k] == 1 and cand[k] == 1)
            b = sum(1 for k in keys if base[k] == 1 and cand[k] == 0)
            c = sum(1 for k in keys if base[k] == 0 and cand[k] == 1)
            d = sum(1 for k in keys if base[k] == 0 and cand[k] == 0)
            res = mcnemar_exact(b, c)
            rd = (c - b) / len(keys)
            raw_rows.append([
                metric, BASELINE_PROMPT_VERSION, candidate, len(keys), a, b, c, d, rd,
                res["p_value"], None, res["method"], "sensitivity_per_repeat_pairing",
                "no interval and no multiplicity adjustment here: repeats of one case are "
                "not independent pairs; the confirmatory test is in e1_main.csv",
            ])
    return header, raw_rows


# --------------------------------------------------------------------------
# E1: the confirmatory comparison (docs/PREREGISTRATION.md section 10)
# --------------------------------------------------------------------------
def majority_vote(values: list[int]) -> int | None:
    """Collapse one case's repeats into one outcome.

    Returns None for an empty list and for a tie. A tie can only happen when a
    failed call has already removed one of an odd number of repeats, and
    section 10 excludes those cases from the pairing instead of inventing a
    rule for them; the count of exclusions is reported next to n.
    """
    if not values:
        return None
    ones = sum(values)
    zeros = len(values) - ones
    if ones == zeros:
        return None
    return int(ones > zeros)


def _case_outcomes(
    observations: list[dict], metric: str, prompt_version: str,
    exp_id: str | None = None, repeats: Sequence[int] | None = None,
) -> tuple[dict[str, int], int]:
    """One outcome per case for one prompt version, with the number of cases
    dropped for a tied vote."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for obs in observations:
        if obs["prompt_version"] != prompt_version or obs[metric] is None:
            continue
        if exp_id is not None and obs["exp_id"] != exp_id:
            continue
        if repeats is not None and obs["repeat_idx"] not in repeats:
            continue
        buckets[obs["case_id"]].append(obs[metric])
    out: dict[str, int] = {}
    ties = 0
    for case_id, values in buckets.items():
        vote = majority_vote(values)
        if vote is None:
            ties += 1
            continue
        out[case_id] = vote
    return out, ties


def e1_pair_sets(observations: list[dict]) -> dict[tuple[str, str], dict]:
    """Build the pre-registered pairing: baseline = E0 repeats 0-2 of v1,
    candidates = whichever of v2a..v2d have raw data. Returns {} when none do,
    which is how the whole E1 block gets skipped rather than raising."""
    out: dict[tuple[str, str], dict] = {}
    present = sorted({obs["prompt_version"] for obs in observations} & set(E1_VERSIONS))
    if not present:
        return out
    for metric in METRICS:
        base, base_ties = _case_outcomes(
            observations, metric, BASELINE_PROMPT_VERSION,
            exp_id=E1_BASELINE_EXP_ID, repeats=E1_BASELINE_REPEATS,
        )
        for version in present:
            cand, cand_ties = _case_outcomes(observations, metric, version)
            shared = sorted(set(base) & set(cand))
            out[(metric, version)] = {
                "pairs": [(cid, base[cid], cand[cid]) for cid in shared],
                "n_tie": base_ties + cand_ties,
                "n_missing": len(set(base) ^ set(cand)),
                "baseline_label": (
                    f"{BASELINE_PROMPT_VERSION} ({E1_BASELINE_EXP_ID} repeats "
                    f"{'+'.join(str(r) for r in E1_BASELINE_REPEATS)})"
                ),
            }
    return out


def paired_cells(pairs: Sequence[tuple[str, int, int]]) -> dict:
    a = sum(1 for _, x, y in pairs if x == 1 and y == 1)
    b = sum(1 for _, x, y in pairs if x == 1 and y == 0)
    c = sum(1 for _, x, y in pairs if x == 0 and y == 1)
    d = sum(1 for _, x, y in pairs if x == 0 and y == 0)
    return {"a": a, "b": b, "c": c, "d": d, "n": a + b + c + d}


E1_MAIN_HEADER = [
    "metric", "baseline_version", "candidate_version", "n", "n_excluded_tie", "n_excluded_missing",
    "pass_base", "pass_deg", "a", "b", "c", "d",
    "p_raw", "p_holm", "p_bh", "rd", "ci_low", "ci_high", "or",
    "family", "method", "ci_method", "note",
]


def table_e1_main(pair_sets: dict[tuple[str, str], dict]) -> tuple[list[str], list[list]]:
    """One row per (metric, degraded version): the confirmatory test.

    Holm runs across the four primary-metric tests and nothing else, which is
    the family fixed in docs/PREREGISTRATION.md section 4. The secondary
    metric gets Benjamini-Hochberg within its own exploratory family and keeps
    `p_holm` empty, so no reader can mistake one for the other.
    """
    if not pair_sets:
        return E1_MAIN_HEADER, []
    rows: list[list] = []
    index_by_metric: dict[str, list[int]] = defaultdict(list)
    for (metric, version) in sorted(pair_sets):
        info = pair_sets[(metric, version)]
        pairs = info["pairs"]
        if not pairs:
            continue
        cells = paired_cells(pairs)
        a, b, c, d, n = cells["a"], cells["b"], cells["c"], cells["d"], cells["n"]
        mc = mcnemar_exact(b, c)
        ci = newcombe_paired_diff_ci(a, b, c, d)
        odds = (c / b) if b > 0 else None
        primary = metric == E1_PRIMARY_METRIC
        index_by_metric[metric].append(len(rows))
        rows.append([
            metric, info["baseline_label"], version, n, info["n_tie"], info["n_missing"],
            a + b, a + c, a, b, c, d,
            mc["p_value"], None, None, ci["rd"], ci["ci_low"], ci["ci_high"], odds,
            "confirmatory_holm" if primary else "exploratory_bh",
            mc["method"], ci["method"],
            (NEWCOMBE_VALIDATION if primary else NEWCOMBE_VALIDATION + "; secondary metric, exploratory")
            + ("" if b > 0 else "; odds ratio undefined (b = 0)"),
        ])
    p_col = E1_MAIN_HEADER.index("p_raw")
    for metric, idxs in index_by_metric.items():
        pvals = [rows[i][p_col] for i in idxs]
        adjusted = holm_adjust(pvals) if metric == E1_PRIMARY_METRIC else bh_adjust(pvals)
        target = E1_MAIN_HEADER.index("p_holm" if metric == E1_PRIMARY_METRIC else "p_bh")
        for i, value in zip(idxs, adjusted):
            rows[i][target] = value
    return E1_MAIN_HEADER, rows


E1_POWER_HEADER = [
    "metric", "candidate_version", "n", "alpha", "alpha_basis", "power", "n_sim", "seed",
    "n_observed_pairs", "observed_b", "observed_c", "mean_b", "mean_c", "method", "note",
]


def table_e1_power(pair_sets: dict[tuple[str, str], dict], seed: int, n_sim: int = E1_POWER_N_SIM) -> tuple[list[str], list[list]]:
    """Conditional power at n = 30 / 50 / 70 for the primary metric.

    Simulated, and labelled as such in every row: the n = 70 column is a
    resample of the observed pairs, not a second experiment. The realized test
    at n = 70 is the `p_holm` column of `e1_main.csv`.
    """
    if not pair_sets:
        return E1_POWER_HEADER, []
    rows: list[list] = []
    for (metric, version) in sorted(pair_sets):
        if metric != E1_PRIMARY_METRIC:
            continue
        pairs = [(x, y) for _, x, y in pair_sets[(metric, version)]["pairs"]]
        if not pairs:
            continue
        results = paired_power_simulation(
            pairs, E1_POWER_SIZES, seed=seed, label=f"{metric}|{version}",
            n_sim=n_sim, alphas=tuple(a for a, _ in E1_POWER_ALPHAS),
        )
        basis = dict(E1_POWER_ALPHAS)
        for res in results:
            rows.append([
                metric, version, res["n"], res["alpha"], basis[res["alpha"]], res["power"],
                res["n_sim"], res["seed"], res["n_observed_pairs"], res["observed_b"], res["observed_c"],
                res["mean_b"], res["mean_c"], res["method"],
                "simulated by resampling the observed pairs with replacement; "
                "conditional on the observed effect, not a design power calculation",
            ])
    return E1_POWER_HEADER, rows


def figure_e1_forest(main_rows: list[list], out_path: Path, seed: int) -> str | None:
    """Risk difference with its 95% interval, one line per degraded version.
    Primary metric filled, secondary metric open, never merged."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover -- documented fallback
        print(f"!! matplotlib unavailable ({exc}); CSV tables were still written", file=sys.stderr)
        return None
    if not main_rows:
        return None
    col = {name: i for i, name in enumerate(E1_MAIN_HEADER)}
    primary = [r for r in main_rows if r[col["metric"]] == E1_PRIMARY_METRIC]
    secondary = [r for r in main_rows if r[col["metric"]] != E1_PRIMARY_METRIC]
    if not primary:
        return None
    versions = [r[col["candidate_version"]] for r in primary]
    ypos = list(range(len(versions)))[::-1]
    footnote = _wrap(
        f"n={primary[0][col['n']]} paired cases per row, repeats collapsed by majority vote; baseline "
        f"{primary[0][col['baseline_version']]}; 95% interval: {primary[0][col['note']]}; "
        f"exact McNemar, Holm across the four primary tests only; analysis seed={seed}.",
        width=104,
    )
    footnote_lines = footnote.count("\n") + 1

    fig, ax = plt.subplots(figsize=(7.2, 1.1 * len(versions) + 2.0 + 0.16 * footnote_lines))
    for series, offset, style in (
        (primary, 0.0, dict(fmt="o", color="#1f4e79", label=f"{E1_PRIMARY_METRIC} (primary, Holm)")),
        (secondary, -0.22, dict(fmt="s", markerfacecolor="none", color="#7c4a2d",
                                label="passed (secondary, exploratory)")),
    ):
        if not series:
            continue
        order = {r[col["candidate_version"]]: i for i, r in enumerate(series)}
        xs, ys, lo, hi = [], [], [], []
        for y, version in zip(ypos, versions):
            if version not in order:
                continue
            r = series[order[version]]
            xs.append(r[col["rd"]])
            ys.append(y + offset)
            lo.append(r[col["rd"]] - r[col["ci_low"]])
            hi.append(r[col["ci_high"]] - r[col["rd"]])
        ax.errorbar(xs, ys, xerr=[lo, hi], capsize=4, **style)

    ax.axvline(0.0, color="#555555", lw=1.0)
    ax.set_ylim(-0.6, len(versions) - 0.4)
    ax.set_yticks(ypos)
    ax.set_yticklabels([v if len(v) <= 14 else v[:13] + "…" for v in versions])
    ax.set_xlabel("risk difference (degraded - baseline)")
    ax.set_title("E1: effect of each degradation on the primary metric")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=7, loc="best", framealpha=0.95)
    for y, r in zip(ypos, primary):
        p_holm = r[col["p_holm"]]
        label = f"n={r[col['n']]}  b={r[col['b']]}, c={r[col['c']]}"
        if isinstance(p_holm, float):
            label += f"  p_Holm={p_holm:.3f}"
        ax.annotate(label, (0.02, y + 0.18), xycoords=("axes fraction", "data"), fontsize=7, color="#333333")
    fig.text(0.01, 0.01, footnote, fontsize=7, va="bottom")
    fig.subplots_adjust(bottom=(0.55 + 0.14 * footnote_lines) / fig.get_figheight(), left=0.19, right=0.97, top=0.90)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, metadata={"Software": "experiments/analyze.py"})
    plt.close(fig)
    return rel_to_repo(out_path)


def figure_e1_power(power_rows: list[list], main_rows: list[list], out_path: Path, seed: int) -> str | None:
    """Conditional power curve. Simulated points are open markers on a dashed
    line; the one measured point per version -- did the test that actually ran
    at n = 70 reject? -- is a filled marker at 0 or 1. Two different
    quantities, two different markers, never averaged together."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover -- documented fallback
        print(f"!! matplotlib unavailable ({exc}); CSV tables were still written", file=sys.stderr)
        return None
    if not power_rows:
        return None
    pcol = {name: i for i, name in enumerate(E1_POWER_HEADER)}
    mcol = {name: i for i, name in enumerate(E1_MAIN_HEADER)}
    nominal = [r for r in power_rows if r[pcol["alpha_basis"]] == "nominal"]
    if not nominal:
        return None
    versions = sorted({r[pcol["candidate_version"]] for r in nominal})
    colours = ["#1f4e79", "#7c4a2d", "#2e6f40", "#8a2b5e", "#555555"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, version in enumerate(versions):
        pts = sorted((r for r in nominal if r[pcol["candidate_version"]] == version), key=lambda r: r[pcol["n"]])
        colour = colours[i % len(colours)]
        ax.plot([r[pcol["n"]] for r in pts], [r[pcol["power"]] for r in pts],
                marker="o", mfc="none", ls="--", color=colour, label=f"{version} simulated")
        realized = [r for r in main_rows
                    if r[mcol["candidate_version"]] == version and r[mcol["metric"]] == E1_PRIMARY_METRIC]
        if realized and isinstance(realized[0][mcol["p_holm"]], float):
            rejected = 1.0 if realized[0][mcol["p_holm"]] < 0.05 else 0.0
            # nudge the measured markers apart: versions that agree would
            # otherwise stack into one diamond at the same coordinates
            jitter = (i - (len(versions) - 1) / 2) * 0.9
            ax.plot([max(E1_POWER_SIZES) + jitter], [rejected], marker="D", ms=7, ls="none",
                    color=colour, label=f"{version} measured (1 = rejected)")
    ax.axhline(0.8, color="#999999", lw=1.0, ls=":", label="80% power")
    ax.set_xticks(list(E1_POWER_SIZES))
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("cases (n)")
    ax.set_ylabel("share of simulated studies that reject")
    ax.set_title("E1: conditional power of the 70-case gate")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, framealpha=0.95)
    n_sim = nominal[0][pcol["n_sim"]]
    footnote = _wrap(
        f"Open markers: {n_sim} resamples of the observed paired cases at each n, exact McNemar at "
        f"alpha=0.05, analysis seed={seed} -- conditional on the effect this study observed, not a "
        f"design power calculation. Filled diamonds: the single test that actually ran at n=70, plotted "
        f"as 1 if it rejected under Holm and 0 if it did not; one realized decision is not a rate. "
        f"Holm's worst-case level (alpha=0.0125) is in e1_power.csv.",
        width=104,
    )
    fig.text(0.01, 0.01, footnote, fontsize=7, va="bottom")
    fig.tight_layout(rect=(0, 0.08 + 0.035 * footnote.count("\n"), 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, metadata={"Software": "experiments/analyze.py"})
    plt.close(fig)
    return rel_to_repo(out_path)


def table_tokens(rows: list[dict]) -> tuple[list[str], list[list]]:
    header = [
        "exp_id", "run_id", "tier", "request_model", "response_model", "n_calls", "n_ok", "n_failed",
        "input_tokens_mean", "input_tokens_median", "output_tokens_mean", "output_tokens_median",
        "latency_ms_p50", "latency_ms_p95", "cost_usd_total", "cost_usd_per_call",
    ]
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["exp_id"], row["run_id"], row["tier"], row["request_model"], str(row.get("response_model")))].append(row)
    out: list[list] = []
    for key in sorted(buckets, key=lambda k: tuple(str(x) for x in k)):
        group = buckets[key]
        ok = [r for r in group if r["ok"]]
        ins = [r["input_tokens"] for r in ok]
        outs = [r["output_tokens"] for r in ok]
        lats = sorted(r["latency_ms"] for r in ok if r["latency_ms"] is not None)
        total = sum(row_cost(r) for r in group)
        p50 = median(lats) if lats else None
        p95 = lats[min(len(lats) - 1, int(round(0.95 * (len(lats) - 1))))] if lats else None
        out.append([
            key[0], key[1], key[2], key[3], key[4], len(group), len(ok), len(group) - len(ok),
            sum(ins) / len(ins) if ins else None, median(ins) if ins else None,
            sum(outs) / len(outs) if outs else None, median(outs) if outs else None,
            p50, p95, total, total / len(group) if group else None,
        ])
    return header, out


def _sd(values: list[float]) -> float | None:
    return stdev(values) if len(values) > 1 else None


def table_case_instability(observations: list[dict]) -> tuple[list[str], list[list]]:
    """Per case, across the repeats of one run: did the same input produce the
    same verdict every time? A case that comes out 4/5 is the unit of
    run-to-run noise a regression gate has to see past."""
    header = [
        "exp_id", "run_id", "prompt_version", "metric", "case_id", "language", "difficulty",
        "n_obs", "k", "rate", "unanimous",
    ]
    buckets: dict[tuple, list[int]] = defaultdict(list)
    meta: dict[tuple, tuple] = {}
    for obs in observations:
        for metric in METRICS:
            if obs[metric] is None:
                continue
            key = (obs["exp_id"], obs["run_id"], obs["prompt_version"], metric, obs["case_id"])
            buckets[key].append(obs[metric])
            meta[key] = (obs["language"], obs["difficulty"])
    rows = []
    for key in sorted(buckets, key=lambda k: tuple(str(x) for x in k)):
        values = buckets[key]
        n, k = len(values), sum(values)
        language, difficulty = meta[key]
        rows.append([key[0], key[1], key[2], key[3], key[4], language, difficulty,
                     n, k, k / n, int(k in (0, n))])
    return header, rows


def table_run_spread(observations: list[dict]) -> tuple[list[str], list[list]]:
    """The noise floor itself: how far apart repeats of the *same* prompt land,
    put next to the deltas the production gate reacts to."""
    warn, crit = gate_thresholds()
    header = [
        "exp_id", "run_id", "prompt_version", "metric", "n_runs", "n_per_run",
        "mean", "sd_sample", "min", "max", "range", "max_pairwise_abs_delta",
        "warn_threshold", "critical_threshold",
        "runs_over_warn_vs_first", "runs_over_critical_vs_first",
        "n_cases", "unstable_cases", "instability_rate",
    ]
    per_run: dict[tuple, dict] = defaultdict(lambda: defaultdict(list))
    per_case: dict[tuple, dict] = defaultdict(lambda: defaultdict(list))
    for obs in observations:
        for metric in METRICS:
            if obs[metric] is None:
                continue
            key = (obs["exp_id"], obs["run_id"], obs["prompt_version"], metric)
            per_run[key][obs["repeat_idx"]].append(obs[metric])
            per_case[key][obs["case_id"]].append(obs[metric])
    rows = []
    for key in sorted(per_run, key=lambda k: tuple(str(x) for x in k)):
        by_repeat = per_run[key]
        repeats = sorted(by_repeat)
        rates = [sum(by_repeat[r]) / len(by_repeat[r]) for r in repeats]
        sizes = sorted({len(by_repeat[r]) for r in repeats})
        first = rates[0]
        pairwise = max((abs(a - b) for i, a in enumerate(rates) for b in rates[i + 1:]), default=0.0)
        cases = per_case[key]
        unstable = sum(1 for cid in cases if sum(cases[cid]) not in (0, len(cases[cid])))
        rows.append([
            key[0], key[1], key[2], key[3], len(repeats),
            "|".join(str(s) for s in sizes),
            sum(rates) / len(rates), _sd(rates), min(rates), max(rates),
            max(rates) - min(rates), pairwise, warn, crit,
            sum(1 for r in rates[1:] if abs(r - first) > warn),
            sum(1 for r in rates[1:] if abs(r - first) > crit),
            len(cases), unstable, unstable / len(cases) if cases else None,
        ])
    return header, rows


def table_judge_rescore(rows_raw: list[dict]) -> tuple[list[str], list[list]]:
    """Judge-isolation arm only: the classifier output is frozen, so every
    score difference on a case is the judge disagreeing with itself. Rows whose
    `judged_summary_source` is not the frozen one are excluded, because there
    the summary changed too and the two sources cannot be separated."""
    header = [
        "exp_id", "run_id", "case_id", "n_scores", "scores", "distinct_scores", "all_equal",
        "mean", "sd_sample", "min", "max", "n_score_ge_3", "gate_flip",
        "fleiss_kappa", "krippendorff_alpha", "note",
    ]
    buckets: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
    for row in rows_raw:
        if row.get("tier") != "judge" or not row.get("ok"):
            continue
        if row.get("judged_summary_source") != JUDGE_PASS_SOURCE:
            continue
        score = (row.get("parsed") or {}).get("score")
        if score is None:
            continue
        buckets[(row["exp_id"], row["run_id"], row["case_id"])].append((row["repeat_idx"], int(score)))
    out = []
    for key in sorted(buckets, key=lambda k: tuple(str(x) for x in k)):
        ordered = [s for _, s in sorted(buckets[key])]
        n_ge = sum(1 for s in ordered if s >= DEFAULT_PASS_THRESHOLD)
        out.append([
            key[0], key[1], key[2], len(ordered), "|".join(str(s) for s in ordered),
            len(set(ordered)), int(len(set(ordered)) == 1),
            sum(ordered) / len(ordered), _sd([float(s) for s in ordered]),
            min(ordered), max(ordered), n_ge,
            int(n_ge not in (0, len(ordered))),
            None, None, NOT_IMPLEMENTED_NOTE,
        ])
    return header, out


def table_judge_rescore_summary(rescore_rows: list[list]) -> tuple[list[str], list[list]]:
    """One line per judge-isolation run, so a stability claim in the write-up
    has a single cell to point at."""
    header = [
        "exp_id", "run_id", "n_cases", "scores_per_case", "share_unanimous",
        "mean_within_case_sd", "max_within_case_range", "share_gate_flip",
        "fleiss_kappa", "krippendorff_alpha", "note",
    ]
    grouped: dict[tuple, list[list]] = defaultdict(list)
    for row in rescore_rows:
        grouped[(row[0], row[1])].append(row)
    out = []
    for key in sorted(grouped, key=lambda k: tuple(str(x) for x in k)):
        group = grouped[key]
        sds = [r[8] for r in group if r[8] is not None]
        counts = sorted({r[3] for r in group})
        out.append([
            key[0], key[1], len(group), "|".join(str(c) for c in counts),
            sum(r[6] for r in group) / len(group),
            (sum(sds) / len(sds)) if sds else None,
            max((r[10] - r[9]) for r in group),
            sum(r[12] for r in group) / len(group),
            None, None, NOT_IMPLEMENTED_NOTE,
        ])
    return header, out


def table_judge_scores(observations: list[dict]) -> tuple[list[str], list[list]]:
    """Judge score distribution -- the evidence for or against scale collapse,
    one of the five known LLM-as-judge failure modes."""
    header = ["exp_id", "run_id", "prompt_version", "score", "n", "share", "n_total", "method"]
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for obs in observations:
        if obs["judge_score"] is None:
            continue
        buckets[(obs["exp_id"], obs["run_id"], obs["prompt_version"])].append(obs["judge_score"])
    rows: list[list] = []
    for key in sorted(buckets, key=lambda k: tuple(str(x) for x in k)):
        scores = buckets[key]
        for score in range(1, 6):
            count = sum(1 for s in scores if s == score)
            rows.append([key[0], key[1], key[2], score, count, count / len(scores), len(scores), "count"])
    return header, rows


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def figure_rates(rate_rows: list[list], out_path: Path, seed: int) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover -- documented fallback
        print(f"!! matplotlib unavailable ({exc}); CSV tables were still written", file=sys.stderr)
        return None

    points = [r for r in rate_rows if r[10] == "wilson" and r[3] != "pooled" and r[4] == "category_match"]
    if not points:
        return None
    labels = [f"{r[1]}\nrepeat {r[3]}" for r in points]
    est = [r[7] for r in points]
    lo = [r[7] - r[8] for r in points]
    hi = [r[9] - r[7] for r in points]
    n_total = points[0][5]

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(points)), 4.2))
    ax.errorbar(range(len(points)), est, yerr=[lo, hi], fmt="o", capsize=4, color="#1f4e79")
    ax.set_xticks(range(len(points)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("category_match rate")
    ax.set_title("Primary metric by run (95% Wilson interval)")
    ax.grid(axis="y", alpha=0.3)
    footnote = _wrap(
        f"n={n_total} cases per point; 95% Wilson score interval; analysis seed={seed}; "
        f"points are independent API runs of the same prompt version.",
        width=90,
    )
    fig.text(0.01, 0.01, footnote, fontsize=7, va="bottom")
    fig.tight_layout(rect=(0, 0.04 + 0.03 * footnote.count("\n"), 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, metadata={"Software": "experiments/analyze.py"})
    plt.close(fig)
    return rel_to_repo(out_path)


def figure_noise_floor(rate_rows: list[list], out_path: Path, seed: int) -> str | None:
    """Repeats of one prompt version, both metrics, with the production gate
    warning and critical bands drawn around the first run -- the run a real
    gate would have stored as its baseline."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover -- documented fallback
        print(f"!! matplotlib unavailable ({exc}); CSV tables were still written", file=sys.stderr)
        return None

    warn, crit = gate_thresholds()
    wilson = [r for r in rate_rows if r[10] == "wilson" and r[3] != "pooled"]
    exp_ids = sorted({r[0] for r in wilson if r[4] == "passed"})
    if not exp_ids:
        return None
    exp_id = exp_ids[0]
    series = {}
    for metric in ("passed", "category_match"):
        pts = sorted((r for r in wilson if r[0] == exp_id and r[4] == metric), key=lambda r: r[3])
        if pts:
            series[metric] = pts
    if "passed" not in series:
        return None
    ref = series["passed"][0][7]
    n_points = len(series["passed"])
    n_total = series["passed"][0][5]

    fig, ax = plt.subplots(figsize=(max(6.5, 1.7 * n_points), 4.6))
    styles = {"passed": ("o", "#1f4e79", "passed (gate metric)"),
              "category_match": ("s", "#7c4a2d", "category_match (primary)")}
    offsets = {"passed": -0.06, "category_match": 0.06}
    for metric, pts in series.items():
        marker, colour, label = styles[metric]
        xs = [i + offsets[metric] for i in range(len(pts))]
        est = [r[7] for r in pts]
        lo = [r[7] - r[8] for r in pts]
        hi = [r[9] - r[7] for r in pts]
        ax.errorbar(xs, est, yerr=[lo, hi], fmt=marker, capsize=4, color=colour, label=label)

    ax.axhline(ref, color="#555555", lw=1.0, label=f"run 1 passed = {ref:.4f}")
    for delta, colour, style, name in ((warn, "#c98a00", "--", "warning"), (crit, "#b00020", "-.", "critical")):
        for sign in (1, -1):
            ax.axhline(ref + sign * delta, color=colour, lw=1.0, ls=style,
                       label=(f"gate {name} +/- {delta:.0%}" if sign == 1 else None))

    lows = [r[8] for pts in series.values() for r in pts] + [ref - crit]
    highs = [r[9] for pts in series.values() for r in pts] + [ref + crit]
    # Extra headroom at the bottom so the legend box never sits on the
    # critical band -- the band is the whole point of this figure.
    ax.set_ylim(max(0.0, min(lows) - 0.07), min(1.02, max(highs) + 0.02))
    ax.set_xticks(range(n_points))
    ax.set_xticklabels([f"run {r[3] + 1}" for r in series["passed"]], fontsize=9)
    ax.set_ylabel("rate")
    ax.set_title(f"Noise floor: {n_points} repeats of the same prompt ({exp_id})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, loc="lower left", ncol=3, framealpha=0.95)
    footnote = _wrap(
        f"n={n_total} cases per point; 95% Wilson score interval; analysis seed={seed}; "
        f"every point is an independent API run of the same prompt version, so all spread is noise. "
        f"Bands are the production gate default thresholds ({warn:.0%} warning, {crit:.0%} critical, "
        f"evalkit/diff.py argparse defaults, SPEC.md section 5) applied to the first run passed rate.",
        width=100,
    )
    fig.text(0.01, 0.01, footnote, fontsize=7, va="bottom")
    fig.tight_layout(rect=(0, 0.05 + 0.035 * footnote.count("\n"), 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, metadata={"Software": "experiments/analyze.py"})
    plt.close(fig)
    return rel_to_repo(out_path)


def figure_judge_scores(score_rows: list[list], out_path: Path, seed: int) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return None
    if not score_rows:
        return None
    totals = {s: 0 for s in range(1, 6)}
    n_total = 0
    for row in score_rows:
        totals[row[3]] += row[4]
        n_total = max(n_total, row[6])
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.bar(list(totals), [totals[s] for s in totals], color="#7c4a2d")
    ax.set_xlabel("judge score (1-5)")
    ax.set_ylabel("calls")
    ax.set_title("Judge score distribution")
    footnote = _wrap(f"n={sum(totals.values())} judge calls; analysis seed={seed}; counts, no interval.", width=60)
    fig.text(0.01, 0.01, footnote, fontsize=7, va="bottom")
    fig.tight_layout(rect=(0, 0.06 + 0.03 * footnote.count("\n"), 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, metadata={"Software": "experiments/analyze.py"})
    plt.close(fig)
    return rel_to_repo(out_path)


# --------------------------------------------------------------------------
# self-check: run the E1 estimators on a pairing whose answer is known
# --------------------------------------------------------------------------
def selfcheck_pair_sets(observations: list[dict]) -> dict[tuple[str, str], dict]:
    """Pair E0's repeat 1 against its repeat 2 -- same prompt, same day, so the
    true difference is zero by construction.

    This exists so the E1 code path is exercised before E1 exists. It is a
    plumbing check, not a result: it cannot show that the estimators
    discriminate, only that they run end to end and return zero when they
    should. `experiments/stats.py` carries the non-null cases.
    """
    out: dict[tuple[str, str], dict] = {}
    for metric in METRICS:
        sides = []
        for repeat in (0, 1):
            sides.append({
                obs["case_id"]: obs[metric]
                for obs in observations
                if obs["exp_id"] == E1_BASELINE_EXP_ID
                and obs["prompt_version"] == BASELINE_PROMPT_VERSION
                and obs["repeat_idx"] == repeat
                and obs[metric] is not None
            })
        base, cand = sides
        shared = sorted(set(base) & set(cand))
        out[(metric, "e0_repeat2_vs_repeat1")] = {
            "pairs": [(cid, base[cid], cand[cid]) for cid in shared],
            "n_tie": 0,
            "n_missing": len(set(base) ^ set(cand)),
            "baseline_label": f"{BASELINE_PROMPT_VERSION} ({E1_BASELINE_EXP_ID} repeat 0)",
        }
    return out


def run_self_check(observations: list[dict], seed: int, out_dir: Path | None, n_sim: int) -> int:
    pair_sets = {k: v for k, v in selfcheck_pair_sets(observations).items() if v["pairs"]}
    if not pair_sets:
        print("self-check: no E0 baseline observations found; nothing to check", file=sys.stderr)
        return 1
    header, rows = table_e1_main(pair_sets)
    p_header, p_rows = table_e1_power(pair_sets, seed, n_sim=n_sim)
    widths = {"metric": 16, "candidate_version": 22}
    print("\ne1_main.csv on the zero-difference pairing (E0 repeat 1 vs repeat 2)")
    show = ["metric", "candidate_version", "n", "a", "b", "c", "d", "p_raw", "p_holm", "p_bh",
            "rd", "ci_low", "ci_high", "or"]
    print("  " + "  ".join(f"{name:>{widths.get(name, 9)}}" for name in show))
    for row in rows:
        cells = []
        for name in show:
            value = row[header.index(name)]
            text = "" if value is None else (f"{value:.6f}" if isinstance(value, float) else str(value))
            cells.append(f"{text:>{widths.get(name, 9)}}")
        print("  " + "  ".join(cells))
    print("\ne1_power.csv on the same pairing")
    p_show = ["metric", "candidate_version", "n", "alpha", "alpha_basis", "power", "observed_b", "observed_c"]
    print("  " + "  ".join(f"{name:>{widths.get(name, 11)}}" for name in p_show))
    for row in p_rows:
        cells = []
        for name in p_show:
            value = row[p_header.index(name)]
            text = "" if value is None else (f"{value:.6f}" if isinstance(value, float) else str(value))
            cells.append(f"{text:>{widths.get(name, 11)}}")
        print("  " + "  ".join(cells))

    failures = []
    for row in rows:
        if row[header.index("b")] != 0 or row[header.index("c")] != 0:
            failures.append(f"{row[0]}: expected no discordant pairs, got b={row[header.index('b')]} "
                            f"c={row[header.index('c')]}")
        if abs(row[header.index("rd")]) > 1e-12:
            failures.append(f"{row[0]}: expected rd = 0, got {row[header.index('rd')]}")
        if not (row[header.index("ci_low")] < 0 < row[header.index("ci_high")]):
            failures.append(f"{row[0]}: interval does not straddle zero")
    for row in p_rows:
        if row[p_header.index("power")] != 0.0:
            failures.append(f"power at n={row[p_header.index('n')]} should be 0 on a null pairing")

    if out_dir is not None:
        write_csv(out_dir / "tables" / "e1_main.csv", header, rows)
        write_csv(out_dir / "tables" / "e1_power.csv", p_header, p_rows)
        for fig in (
            figure_e1_forest(rows, out_dir / "figures" / "e1_forest.png", seed),
            figure_e1_power(p_rows, rows, out_dir / "figures" / "e1_power.png", seed),
        ):
            print(f"self-check figure -> {fig}")
        print(f"self-check artifacts written under {out_dir}")

    if failures:
        print("\nself-check FAILED:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    print("\nself-check PASSED: zero discordant pairs, rd = 0, interval straddles 0, "
          "simulated power 0 at every n. This checks the plumbing, not the discrimination.")
    return 0


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------
def build_manifest(files: list[Path], rows: list[dict], out_dir: Path, seed: int, n_boot: int, tables: list[str], figures: list[str], dataset_path: Path) -> dict:
    by_file: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_file[row["_source_file"]].append(row)

    entries = []
    raw_cost = 0.0
    raw_cost_recorded = 0.0
    for path in files:
        rel = rel_to_repo(path)
        group = by_file.get(rel, [])
        cost = sum(row_cost(r) for r in group)
        recorded = sum(r.get("cost_usd", 0.0) for r in group)
        raw_cost += cost
        raw_cost_recorded += recorded
        stamps = sorted(r["timestamp_utc"] for r in group if r.get("timestamp_utc"))
        models: dict[str, list[str]] = defaultdict(list)
        for r in group:
            models[r["request_model"]].append(str(r.get("response_model")))
        entries.append({
            "path": rel,
            "sha256": sha256_file(path),
            "lines": len(group),
            "ok": sum(1 for r in group if r.get("ok")),
            "failed": sum(1 for r in group if not r.get("ok")),
            "cache_hits": sum(1 for r in group if r.get("cache_hit")),
            "calls_by_tier": {t: sum(1 for r in group if r["tier"] == t) for t in sorted({r["tier"] for r in group})},
            "request_to_response_model": {k: sorted(set(v)) for k, v in sorted(models.items())},
            "cost_usd": round(cost, 8),
            "cost_usd_recorded_by_runner": round(recorded, 8),
            "cost_usd_unrecorded_at_run_time": round(cost - recorded, 8),
            "first_timestamp_utc": stamps[0] if stamps else None,
            "last_timestamp_utc": stamps[-1] if stamps else None,
        })

    probe_dir = REPO_ROOT / "experiments" / "results" / "probes"
    probes = []
    probe_cost = 0.0
    for path in sorted(probe_dir.glob("*.json")) if probe_dir.exists() else []:
        data = json.loads(path.read_text(encoding="utf-8"))
        probe_cost += float(data.get("total_cost_usd", 0.0))
        probes.append({
            "path": rel_to_repo(path),
            "sha256": sha256_file(path),
            "anthropic_version": data.get("sdk", {}).get("anthropic_version"),
            "cost_usd": data.get("total_cost_usd"),
        })

    ledger_path = REPO_ROOT / "experiments" / "results" / "cost_ledger.json"
    ledger_cumulative = None
    if ledger_path.exists():
        ledger_cumulative = json.loads(ledger_path.read_text(encoding="utf-8")).get("cumulative_usd")

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "analysis": {"seed": seed, "n_boot": n_boot, "pass_threshold": DEFAULT_PASS_THRESHOLD, "python": sys.version.split()[0]},
        "inputs": {
            "dataset": {"path": rel_to_repo(dataset_path), "sha256": sha256_file(dataset_path)},
            "prompts": {
                rel_to_repo(p): sha256_file(p)
                for p in sorted((REPO_ROOT / "prompts").glob("*.yaml"))
            },
        },
        "raw_files": entries,
        "probe_files": probes,
        "totals": {
            "raw_lines": sum(e["lines"] for e in entries),
            "raw_cost_usd": round(raw_cost, 8),
            "raw_cost_usd_recorded_by_runner": round(raw_cost_recorded, 8),
            "raw_cost_usd_unrecorded_at_run_time": round(raw_cost - raw_cost_recorded, 8),
            "probes_cost_usd": round(probe_cost, 8),
            "total_cost_usd": round(raw_cost + probe_cost, 8),
            "cost_ledger_cumulative_usd": ledger_cumulative,
            "cost_ledger_note": "the ledger records runner runs only; probe scripts are accounted separately",
            "cost_method_note": (
                "raw_cost_usd is recomputed from each row's usage at the pinned prices in "
                "experiments/__init__.py; raw_cost_usd_recorded_by_runner is what the runner "
                "wrote into the raw files at run time. The gap is billed-but-unrecorded spend "
                "on calls that returned tokens and then failed validation "
                "(experiments/COST_CALIBRATION.md section 3)"
            ),
        },
        "outputs": {"tables": tables, "figures": figures},
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default=str(REPO_ROOT / "experiments" / "results" / "raw"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "experiments" / "results"))
    p.add_argument("--dataset", default=str(REPO_ROOT / "golden_dataset.json"))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--exp-ids", default=None, help="comma-separated subset of experiment ids")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--power-n-sim", type=int, default=E1_POWER_N_SIM, help="resamples per point in e1_power.csv")
    p.add_argument("--self-check", action="store_true",
                   help="run the E1 estimators on E0's repeat 1 vs repeat 2 (true difference zero) and exit")
    p.add_argument("--self-check-out", default=None,
                   help="with --self-check, also write the tables and figures here (not into results/)")
    args = p.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    dataset_path = Path(args.dataset)
    exp_ids = [s.strip() for s in args.exp_ids.split(",")] if args.exp_ids else None

    rows, files = load_raw(raw_dir, exp_ids)
    if not rows:
        raise SystemExit(f"no raw JSONL rows under {raw_dir} (run experiments/runner.py first)")
    observations = build_observations(rows, dataset_path)

    if args.self_check:
        return run_self_check(
            observations, args.seed,
            Path(args.self_check_out) if args.self_check_out else None,
            args.power_n_sim,
        )

    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"

    pair_sets = e1_pair_sets(observations)
    if not pair_sets:
        print(f"e1: {E1_EMPTY_NOTE}")
    e1_main_spec = table_e1_main(pair_sets)
    e1_power_spec = table_e1_power(pair_sets, args.seed, n_sim=args.power_n_sim)

    rescore_header, rescore_rows = table_judge_rescore(rows)
    specs = [
        ("per_case_outcomes.csv", table_per_case(observations)),
        ("rates_by_run.csv", table_rates(observations)),
        ("rates_run_spread.csv", table_run_spread(observations)),
        ("per_case_instability.csv", table_case_instability(observations)),
        ("rates_by_stratum.csv", table_strata(observations)),
        ("rates_bootstrap.csv", table_bootstrap(observations, args.seed, args.n_boot)),
        ("paired_mcnemar.csv", table_mcnemar(observations)),
        ("e1_main.csv", e1_main_spec),
        ("e1_power.csv", e1_power_spec),
        ("tokens_latency_by_tier.csv", table_tokens(rows)),
        ("judge_score_distribution.csv", table_judge_scores(observations)),
        ("judge_rescore_stability.csv", (rescore_header, rescore_rows)),
        ("judge_rescore_summary.csv", table_judge_rescore_summary(rescore_rows)),
    ]
    written: list[str] = []
    for name, (header, body) in specs:
        write_csv(tables_dir / name, header, body)
        written.append(rel_to_repo(tables_dir / name))
        print(f"{name:32s} {len(body):6d} rows")

    figures: list[str] = []
    if not args.no_figures:
        by_name = dict(specs)
        rate_rows = by_name["rates_by_run.csv"][1]
        score_rows = by_name["judge_score_distribution.csv"][1]
        for fig in (
            figure_rates(rate_rows, figures_dir / "primary_metric_by_run.png", args.seed),
            figure_noise_floor(rate_rows, figures_dir / "noise_floor_vs_gate_thresholds.png", args.seed),
            figure_judge_scores(score_rows, figures_dir / "judge_score_distribution.png", args.seed),
            figure_e1_forest(e1_main_spec[1], figures_dir / "e1_forest.png", args.seed),
            figure_e1_power(e1_power_spec[1], e1_main_spec[1], figures_dir / "e1_power.png", args.seed),
        ):
            if fig:
                figures.append(fig)
                print(f"figure -> {fig}")

    manifest = build_manifest(files, rows, out_dir, args.seed, args.n_boot, written, figures, dataset_path)
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"\nMANIFEST -> {rel_to_repo(out_dir / 'MANIFEST.json')}"
        f"\n  raw lines {manifest['totals']['raw_lines']}"
        f"  raw ${manifest['totals']['raw_cost_usd']:.6f}"
        f"  probes ${manifest['totals']['probes_cost_usd']:.6f}"
        f"  total ${manifest['totals']['total_cost_usd']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
