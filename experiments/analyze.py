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

The logistic model with case-clustered standard errors arrived on 2026-09-16
with E5 (`table_e5_logit`). It is the one estimator here that is not standard
library: statsmodels fits it, and when statsmodels is absent the table is
written without coefficients rather than with approximated ones. Its
pre-specified baseline-arm form is not estimable at all -- two levels have no
failures -- and is reported as not fitted, with the levels named.

What is still not implemented, and is therefore absent from the output rather
than approximated: Fleiss kappa, Krippendorff alpha, and Firth's penalised
likelihood for the separated model. See the table in experiments/README.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
    cohen_kappa,
    exact_binomial_test,
    fisher_exact_2x2,
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
E1_FAMILY_CONFIRMATORY = "confirmatory_holm"
E1_FAMILY_SECONDARY = "exploratory_bh"
E1_FAMILY_LEAK_SENSITIVITY = "sensitivity_drop_leaked_case"
# PREREGISTRATION section 10.10.1: few-shot example 4 of prompts/v1.yaml is this
# case verbatim, label included, so v2a and v2b remove a leaked evaluation label
# at the same time as they remove examples. H1 and H2 are recomputed without it.
E1_LEAKED_CASE_ID = "case-043"
E1_LEAK_SENSITIVITY_VERSIONS = ("v2a", "v2b")
E1_POWER_SIZES = (30, 50, 70)
E1_POWER_N_SIM = 2000
# alpha = 0.05 is the nominal per-test level; 0.0125 is what Holm charges the
# first-rejected test in a family of four, i.e. the worst case for this design.
E1_POWER_ALPHAS = ((0.05, "nominal"), (0.0125, "holm_worst_case"))
E1_EMPTY_NOTE = "no E1 raw data under experiments/results/raw/ yet; header written, no rows"

# ---- E5, stratified re-analysis of the E0/E1 raw data (no new calls) ------
# Everything in E5 is exploratory (PREREGISTRATION section 4): the dataset was
# built to a stratum quota, not sampled, so a stratum rate describes these 70
# emails and nothing else. The arms are named rather than globbed, because the
# raw directory also holds a 10-case smoke run tagged v1 and a judge-isolation
# arm with no classifier call -- neither belongs in a coverage table.
E5_BASELINE_ARM = ("e0_noise", "v1")
E5_DEGRADED_ARMS = (("e1_v2a", "v2a"), ("e1_v2b", "v2b"), ("e1_v2c", "v2c"), ("e1_v2d", "v2d"))
E5_METRIC = E1_PRIMARY_METRIC
# (column in an observation) for each stratum kind
E5_STRATUM_KINDS = (("language", "language"), ("difficulty", "difficulty"), ("category", "expected_category"))
E5_FAMILY_BASELINE = "exploratory_bh_strata_baseline"
E5_FAMILY_DEGRADED = "exploratory_bh_strata_degraded"
E5_FAMILY_DESCRIPTIVE = "descriptive"
E5_UNIT_NOTE = (
    "one outcome per case, repeats collapsed by majority vote "
    "(docs/PREREGISTRATION.md section 10.4), so n is a count of emails, not of calls"
)
E5_FISHER_NOTE = (
    "Fisher exact, this stratum against every other case in the same arm; "
    "two strata are n = 7 and n = 8, where chi-square does not apply"
)
# pass ~ language + difficulty + category, one row per (case, repeat), standard
# errors clustered on case. Reference levels are the largest stratum of each
# kind, so every coefficient reads against the majority case.
E5_LOGIT_FORMULA = "category_match ~ language + difficulty + category"
E5_LOGIT_SE_TYPE = "cluster_robust_by_case (statsmodels cov_type=cluster, HC1-style within-cluster sum)"
# ---- E2, pairwise judge and position bias --------------------------------
# Named rather than globbed for the same reason as E5: the raw directory also
# holds `e2_smoke`, a 2-case 16-call calibration run made before the real one,
# which is committed evidence but is not part of any E2 number.
E2_EXP_ID = "e2_pairwise"
E2_TIER = "judge_pairwise"
E2_LAYERS = ("easy", "hard")
E2_ORDERS = ("left_first", "right_first")
E2_FAMILY_CONSISTENCY = "exploratory_bh_e2_consistency"
E2_FAMILY_POSITION = "exploratory_bh_e2_position"
E2_FAMILY_PAIRED_LAYER = "sensitivity_paired_layers"
E2_TIE_NOTE = (
    "a tie is neither a first-position win nor a second-position win, so it is out of the binomial denominator and reported as its own rate"
)
E2_DEPENDENCE_NOTE = (
    "the exact binomial treats calls as independent; each case contributes several calls, so the case-cluster bootstrap interval in the same row is the honest width"
)
# E2 ran in two batches: 2026-09-15 wrote all 560 cells and the API answered 94
# of them before the account hit its spend limit; 2026-09-16 re-sent the 466
# that had failed. Both raw files stay in the repository, so a cell can appear
# twice -- once refused, once answered -- and `e2_cell_rows` resolves that.
E2_DEDUP_NOTE = (
    "one row per design cell (judge model, layer, order, pair) across both batches; where a cell has "
    "a failed row and a successful row, the successful row is the observation "
    "(docs/PREREGISTRATION.md section 9, 2026-09-16)"
)

# ---- E4, rater agreement -------------------------------------------------
# Three raters label the same 70 emails from the same four definitions: the
# gold labels in golden_dataset.json (one human, July 2026), a second human who
# is not on the project (experiments/data/annotator2_labels.json, due
# 2026-09-23), and the models of the E4 annotation arm. Everything here is an
# agreement estimate with an interval, not a test: there is no null hypothesis
# in this table and no p-value in it.
E4_EXP_ID = "e4_annot"
E4_TIER = "annotator"
E4_CATEGORIES = ("billing", "technical", "account", "general")
E4_GOLD_RATER = "human-1 (gold)"
E4_HUMAN2_RATER = "human-2 (non-member)"
E4_HUMAN2_PATH = REPO_ROOT / "experiments" / "data" / "annotator2_labels.json"
E4_FAMILY = "descriptive"
E4_KAPPA_HEADER = [
    "family", "rater_a", "rater_b", "n", "n_excluded", "po", "pe", "kappa",
    "boot_ci_low", "boot_ci_high", "n_boot", "seed", "landis_koch", "method", "note",
]
E4_CONFUSION_HEADER = [
    "row_rater", "column_rater", "row_label", *E4_CATEGORIES, "n_row", "note",
]
E4_KAPPA_NOTE = (
    "Cohen (1960) kappa on one label per case; interval is a percentile bootstrap over cases "
    "(one case = one unit), so it reflects sampling of these 70 emails and not annotator variance"
)
E4_BAND_NOTE = (
    "band names are Landis & Koch (1977) Table 1, a convention for reading the number, not a test"
)
E4_EMPTY_NOTE = (
    "no e4_annot raw data and no annotator2_labels.json yet; header written, no rows"
)

E5_FIRTH_NOTE = (
    "Firth penalised likelihood not run: no validated implementation is available here "
    "(statsmodels 0.15.0 has none and firthlogist is not installed), and this package does "
    "not ship an unvalidated one. Recorded as not done rather than approximated."
)


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

    Only the `classifier` and `judge` tiers are joined here. E2's
    `judge_pairwise` rows carry no category and no 1-5 score, so folding them
    into this table would invent one observation per pairwise call with every
    metric None -- invisible in the rate tables, but a new `prompt_version` in
    `paired_mcnemar.csv`. They are read straight from the raw rows by the E2
    tables instead.
    """
    cases = {c.id: c for c in load_all_cases(dataset_path)}
    grouped: dict[tuple, dict] = defaultdict(dict)
    for row in rows:
        if row["tier"] not in ("classifier", "judge"):
            continue
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
            E1_FAMILY_CONFIRMATORY if primary else E1_FAMILY_SECONDARY,
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
    rows.extend(table_e1_leak_sensitivity(pair_sets))
    return E1_MAIN_HEADER, rows


def table_e1_leak_sensitivity(pair_sets: dict[tuple[str, str], dict]) -> list[list]:
    """Section 10.10.1: H1 and H2 recomputed with the leaked case dropped.

    These rows sit in `e1_main.csv` next to the primary rows, as the
    pre-registration requires, and are told apart by the `family` column. They
    carry no adjusted p-value: Holm is fixed across the four confirmatory
    tests and nothing else, so adding a fifth and sixth number to that family
    would change the confirmatory result, which is the one thing a sensitivity
    analysis must not do. `p_raw` is the value to read in these rows.
    """
    out: list[list] = []
    for version in E1_LEAK_SENSITIVITY_VERSIONS:
        info = pair_sets.get((E1_PRIMARY_METRIC, version))
        if not info or not info["pairs"]:
            continue
        kept = [pair for pair in info["pairs"] if pair[0] != E1_LEAKED_CASE_ID]
        dropped = len(info["pairs"]) - len(kept)
        if not kept:
            continue
        cells = paired_cells(kept)
        a, b, c, d, n = cells["a"], cells["b"], cells["c"], cells["d"], cells["n"]
        mc = mcnemar_exact(b, c)
        ci = newcombe_paired_diff_ci(a, b, c, d)
        note = (
            NEWCOMBE_VALIDATION
            + "; PREREGISTRATION section 10.10.1 sensitivity analysis, "
            + E1_LEAKED_CASE_ID
            + (" dropped from the pairing" if dropped
               else " was already absent from the pairing, so this row repeats the confirmatory one")
            + "; no multiplicity adjustment, read p_raw"
        )
        if b == 0:
            note += "; odds ratio undefined (b = 0)"
        out.append([
            E1_PRIMARY_METRIC, info["baseline_label"], version, n, info["n_tie"], info["n_missing"],
            a + b, a + c, a, b, c, d,
            mc["p_value"], None, None, ci["rd"], ci["ci_low"], ci["ci_high"],
            (c / b) if b > 0 else None,
            E1_FAMILY_LEAK_SENSITIVITY, mc["method"], ci["method"], note,
        ])
    return out


E1_POWER_HEADER = [
    "metric", "candidate_version", "n", "alpha", "alpha_basis", "power", "mc_se", "n_sim", "seed",
    "n_observed_pairs", "observed_b", "observed_c", "mean_b", "mean_c", "method", "note",
]


def monte_carlo_se(power: float, n_sim: int) -> float:
    """Standard error of a simulated rejection rate: sqrt(p(1-p)/n_sim).

    Every `power` value in `e1_power.csv` is a proportion of `n_sim` draws, so
    it carries simulation error of its own. At n_sim = 2000 the worst case is
    0.011, which is why the power numbers in the write-up are read to two
    decimal places and not three. Raising n_sim would shrink this; it is not
    an interval on the effect, only on the simulation.
    """
    if n_sim <= 0:
        return 0.0
    return math.sqrt(max(power * (1.0 - power), 0.0) / n_sim)


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
                monte_carlo_se(res["power"], res["n_sim"]),
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
    primary = [r for r in main_rows
               if r[col["metric"]] == E1_PRIMARY_METRIC and r[col["family"]] == E1_FAMILY_CONFIRMATORY]
    secondary = [r for r in main_rows if r[col["family"]] == E1_FAMILY_SECONDARY]
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
                    if r[mcol["candidate_version"]] == version
                    and r[mcol["metric"]] == E1_PRIMARY_METRIC
                    and r[mcol["family"]] == E1_FAMILY_CONFIRMATORY]
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
    n_pairs = nominal[0][pcol["n_observed_pairs"]]
    worst_se = monte_carlo_se(0.5, n_sim)
    footnote = _wrap(
        f"Open markers: {n_sim} resamples of the n={n_pairs} observed paired cases at each n, exact "
        f"McNemar at alpha=0.05, analysis seed={seed} -- conditional on the effect this study observed, "
        f"not a design power calculation. No interval applies to a power curve: each point is a "
        f"simulated rejection rate, and its Monte-Carlo standard error (at most {worst_se:.3f} at "
        f"n_sim={n_sim}) is the mc_se column of e1_power.csv. Filled diamonds: the single test that "
        f"actually ran at n=70, plotted as 1 if it rejected under Holm and 0 if it did not; one "
        f"realized decision is not a rate. Holm's worst-case level (alpha=0.0125) is in e1_power.csv.",
        width=104,
    )
    fig.text(0.01, 0.01, footnote, fontsize=7, va="bottom")
    fig.tight_layout(rect=(0, 0.08 + 0.035 * footnote.count("\n"), 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, metadata={"Software": "experiments/analyze.py"})
    plt.close(fig)
    return rel_to_repo(out_path)


# --------------------------------------------------------------------------
# E5: stratified coverage (exploratory, no new API calls)
# --------------------------------------------------------------------------
E5_STRATA_HEADER = [
    "family", "exp_id", "prompt_version", "metric", "unit", "stratum_kind", "stratum",
    "n", "k", "estimate", "wilson_lo", "wilson_hi", "ci_width",
    "n_rest", "k_rest", "estimate_rest", "difference", "fisher_p", "p_bh", "n_in_bh_family",
    "n_excluded_tie", "ci_method", "test", "note",
]

E5_LOGIT_HEADER = [
    "model_id", "family", "data", "formula", "n_obs", "n_clusters", "fitted",
    "term", "reference_level", "coefficient", "std_error", "se_type", "z", "p_value",
    "odds_ratio", "or_ci_low", "or_ci_high", "converged", "method", "note",
]


def _case_number(case_id: str) -> int | None:
    """case-064 -> 64. None for an id that does not follow that shape."""
    tail = case_id.rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() else None


def _membership_note(case_ids: Sequence[str]) -> str:
    """Whether a stratum is a consecutive block of case ids.

    `mixed` is: case-064 to case-070 were written last, in one sitting, to fill
    the 10% code-switched quota. That is a property of how the dataset was
    built and it belongs next to the stratum's interval, so it is derived from
    the ids at analysis time instead of being typed into a document that can go
    stale if the dataset is ever rebuilt.
    """
    ids = sorted(case_ids)
    numbers = [n for n in (_case_number(cid) for cid in ids) if n is not None]
    if len(numbers) == len(ids) and len(numbers) > 1 and numbers == list(range(numbers[0], numbers[0] + len(numbers))):
        return (
            f"members are the consecutive block {ids[0]}..{ids[-1]}, written as one "
            "quota fill rather than drawn across the dataset"
        )
    return f"members: {', '.join(ids)}" if len(ids) <= 8 else f"{len(ids)} cases, not a consecutive block"


def e5_level_order(observations: list[dict], column: str) -> list[str]:
    """Stratum levels, largest first, ties broken by name -- fixed here so two
    analysis runs cannot order a table differently."""
    counts: dict[str, set] = defaultdict(set)
    for obs in observations:
        counts[obs[column]].add(obs["case_id"])
    return sorted(counts, key=lambda level: (-len(counts[level]), level))


def e5_arms(observations: list[dict]) -> list[tuple[str, str]]:
    """The (exp_id, prompt_version) arms that actually have raw data."""
    present = {(obs["exp_id"], obs["prompt_version"]) for obs in observations}
    return [arm for arm in (E5_BASELINE_ARM, *E5_DEGRADED_ARMS) if arm in present]


def table_e5_strata(observations: list[dict]) -> tuple[list[str], list[list]]:
    """Per-stratum rate with a Wilson interval, plus Fisher exact against the
    rest of the same arm.

    The interval is the result here, not the p-value. A stratum of 7 cases
    cannot separate 65% from 100%, and the point of the table is to show how
    wide those intervals are next to the headline 64/70 -- so `ci_width` is a
    column rather than something a reader has to subtract.
    """
    rows: list[list] = []
    arms = e5_arms(observations)
    if not arms:
        return E5_STRATA_HEADER, rows
    by_case = {obs["case_id"]: obs for obs in observations}
    level_order = {kind: e5_level_order(observations, column) for kind, column in E5_STRATUM_KINDS}

    pending: dict[str, list[int]] = defaultdict(list)  # family -> row indices
    for exp_id, version in arms:
        outcomes, n_tie = _case_outcomes(observations, E5_METRIC, version, exp_id=exp_id)
        if not outcomes:
            continue
        family = E5_FAMILY_BASELINE if (exp_id, version) == E5_BASELINE_ARM else E5_FAMILY_DEGRADED
        n_all, k_all = len(outcomes), sum(outcomes.values())
        lo, hi = wilson_ci(k_all, n_all)
        rows.append([
            E5_FAMILY_DESCRIPTIVE, exp_id, version, E5_METRIC, "case", "overall", "all",
            n_all, k_all, k_all / n_all, lo, hi, hi - lo,
            None, None, None, None, None, None, None,
            n_tie, "wilson", "", E5_UNIT_NOTE,
        ])
        for kind, column in E5_STRATUM_KINDS:
            for level in level_order[kind]:
                members = sorted(cid for cid in outcomes if by_case[cid][column] == level)
                if not members:
                    continue
                member_set = set(members)
                values = [outcomes[cid] for cid in members]
                n, k = len(values), sum(values)
                lo, hi = wilson_ci(k, n)
                rest = [v for cid, v in outcomes.items() if cid not in member_set]
                n_rest, k_rest = len(rest), sum(rest)
                fisher = fisher_exact_2x2(k, n - k, k_rest, n_rest - k_rest)
                pending[family].append(len(rows))
                rows.append([
                    family, exp_id, version, E5_METRIC, "case", kind, level,
                    n, k, k / n, lo, hi, hi - lo,
                    n_rest, k_rest, k_rest / n_rest if n_rest else None,
                    fisher["difference"], fisher["p_value"], None, None,
                    n_tie, "wilson", fisher["method"], _membership_note(members),
                ])

    col = {name: i for i, name in enumerate(E5_STRATA_HEADER)}
    for family, idxs in pending.items():
        adjusted = bh_adjust([rows[i][col["fisher_p"]] for i in idxs])
        for i, value in zip(idxs, adjusted):
            rows[i][col["p_bh"]] = value
            rows[i][col["n_in_bh_family"]] = len(idxs)
    return E5_STRATA_HEADER, rows


def _e5_design(observations: list[dict], arms: Sequence[tuple[str, str]], with_version: bool):
    """Design matrix for the E5 logit, built by hand.

    No pandas and no patsy: the dummy coding is four lines and doing it here
    keeps the one place a reference level could silently change visible in this
    file. Returns (y, X, column names, reference levels, groups, rows used).
    """
    keep = set(arms)
    data = [
        obs for obs in observations
        if (obs["exp_id"], obs["prompt_version"]) in keep and obs[E5_METRIC] is not None
    ]
    data.sort(key=lambda o: (o["exp_id"], o["run_id"], o["repeat_idx"], o["case_id"]))
    if not data:
        return None
    levels = {kind: e5_level_order(data, column) for kind, column in E5_STRATUM_KINDS}
    columns: list[tuple[str, str, str]] = []  # (term, column, level)
    for kind, column in E5_STRATUM_KINDS:
        for level in levels[kind][1:]:  # first (largest) level is the reference
            columns.append((f"{kind}[{level}]", column, level))
    if with_version:
        versions = sorted({obs["prompt_version"] for obs in data})
        for version in versions[1:]:
            columns.append((f"prompt_version[{version}]", "prompt_version", version))
    y = [obs[E5_METRIC] for obs in data]
    X = [[1.0] + [1.0 if obs[column] == level else 0.0 for _, column, level in columns] for obs in data]
    names = ["const"] + [term for term, _, _ in columns]
    references = {kind: levels[kind][0] for kind, _ in E5_STRATUM_KINDS}
    if with_version:
        references["prompt_version"] = sorted({obs["prompt_version"] for obs in data})[0]
    groups = [obs["case_id"] for obs in data]
    return y, X, names, references, groups, data


def _e5_constant_levels(data: list[dict]) -> list[str]:
    """Predictor levels whose outcome never varies.

    Quasi-complete separation: the maximum-likelihood coefficient for such a
    level is unbounded, so the fit returns a large number with a large standard
    error and means nothing. Firth's penalised likelihood is the standard
    remedy and is not implemented here, so the model is reported as not fitted
    and these levels are named.
    """
    constant: list[str] = []
    for kind, column in E5_STRATUM_KINDS:
        cells: dict[str, list[int]] = defaultdict(list)
        for obs in data:
            cells[obs[column]].append(obs[E5_METRIC])
        for level in sorted(cells):
            values = cells[level]
            if values and (sum(values) == 0 or sum(values) == len(values)):
                constant.append(f"{kind}[{level}]={sum(values)}/{len(values)}")
    return constant


def table_e5_logit(observations: list[dict]) -> tuple[list[str], list[list]]:
    """Logistic regression of the primary metric on the three stratum kinds,
    standard errors clustered on case.

    Three rows of model, one row of absence:

      m1  the pre-specified formula on every arm that ran (E0 + E1), which is
          the only version of it that is estimable -- see m2;
      m2  the same formula on the baseline arm alone, which is not fitted
          whenever a level in that arm has no failures at all; the levels and
          their counts go in the row's note;
      m3  m1 plus a prompt-version term, because m1 pools five prompts and the
          four degraded ones are degraded on purpose. Sensitivity, not the
          pre-specified formula.
      m4  Firth, not run.
    """
    rows: list[list] = []
    arms = e5_arms(observations)
    if not arms:
        return E5_LOGIT_HEADER, rows
    baseline = [arm for arm in arms if arm == E5_BASELINE_ARM]
    specs = [
        ("m1_pooled_arms", "primary", arms, False),
        ("m2_baseline_arm_only", "sensitivity", baseline, False),
        ("m3_pooled_plus_version_term", "sensitivity", arms, True),
    ]
    try:
        import numpy as np
        import statsmodels.api as sm
    except ImportError as exc:  # pragma: no cover -- documented fallback
        print(f"!! statsmodels/numpy unavailable ({exc}); e5_logit.csv written without coefficients", file=sys.stderr)
        for model_id, family, subset, with_version in specs:
            rows.append([
                model_id, family, "+".join(f"{e}:{v}" for e, v in subset), E5_LOGIT_FORMULA,
                None, None, "no", "(model not fitted)", "", None, None, E5_LOGIT_SE_TYPE,
                None, None, None, None, None, "", "logit_cluster_robust",
                "not fitted: numpy/statsmodels do not import in the interpreter that produced this "
                "file (see experiments/README.md, Environment). No coefficients are approximated.",
            ])
        rows.append(_e5_firth_row())
        return E5_LOGIT_HEADER, rows

    for model_id, family, subset, with_version in specs:
        built = _e5_design(observations, subset, with_version) if subset else None
        if built is None:
            continue
        y, X, names, references, groups, data = built
        label = "+".join(f"{e}:{v}" for e, v in subset)
        formula = E5_LOGIT_FORMULA + (" + prompt_version" if with_version else "")
        constant = _e5_constant_levels(data)
        n_obs, n_clusters = len(y), len(set(groups))
        if constant:
            rows.append([
                model_id, family, label, formula, n_obs, n_clusters, "no",
                "(model not fitted)", "", None, None, E5_LOGIT_SE_TYPE, None, None, None, None, None,
                "", "logit_cluster_robust",
                "not fitted: separation. These levels have a constant outcome, so their maximum-"
                f"likelihood coefficients are unbounded: {'; '.join(constant)}. " + E5_FIRTH_NOTE,
            ])
            continue
        arr_y = np.asarray(y, dtype=float)
        arr_X = np.asarray(X, dtype=float)
        codes = {cid: i for i, cid in enumerate(sorted(set(groups)))}
        arr_g = np.asarray([codes[g] for g in groups])
        try:
            fit = sm.Logit(arr_y, arr_X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": arr_g})
        except Exception as exc:  # noqa: BLE001 -- a failed fit is a row, not a crash
            rows.append([
                model_id, family, label, formula, n_obs, n_clusters, "no",
                "(model not fitted)", "", None, None, E5_LOGIT_SE_TYPE, None, None, None, None, None,
                "", "logit_cluster_robust",
                f"not fitted: {type(exc).__name__}: {str(exc)[:200]}",
            ])
            continue
        conf = fit.conf_int()
        for i, term in enumerate(names):
            kind = term.split("[", 1)[0]
            rows.append([
                model_id, family, label, formula, n_obs, n_clusters, "yes",
                term, references.get(kind, ""), float(fit.params[i]), float(fit.bse[i]), E5_LOGIT_SE_TYPE,
                float(fit.tvalues[i]), float(fit.pvalues[i]),
                float(math.exp(fit.params[i])), float(math.exp(conf[i][0])), float(math.exp(conf[i][1])),
                "yes" if bool(fit.mle_retvals.get("converged", False)) else "no", "logit_cluster_robust",
                "exploratory; p-values here are not BH-adjusted and are not part of any family",
            ])
    rows.append(_e5_firth_row())
    return E5_LOGIT_HEADER, rows


def _e5_firth_row() -> list:
    return [
        "m4_firth_sparse_cell_sensitivity", "sensitivity", "", "firth-penalised " + E5_LOGIT_FORMULA,
        None, None, "no", "(not run)", "", None, None, "", None, None, None, None, None,
        "", "firth_penalised_likelihood", E5_FIRTH_NOTE,
    ]


def figure_e5_forest(strata_rows: list[list], out_path: Path, seed: int) -> str | None:
    """The baseline arm's stratum rates with their Wilson intervals.

    Only the baseline arm is drawn. The width of each bar is the argument: at
    n = 7 the interval spans a third of the scale, so "mixed-language cases
    pass 100%" and "mixed-language cases pass two thirds of the time" are the
    same measurement here.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover -- documented fallback
        print(f"!! matplotlib unavailable ({exc}); CSV tables were still written", file=sys.stderr)
        return None
    col = {name: i for i, name in enumerate(E5_STRATA_HEADER)}
    exp_id, version = E5_BASELINE_ARM
    rows = [r for r in strata_rows if r[col["exp_id"]] == exp_id and r[col["prompt_version"]] == version]
    if not rows:
        return None
    overall = next((r for r in rows if r[col["stratum_kind"]] == "overall"), None)
    ordered: list[list] = []
    for kind, _ in E5_STRATUM_KINDS:
        ordered.extend([r for r in rows if r[col["stratum_kind"]] == kind])
    if not ordered:
        return None
    labels = [f"{r[col['stratum_kind']]}: {r[col['stratum']]}" for r in ordered]
    ypos = list(range(len(ordered)))[::-1]
    widest = max(ordered, key=lambda r: r[col["ci_width"]])
    footnote = _wrap(
        f"Baseline arm {version} ({exp_id}), primary metric {E5_METRIC}; {E5_UNIT_NOTE}. "
        f"95% Wilson intervals; {E5_FISHER_NOTE}, BH-adjusted within the "
        f"{len(ordered)} rows shown (family {E5_FAMILY_BASELINE}); all exploratory. "
        f"Widest interval: {widest[col['stratum_kind']]} = {widest[col['stratum']]}, "
        f"n = {widest[col['n']]}, width {widest[col['ci_width']]:.3f}. Analysis seed = {seed}.",
        width=104,
    )
    footnote_lines = footnote.count("\n") + 1

    fig, ax = plt.subplots(figsize=(7.6, 0.44 * len(ordered) + 2.2 + 0.16 * footnote_lines))
    if overall is not None:
        ax.axvline(overall[col["estimate"]], color="#1f4e79", lw=1.0, ls="--",
                   label=f"whole arm {overall[col['k']]}/{overall[col['n']]} = {overall[col['estimate']]:.3f}")
    kinds = [r[col["stratum_kind"]] for r in ordered]
    palette = {"language": "#1f4e79", "difficulty": "#7c4a2d", "category": "#4a6b3a"}
    for y, r in zip(ypos, ordered):
        colour = palette.get(r[col["stratum_kind"]], "#333333")
        est = r[col["estimate"]]
        ax.errorbar([est], [y], xerr=[[est - r[col["wilson_lo"]]], [r[col["wilson_hi"]] - est]],
                    fmt="o", color=colour, capsize=4)
        p_bh = r[col["p_bh"]]
        tail = f"  p_BH={p_bh:.3f}" if isinstance(p_bh, float) else ""
        ax.annotate(f"{r[col['k']]}/{r[col['n']]}   width {r[col['ci_width']]:.2f}{tail}",
                    (1.02, y), xycoords=("axes fraction", "data"), fontsize=7.5,
                    color="#333333", va="center")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.6, len(ordered) - 0.4)
    ax.set_xlabel(f"{E5_METRIC} rate (95% Wilson interval)")
    ax.set_title("E5: how much of the dataset each stratum interval actually pins down")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.95)
    fig.text(0.01, 0.01, footnote, fontsize=7, va="bottom")
    fig.subplots_adjust(bottom=(0.55 + 0.14 * footnote_lines) / fig.get_figheight(),
                        left=0.19, right=0.70, top=0.92)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, metadata={"Software": "experiments/analyze.py"})
    plt.close(fig)
    return rel_to_repo(out_path)


# --------------------------------------------------------------------------
# E2: pairwise judge, order swapped (exploratory)
# --------------------------------------------------------------------------
E2_CONSISTENCY_HEADER = [
    "row_type", "family", "judge_model", "layer", "comparison",
    "n", "k", "estimate", "wilson_lo", "wilson_hi", "ci_width",
    "b", "c", "p_value", "p_bh", "test", "n_excluded", "note",
]

E2_POSITION_HEADER = [
    "row_type", "family", "judge_model", "layer", "subset",
    "n_calls", "n_ok", "n_tie", "tie_rate", "n_decisive", "k_first_position",
    "estimate", "wilson_lo", "wilson_hi", "p_binomial", "p_bh",
    "cluster_ci_low", "cluster_ci_high", "n_clusters", "n_boot", "seed", "test", "note",
]

E2_ALL = "(both)"


def e2_cell_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    """One raw row per E2 design cell, taken across every batch on disk.

    The experiment was run twice (E2_DEDUP_NOTE): the second run re-sent only
    the cells the first one could not get an answer for, so a cell holds either
    one row or two, and when it holds two exactly one of them succeeded. The
    successful row is the observation and the refused row stays in the raw file
    as the record of what happened.

    Two successful rows in one cell is not a case with a rule. It would mean
    the same question was asked twice and somebody has to choose which answer
    counts -- which is the choice this whole package exists to avoid -- so it
    stops the analysis and names the cell instead.
    """
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("exp_id") != E2_EXP_ID or row.get("tier") != E2_TIER:
            continue
        grouped[(str(row.get("request_model")), str(row.get("layer")),
                 str(row.get("order")), str(row.get("pair_id")))].append(row)
    chosen: list[dict] = []
    cells_twice = 0
    superseded = 0
    for key in sorted(grouped):
        # Sorted so that "the first row" is a property of the data, not of the
        # order rglob happened to return the files in.
        group = sorted(grouped[key], key=lambda r: (str(r.get("timestamp_utc")), str(r.get("call_id"))))
        ok_rows = [r for r in group if r.get("ok")]
        if len(ok_rows) > 1:
            where = "; ".join(f"{r.get('_source_file')} call_id={r.get('call_id')}" for r in ok_rows)
            raise SystemExit(
                "E2 design cell " + " / ".join(key) + f" has {len(ok_rows)} successful rows ({where}). "
                "One cell is one question asked once: two answers cannot be resolved by a rule, so the "
                "analysis stops rather than picking one."
            )
        if len(group) > 1:
            cells_twice += 1
            superseded += len(group) - 1
        chosen.append(ok_rows[0] if ok_rows else group[0])
    return chosen, {
        "cells": len(chosen),
        "cells_present_in_two_batches": cells_twice,
        "rows_superseded_by_a_later_answer": superseded,
    }


def e2_calls(rows: list[dict]) -> list[dict]:
    """One entry per E2 design cell, with the verdict translated out of
    position space.

    `choice` is "A" or "B" -- a position. `winner` is the summary *source* that
    position held on this call, which is the same quantity in both orders and
    the only one worth comparing across them.
    """
    out: list[dict] = []
    for row in e2_cell_rows(rows)[0]:
        choice = (row.get("parsed") or {}).get("choice") if row.get("ok") else None
        if choice == "A":
            winner, first_win = row["position_a_source"], 1
        elif choice == "B":
            winner, first_win = row["position_b_source"], 0
        elif choice == "tie":
            winner, first_win = "tie", None
        else:
            winner, first_win = None, None
        out.append({
            "judge_model": row["request_model"],
            "response_model": row.get("response_model"),
            "source_file": row.get("_source_file"),
            "layer": row["layer"],
            "order": row["order"],
            "pair_id": row["pair_id"],
            "case_id": row["case_id"],
            "ok": bool(row.get("ok")),
            "error_type": row.get("error_type"),
            "status_code": row.get("status_code"),
            "choice": choice,
            "winner": winner,
            "first_position_win": first_win,
            "left_source": row["left_source"],
            "right_source": row["right_source"],
            "summaries_identical": bool(row.get("summaries_identical")),
        })
    return out


def e2_pairs(calls: list[dict]) -> list[dict]:
    """One entry per (judge model, pair) judged in both orders.

    A pair is dropped when either of its two calls failed, because "the verdict
    survived the swap" is not answerable with one verdict. Dropped pairs are
    counted into `n_excluded` rather than being scored as inconsistent.
    """
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for call in calls:
        grouped[(call["judge_model"], call["layer"], call["pair_id"])][call["order"]] = call
    out: list[dict] = []
    for key in sorted(grouped):
        model, layer, pair_id = key
        both = grouped[key]
        first, second = both.get(E2_ORDERS[0]), both.get(E2_ORDERS[1])
        complete = bool(first and second and first["ok"] and second["ok"])
        out.append({
            "judge_model": model,
            "layer": layer,
            "pair_id": pair_id,
            "case_id": (first or second)["case_id"],
            "complete": complete,
            "consistent": int(first["winner"] == second["winner"]) if complete else None,
            "winner": first["winner"] if complete and first["winner"] == second["winner"] else None,
            "left_source": (first or second)["left_source"],
            "right_source": (first or second)["right_source"],
            "summaries_identical": (first or second)["summaries_identical"],
        })
    return out


def _e2_rate_row(row_type, family, model, layer, comparison, values, note, n_excluded=None):
    n, k = len(values), sum(values)
    if n == 0:
        return None
    lo, hi = wilson_ci(k, n)
    return [row_type, family, model, layer, comparison, n, k, k / n, lo, hi, hi - lo,
            None, None, None, None, "", n_excluded, note]


def table_e2_consistency(rows: list[dict]) -> tuple[list[str], list[list]]:
    """Order consistency: for each pair, did swapping A and B change which
    summary won?

    The four cells (two judge models x two layers) are the headline. The tests
    underneath are labelled in `row_type` because they are a different shape of
    row: the model comparison is paired (the same 70 pairs go to both models),
    and the layer comparison is reported both ways -- Fisher exact as
    pre-registered, and a paired McNemar as a sensitivity row, because the easy
    and hard layers are built from the same 70 emails and are therefore not
    independent samples.
    """
    calls = e2_calls(rows)
    if not calls:
        return E2_CONSISTENCY_HEADER, []
    pairs = e2_pairs(calls)
    models = sorted({p["judge_model"] for p in pairs})
    out: list[list] = []

    def subset(model=None, layer=None):
        return [p for p in pairs
                if (model is None or p["judge_model"] == model)
                and (layer is None or p["layer"] == layer)]

    # Coverage first, on purpose. The 2026-09-15 run stopped 94 calls in when
    # the account hit its API spend limit, so most of these cells are empty and
    # the ones that are not are partial. A reader who sees a rate before seeing
    # how much of the design produced it can quote the rate by accident.
    for model in sorted({c["judge_model"] for c in calls}):
        for layer in E2_LAYERS:
            group = [c for c in calls if c["judge_model"] == model and c["layer"] == layer]
            if not group:
                continue
            ok = [c for c in group if c["ok"]]
            complete = [p for p in subset(model, layer) if p["complete"]]
            errors = sorted({str(c["error_type"]) for c in group if not c["ok"]})
            by_batch: dict[str, int] = defaultdict(int)
            for call in ok:
                by_batch[Path(str(call["source_file"])).name] += 1
            batches = ", ".join(f"{name}={n}" for name, n in sorted(by_batch.items())) or "none"
            out.append([
                "coverage", E5_FAMILY_DESCRIPTIVE, model, layer, "design cells",
                len(group), len(ok), len(ok) / len(group), None, None, None,
                None, None, None, None, "", len(group) - len(ok),
                f"{len(complete)} of {len(group) // len(E2_ORDERS)} pairs have both orders; "
                f"failure types: {', '.join(errors) if errors else 'none'}; "
                f"answered rows by batch: {batches}",
            ])

    unit_note = "unit = one pair judged in both orders; consistent means the same summary source won both times (tie counts as a verdict)"
    for model in [*models, None]:
        for layer in [*E2_LAYERS, None]:
            group = subset(model, layer)
            complete = [p for p in group if p["complete"]]
            row = _e2_rate_row(
                "rate", E2_FAMILY_CONSISTENCY if (model and layer) else E5_FAMILY_DESCRIPTIVE,
                model or E2_ALL, layer or E2_ALL, "order swapped",
                [p["consistent"] for p in complete], unit_note,
                n_excluded=len(group) - len(complete),
            )
            if row:
                out.append(row)

    # Which source won, among the pairs where the verdict survived the swap and
    # was not a tie. Descriptive: the easy layer's two sources are a baseline
    # and a degradation, but neither is a gold answer, so this is "what the
    # judge preferred", not "which summary was better".
    for model in models:
        for layer in E2_LAYERS:
            decided = [p for p in subset(model, layer) if p["complete"] and p["winner"] not in (None, "tie")]
            if not decided:
                continue
            left, right = decided[0]["left_source"], decided[0]["right_source"]
            row = _e2_rate_row(
                "winner_share", E5_FAMILY_DESCRIPTIVE, model, layer, f"{left} vs {right}",
                [int(p["winner"] == left) for p in decided],
                f"share of decisive consistent pairs won by {left}; "
                f"{len(subset(model, layer)) - len(decided)} of {len(subset(model, layer))} pairs were a tie or order-dependent",
            )
            if row:
                out.append(row)

    # Model comparison: paired, same pair ids on both models.
    bh_idx: list[int] = []
    if len(models) == 2:
        m1, m2 = models
        for layer in [*E2_LAYERS, None]:
            by_pair = {}
            for pair in pairs:
                if layer is not None and pair["layer"] != layer:
                    continue
                if not pair["complete"]:
                    continue
                by_pair.setdefault(pair["pair_id"], {})[pair["judge_model"]] = pair["consistent"]
            shared = sorted(k for k, v in by_pair.items() if m1 in v and m2 in v)
            if not shared:
                continue
            b = sum(1 for k in shared if by_pair[k][m1] == 1 and by_pair[k][m2] == 0)
            c = sum(1 for k in shared if by_pair[k][m1] == 0 and by_pair[k][m2] == 1)
            res = mcnemar_exact(b, c)
            bh_idx.append(len(out))
            out.append([
                "test", E2_FAMILY_CONSISTENCY, E2_ALL, layer or E2_ALL, f"{m1} vs {m2}",
                len(shared), None, None, None, None, None,
                b, c, res["p_value"], None, res["method"], None,
                f"paired on pair_id; b = consistent under {m1} only, c = consistent under {m2} only",
            ])

    # Layer comparison: the pre-registered unpaired test, then the paired one.
    for model in [*models, None]:
        easy = [p for p in subset(model, "easy") if p["complete"]]
        hard = [p for p in subset(model, "hard") if p["complete"]]
        if not easy or not hard:
            continue
        ke, kh = sum(p["consistent"] for p in easy), sum(p["consistent"] for p in hard)
        fisher = fisher_exact_2x2(ke, len(easy) - ke, kh, len(hard) - kh)
        bh_idx.append(len(out))
        out.append([
            "test", E2_FAMILY_CONSISTENCY, model or E2_ALL, "easy vs hard", "layer",
            len(easy) + len(hard), None, fisher["difference"], None, None, None,
            None, None, fisher["p_value"], None, fisher["method"], None,
            f"easy {ke}/{len(easy)} vs hard {kh}/{len(hard)}; estimate is the difference in rates. "
            "Fisher exact assumes two independent samples and these two layers are built from the "
            "same 70 emails, so the paired row below is the honest version",
        ])
        # Keyed by (judge model, case), not by case: in the pooled row both
        # models contribute a pair for the same email, and a case-only key
        # kept whichever model was iterated last -- a row labelled "(both)"
        # that reported one model. Found on 2026-09-16 after the full run,
        # recorded in docs/PREREGISTRATION.md section 9.
        paired_keys = {(p["judge_model"], p["pair_id"].split(":", 1)[1]): p["consistent"] for p in easy}
        hard_by_case = {(p["judge_model"], p["pair_id"].split(":", 1)[1]): p["consistent"] for p in hard}
        shared = sorted(set(paired_keys) & set(hard_by_case))
        b = sum(1 for k in shared if paired_keys[k] == 1 and hard_by_case[k] == 0)
        c = sum(1 for k in shared if paired_keys[k] == 0 and hard_by_case[k] == 1)
        res = mcnemar_exact(b, c)
        out.append([
            "test", E2_FAMILY_PAIRED_LAYER, model or E2_ALL, "easy vs hard", "layer (paired by case)",
            len(shared), None, None, None, None, None,
            b, c, res["p_value"], None, res["method"], None,
            "sensitivity: the same email appears in both layers, so the layers are paired; "
            "no multiplicity adjustment, this row is not in any family",
        ])

    col = {name: i for i, name in enumerate(E2_CONSISTENCY_HEADER)}
    if bh_idx:
        adjusted = bh_adjust([out[i][col["p_value"]] for i in bh_idx])
        for i, value in zip(bh_idx, adjusted):
            out[i][col["p_bh"]] = value
    return E2_CONSISTENCY_HEADER, out


def _e2_position_row(row_type, family, model, layer, subset_label, calls, seed, n_boot, note):
    ok_calls = [c for c in calls if c["ok"]]
    decisive = [c for c in ok_calls if c["first_position_win"] is not None]
    n_tie = sum(1 for c in ok_calls if c["choice"] == "tie")
    if not ok_calls:
        return None
    k = sum(c["first_position_win"] for c in decisive)
    n = len(decisive)
    if n:
        lo, hi = wilson_ci(k, n)
        p_binom = exact_binomial_test(k, n, 0.5)
        clusters: dict[str, list[int]] = defaultdict(list)
        for c in decisive:
            clusters[c["case_id"]].append(c["first_position_win"])
        units = [clusters[cid] for cid in sorted(clusters)]
        boot = bootstrap_ci(
            units,
            lambda u: sum(sum(x) for x in u) / sum(len(x) for x in u),
            seed=seed, n_boot=n_boot,
        )
    else:
        lo = hi = p_binom = None
        boot = {"ci_low": None, "ci_high": None, "n_units": 0}
    return [
        row_type, family, model, layer, subset_label,
        len(calls), len(ok_calls), n_tie, n_tie / len(ok_calls), n, k if n else None,
        (k / n) if n else None, lo, hi, p_binom, None,
        boot["ci_low"], boot["ci_high"], boot["n_units"], n_boot if n else None, seed if n else None,
        "exact_binomial_vs_0.5" if n else "", note,
    ]


def table_e2_position(rows: list[dict], seed: int, n_boot: int) -> tuple[list[str], list[list]]:
    """How often the judge picked whichever summary was shown first.

    Under no position preference this is 0.5. The pre-registered test is the
    exact binomial (PREREGISTRATION section 6); it treats the calls as
    independent, which they are not -- each email contributes up to eight of
    them -- so every row also carries a case-cluster bootstrap interval, and
    the two are printed side by side rather than one replacing the other.

    The last rows are the pairs whose two summaries are byte-identical. There
    the correct answer is "tie" by construction, so any A or B is position
    preference with nothing else mixed in. n is tiny; it is reported as a
    count, not as a rate with an interval to be quoted.
    """
    calls = e2_calls(rows)
    if not calls:
        return E2_POSITION_HEADER, []
    models = sorted({c["judge_model"] for c in calls})
    out: list[list] = []
    bh_idx: list[int] = []

    def subset(model=None, layer=None, identical=None):
        return [c for c in calls
                if (model is None or c["judge_model"] == model)
                and (layer is None or c["layer"] == layer)
                and (identical is None or c["summaries_identical"] == identical)]

    for model in [*models, None]:
        for layer in [*E2_LAYERS, None]:
            group = subset(model, layer)
            is_cell = bool(model and layer)
            row = _e2_position_row(
                "position", E2_FAMILY_POSITION if is_cell else E5_FAMILY_DESCRIPTIVE,
                model or E2_ALL, layer or E2_ALL, "all pairs", group, seed, n_boot,
                E2_TIE_NOTE + "; " + E2_DEPENDENCE_NOTE,
            )
            if row is None:
                continue
            if is_cell:
                bh_idx.append(len(out))
            out.append(row)

    for model in [*models, None]:
        group = subset(model, None, True)
        if not group:
            continue
        row = _e2_position_row(
            "position", E5_FAMILY_DESCRIPTIVE, model or E2_ALL, "hard", "identical summaries",
            group, seed, n_boot,
            "both candidates are the same string, so a tie is the only verdict that is not a "
            "position preference; no interval from this many calls is worth quoting",
        )
        if row:
            out.append(row)

    col = {name: i for i, name in enumerate(E2_POSITION_HEADER)}
    if bh_idx:
        adjusted = bh_adjust([out[i][col["p_binomial"]] for i in bh_idx])
        for i, value in zip(bh_idx, adjusted):
            out[i][col["p_bh"]] = value
    return E2_POSITION_HEADER, out


def figure_e2_consistency(consistency_rows: list[list], position_rows: list[list], out_path: Path, seed: int) -> str | None:
    """Two panels of the same four cells: how often the verdict survived the
    swap, and how often the first position won. The second panel is the reason
    the first one matters."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover -- documented fallback
        print(f"!! matplotlib unavailable ({exc}); CSV tables were still written", file=sys.stderr)
        return None
    ccol = {name: i for i, name in enumerate(E2_CONSISTENCY_HEADER)}
    pcol = {name: i for i, name in enumerate(E2_POSITION_HEADER)}
    cells = [r for r in consistency_rows
             if r[ccol["row_type"]] == "rate" and r[ccol["judge_model"]] != E2_ALL and r[ccol["layer"]] != E2_ALL]
    pos = [r for r in position_rows
           if r[pcol["row_type"]] == "position" and r[pcol["subset"]] == "all pairs"
           and r[pcol["judge_model"]] != E2_ALL and r[pcol["layer"]] != E2_ALL]
    if not cells or not pos:
        return None
    # The design is two judge models x two layers. A figure drawn from fewer
    # cells looks exactly like the designed one to anybody who does not read
    # the axis labels, so an incomplete run gets no figure and a reason on
    # stderr instead.
    designed = len(E2_LAYERS) * len({r[ccol["judge_model"]] for r in consistency_rows
                                     if r[ccol["row_type"]] == "coverage"})
    if designed and len(cells) < designed:
        print(f"!! e2: {len(cells)} of {designed} (model x layer) cells have complete pairs; "
              "figure skipped rather than drawn from part of the design", file=sys.stderr)
        return None
    labels = [f"{r[ccol['judge_model']]}\n{r[ccol['layer']]}" for r in cells]
    ypos = list(range(len(cells)))[::-1]
    pos_by_key = {(r[pcol["judge_model"]], r[pcol["layer"]]): r for r in pos}
    tie_overall = next((r for r in position_rows
                        if r[pcol["judge_model"]] == E2_ALL and r[pcol["layer"]] == E2_ALL
                        and r[pcol["subset"]] == "all pairs"), None)
    footnote = _wrap(
        f"Left: one pair judged in both orders, {cells[0][ccol['n']]} pairs per cell; consistent = the same "
        f"summary source won both times. Right: share of decisive calls that picked the summary in position A "
        f"(ties excluded, {E2_TIE_NOTE.split(';')[0]}); dashed line at 0.5 is no preference. "
        + (f"Tie rate overall {tie_overall[pcol['tie_rate']]:.3f} of {tie_overall[pcol['n_ok']]} calls. "
           if tie_overall else "")
        + f"95% Wilson intervals; exact binomial against 0.5; BH within the four cells; all exploratory. "
          f"Analysis seed = {seed}.",
        width=112,
    )
    footnote_lines = footnote.count("\n") + 1

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 0.62 * len(cells) + 2.4 + 0.16 * footnote_lines))
    palette = {"easy": "#1f4e79", "hard": "#7c4a2d"}
    ax = axes[0]
    for y, r in zip(ypos, cells):
        est = r[ccol["estimate"]]
        ax.errorbar([est], [y], xerr=[[est - r[ccol["wilson_lo"]]], [r[ccol["wilson_hi"]] - est]],
                    fmt="o", color=palette.get(r[ccol["layer"]], "#333333"), capsize=4)
        ax.annotate(f"{r[ccol['k']]}/{r[ccol['n']]}", (est, y + 0.22), fontsize=7.5, ha="center", color="#333333")
    ax.axvline(1.0, color="#555555", lw=1.0, ls=":")
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("order consistency (95% Wilson)")
    ax.set_title("Same verdict after swapping A and B")

    ax2 = axes[1]
    for y, r in zip(ypos, cells):
        pr = pos_by_key.get((r[ccol["judge_model"]], r[ccol["layer"]]))
        if pr is None or pr[pcol["estimate"]] is None:
            continue
        est = pr[pcol["estimate"]]
        ax2.errorbar([est], [y], xerr=[[est - pr[pcol["wilson_lo"]]], [pr[pcol["wilson_hi"]] - est]],
                     fmt="s", color=palette.get(r[ccol["layer"]], "#333333"), capsize=4, markerfacecolor="none")
        ax2.annotate(f"{pr[pcol['k_first_position']]}/{pr[pcol['n_decisive']]}  p={pr[pcol['p_binomial']]:.3f}",
                     (est, y + 0.22), fontsize=7.5, ha="center", color="#333333")
    ax2.axvline(0.5, color="#555555", lw=1.0, ls="--")
    ax2.set_xlim(0.0, 1.0)
    ax2.set_xlabel("first-position win rate, ties excluded (95% Wilson)")
    ax2.set_title("Did position A win more than half the time?")

    for ax_ in axes:
        ax_.set_ylim(-0.6, len(cells) - 0.4)
        ax_.set_yticks(ypos)
        ax_.grid(axis="x", alpha=0.3)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[1].set_yticklabels([])
    fig.suptitle("E2: an LLM judge asked the same question twice, in two orders")
    fig.text(0.01, 0.01, footnote, fontsize=7, va="bottom")
    fig.subplots_adjust(bottom=(0.55 + 0.14 * footnote_lines) / fig.get_figheight(),
                        left=0.14, right=0.985, top=0.84, wspace=0.08)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, metadata={"Software": "experiments/analyze.py"})
    plt.close(fig)
    return rel_to_repo(out_path)


# --------------------------------------------------------------------------
# E4: agreement between raters (exploratory)
# --------------------------------------------------------------------------
def _landis_koch(kappa: float) -> str:
    """Landis & Koch (1977) Table 1 band for a kappa. A reading convention
    quoted with its source, not a threshold anything in this study depends on."""
    for upper, name in ((0.0, "poor"), (0.20, "slight"), (0.40, "fair"),
                        (0.60, "moderate"), (0.80, "substantial")):
        if kappa <= upper:
            return name
    return "almost perfect"


def e4_rater_labels(rows: list[dict], dataset_path: Path) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    """rater -> {case_id: label}, plus the failed-call count per model rater.

    The gold rater is read from the dataset, not from any run. A model rater
    contributes only its successful calls; a failed call leaves that case out
    of the pair rather than scoring it as a disagreement (PREREGISTRATION
    section 7).
    """
    labels: dict[str, dict[str, str]] = {
        E4_GOLD_RATER: {c.id: c.expected_category for c in load_all_cases(dataset_path)}
    }
    failures: dict[str, int] = defaultdict(int)
    for row in rows:
        if row.get("exp_id") != E4_EXP_ID or row.get("tier") != E4_TIER:
            continue
        rater = f"model:{row.get('request_model')}"
        if not row.get("ok"):
            failures[rater] += 1
            continue
        category = (row.get("parsed") or {}).get("category")
        if not category:
            failures[rater] += 1
            continue
        labels.setdefault(rater, {})[row["case_id"]] = category
    if E4_HUMAN2_PATH.exists():
        data = json.loads(E4_HUMAN2_PATH.read_text(encoding="utf-8"))
        labels[E4_HUMAN2_RATER] = {r["case_id"]: r["label"] for r in data["labels"]}
    return labels, dict(failures)


def e4_rater_pairs(labels: dict[str, dict[str, str]]) -> list[tuple[str, str]]:
    """Which pairs go in the table, in a fixed order: every rater against the
    gold labels first, then the second human against each model, then the
    models against each other."""
    models = sorted(r for r in labels if r.startswith("model:"))
    human2 = [E4_HUMAN2_RATER] if E4_HUMAN2_RATER in labels else []
    pairs = [(E4_GOLD_RATER, other) for other in human2 + models]
    pairs += [(E4_HUMAN2_RATER, m) for m in models if human2]
    pairs += [(models[i], models[j]) for i in range(len(models)) for j in range(i + 1, len(models))]
    return pairs


def table_e4_kappa(rows: list[dict], dataset_path: Path, seed: int, n_boot: int) -> tuple[list[str], list[list]]:
    """Cohen's kappa for each pair of raters, with a case-level bootstrap
    interval.

    What the interval covers and what it does not: resampling cases says how
    much of the number is these 70 emails. It says nothing about annotator
    variance -- there is one second human and one call per model per case, so
    "another annotator would have got this" is outside what the design can
    estimate (PREREGISTRATION section 8).
    """
    labels, failures = e4_rater_labels(rows, dataset_path)
    if set(labels) == {E4_GOLD_RATER}:
        return E4_KAPPA_HEADER, []
    out: list[list] = []
    for rater_a, rater_b in e4_rater_pairs(labels):
        a, b = labels[rater_a], labels[rater_b]
        shared = sorted(set(a) & set(b))
        if not shared:
            continue
        units = [(a[cid], b[cid]) for cid in shared]
        result = cohen_kappa([u[0] for u in units], [u[1] for u in units])
        boot = bootstrap_ci(
            units,
            lambda u: cohen_kappa([x[0] for x in u], [x[1] for x in u])["kappa"],
            seed=seed, n_boot=n_boot,
        )
        excluded = (len(set(a) | set(b)) - len(shared))
        note = E4_KAPPA_NOTE + "; " + E4_BAND_NOTE
        for rater in (rater_a, rater_b):
            if failures.get(rater):
                note += f"; {rater} has {failures[rater]} failed call(s), excluded from the pairing"
        out.append([
            E4_FAMILY, rater_a, rater_b, len(shared), excluded,
            result["po"], result["pe"], result["kappa"],
            boot["ci_low"], boot["ci_high"], n_boot, seed,
            _landis_koch(result["kappa"]), result["method"], note,
        ])
    return E4_KAPPA_HEADER, out


def table_e4_confusions(rows: list[dict], dataset_path: Path) -> list[tuple[str, tuple[list[str], list[list]]]]:
    """One confusion table per model rater: gold label down the side, the
    model's label across the top. Written only for raters that have labels, so
    an absent file means an arm that has not run rather than an empty result."""
    labels, _ = e4_rater_labels(rows, dataset_path)
    gold = labels[E4_GOLD_RATER]
    specs: list[tuple[str, tuple[list[str], list[list]]]] = []
    for rater in sorted(r for r in labels if r.startswith("model:")):
        model = rater.split(":", 1)[1]
        other = labels[rater]
        shared = sorted(set(gold) & set(other))
        if not shared:
            continue
        body: list[list] = []
        for row_label in E4_CATEGORIES:
            counts = [sum(1 for cid in shared if gold[cid] == row_label and other[cid] == col)
                      for col in E4_CATEGORIES]
            body.append([
                E4_GOLD_RATER, rater, row_label, *counts, sum(counts),
                "row = gold label, columns = the model's label on the same email",
            ])
        totals = [sum(1 for cid in shared if other[cid] == col) for col in E4_CATEGORIES]
        body.append([
            E4_GOLD_RATER, rater, "(all)", *totals, len(shared),
            f"column totals; {len(shared)} cases labelled by both raters",
        ])
        specs.append((f"e4_confusion_{model}.csv", (E4_CONFUSION_HEADER, body)))
    return specs


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
        if row[header.index("family")] == E1_FAMILY_LEAK_SENSITIVITY:
            continue
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
            "experiment_prompts": {
                rel_to_repo(p): sha256_file(p)
                for p in sorted((REPO_ROOT / "experiments" / "prompts").glob("*.yaml"))
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
    e5_strata_spec = table_e5_strata(observations)
    e2_consistency_spec = table_e2_consistency(rows)
    e2_position_spec = table_e2_position(rows, args.seed, args.n_boot)
    e4_kappa_spec = table_e4_kappa(rows, dataset_path, args.seed, args.n_boot)
    if not e4_kappa_spec[1]:
        print(f"e4: {E4_EMPTY_NOTE}")
    e1_power_spec = table_e1_power(pair_sets, args.seed, n_sim=args.power_n_sim)

    rescore_header, rescore_rows = table_judge_rescore(rows)
    specs = [
        ("per_case_outcomes.csv", table_per_case(observations)),
        ("rates_by_run.csv", table_rates(observations)),
        ("rates_run_spread.csv", table_run_spread(observations)),
        ("per_case_instability.csv", table_case_instability(observations)),
        ("rates_by_stratum.csv", table_strata(observations)),
        ("e5_strata.csv", e5_strata_spec),
        ("e5_logit.csv", table_e5_logit(observations)),
        ("e2_consistency.csv", e2_consistency_spec),
        ("e2_position_pref.csv", e2_position_spec),
        ("e4_kappa.csv", e4_kappa_spec),
        *table_e4_confusions(rows, dataset_path),
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
            figure_e5_forest(e5_strata_spec[1], figures_dir / "e5_forest.png", args.seed),
            figure_e2_consistency(e2_consistency_spec[1], e2_position_spec[1],
                                  figures_dir / "e2_consistency.png", args.seed),
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
