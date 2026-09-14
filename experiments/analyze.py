"""Analysis: raw JSONL in, CSV tables + PNG figures + MANIFEST.json out.

Reads only from experiments/results/raw/ and never calls an API, so it can be
re-run any number of times for free. Deterministic for a fixed --seed: the
same raw files produce byte-identical CSVs, which is what
checks/experiments_acceptance.sh asserts.

    python -m experiments.analyze --seed 20260920

What is implemented (validated against scipy in `python -m experiments.stats
--cross-check`): Wilson interval, Clopper-Pearson exact interval, exact
McNemar, case-level cluster bootstrap, Holm and BH adjustment.

What is not implemented yet, and is therefore absent from the output rather
than approximated: Newcombe method 10 CI for a paired risk difference,
Fleiss kappa, Krippendorff alpha, the logistic model with clustered standard
errors, and the power curve. See the TODO list in experiments/README.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evalkit.dataset import load_all_cases  # noqa: E402
from evalkit.run_eval import DEFAULT_PASS_THRESHOLD  # noqa: E402
from experiments.stats import bootstrap_ci, clopper_pearson_ci, holm_adjust, mcnemar_exact, wilson_ci  # noqa: E402

DEFAULT_SEED = 20260920
DEFAULT_N_BOOT = 10000
FLOAT_FMT = "{:.6f}"
BASELINE_PROMPT_VERSION = "v1"


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
            "cost_usd": round((clf or {}).get("cost_usd", 0.0) + (judge or {}).get("cost_usd", 0.0), 8),
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
    """Paired comparison of each non-baseline prompt version against v1 on the
    cases both versions scored. Empty until a v2* run exists."""
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
    pvals_primary: list[float] = []
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
            family = "confirmatory_holm" if metric == "category_match" else "exploratory_bh"
            if family == "confirmatory_holm":
                pvals_primary.append(res["p_value"])
            raw_rows.append([
                metric, BASELINE_PROMPT_VERSION, candidate, len(keys), a, b, c, d, rd,
                res["p_value"], None, res["method"], family,
                "CI omitted: newcombe_paired_diff_ci not implemented yet (experiments/stats.py)",
            ])

    adjusted = holm_adjust(pvals_primary)
    idx = 0
    for row in raw_rows:
        if row[12] == "confirmatory_holm":
            row[10] = adjusted[idx]
            idx += 1
    return header, raw_rows


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
        total = sum(r.get("cost_usd", 0.0) for r in group)
        p50 = median(lats) if lats else None
        p95 = lats[min(len(lats) - 1, int(round(0.95 * (len(lats) - 1))))] if lats else None
        out.append([
            key[0], key[1], key[2], key[3], key[4], len(group), len(ok), len(group) - len(ok),
            sum(ins) / len(ins) if ins else None, median(ins) if ins else None,
            sum(outs) / len(outs) if outs else None, median(outs) if outs else None,
            p50, p95, total, total / len(group) if group else None,
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
# manifest
# --------------------------------------------------------------------------
def build_manifest(files: list[Path], rows: list[dict], out_dir: Path, seed: int, n_boot: int, tables: list[str], figures: list[str], dataset_path: Path) -> dict:
    by_file: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_file[row["_source_file"]].append(row)

    entries = []
    raw_cost = 0.0
    for path in files:
        rel = rel_to_repo(path)
        group = by_file.get(rel, [])
        cost = sum(r.get("cost_usd", 0.0) for r in group)
        raw_cost += cost
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
            "probes_cost_usd": round(probe_cost, 8),
            "total_cost_usd": round(raw_cost + probe_cost, 8),
            "cost_ledger_cumulative_usd": ledger_cumulative,
            "cost_ledger_note": "the ledger records runner runs only; probe scripts are accounted separately",
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
    args = p.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    dataset_path = Path(args.dataset)
    exp_ids = [s.strip() for s in args.exp_ids.split(",")] if args.exp_ids else None

    rows, files = load_raw(raw_dir, exp_ids)
    if not rows:
        raise SystemExit(f"no raw JSONL rows under {raw_dir} (run experiments/runner.py first)")
    observations = build_observations(rows, dataset_path)

    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"

    specs = [
        ("per_case_outcomes.csv", table_per_case(observations)),
        ("rates_by_run.csv", table_rates(observations)),
        ("rates_by_stratum.csv", table_strata(observations)),
        ("rates_bootstrap.csv", table_bootstrap(observations, args.seed, args.n_boot)),
        ("paired_mcnemar.csv", table_mcnemar(observations)),
        ("tokens_latency_by_tier.csv", table_tokens(rows)),
        ("judge_score_distribution.csv", table_judge_scores(observations)),
    ]
    written: list[str] = []
    for name, (header, body) in specs:
        write_csv(tables_dir / name, header, body)
        written.append(rel_to_repo(tables_dir / name))
        print(f"{name:32s} {len(body):6d} rows")

    figures: list[str] = []
    if not args.no_figures:
        rate_header, rate_rows = specs[1][1]
        score_header, score_rows = specs[6][1]
        for fig in (
            figure_rates(rate_rows, figures_dir / "primary_metric_by_run.png", args.seed),
            figure_judge_scores(score_rows, figures_dir / "judge_score_distribution.png", args.seed),
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
