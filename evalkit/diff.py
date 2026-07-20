"""Regression diffing between two eval reports (SPEC.md §5).

Severity is driven by the overall pass_rate delta: abs(delta) > warn
threshold -> "warning" (exit 0, flagged in the report); abs(delta) > critical
threshold -> "critical" (exit non-zero). Per-category deltas are computed and
reported alongside, but do not independently escalate severity -- see
docs/DECISIONS.md for why (spec text binds the threshold sentence to the
headline pass_rate metric; per-category is diagnostic detail).

If --baseline points at a file that doesn't exist yet, this is treated as
the "first run bootstraps the baseline" case from SPEC.md §5: the candidate
report is copied to the baseline path, no diff_report.json is produced, and
the process exits 0.

Usage: python -m evalkit.diff --baseline eval_reports/baseline.json --candidate eval_report.json \
           --warn-threshold 0.03 --critical-threshold 0.08 --out diff_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _pass_map(report: dict) -> dict[str, bool]:
    return {c["id"]: bool(c["passed"]) for c in report["cases"]}


def _category_accuracy(report: dict) -> dict[str, float]:
    by_cat: dict[str, list[bool]] = {}
    for c in report["cases"]:
        by_cat.setdefault(c["expected_category"], []).append(bool(c["passed"]))
    return {cat: sum(v) / len(v) for cat, v in by_cat.items()}


def _case_by_id(report: dict, case_id: str) -> dict:
    return next(c for c in report["cases"] if c["id"] == case_id)


def _flip_record(baseline: dict, candidate: dict, case_id: str) -> dict:
    b, c = _case_by_id(baseline, case_id), _case_by_id(candidate, case_id)
    return {
        "id": case_id,
        "baseline_predicted_category": b.get("predicted_category"),
        "candidate_predicted_category": c.get("predicted_category"),
        "baseline_predicted_summary": b.get("predicted_summary"),
        "candidate_predicted_summary": c.get("predicted_summary"),
        "baseline_judge_score": b.get("judge_score"),
        "candidate_judge_score": c.get("judge_score"),
    }


def diff_reports(baseline: dict, candidate: dict, warn_threshold: float, critical_threshold: float) -> dict:
    base_pass, cand_pass = _pass_map(baseline), _pass_map(candidate)
    common_ids = set(base_pass) & set(cand_pass)

    regressions, improvements = [], []
    for case_id in sorted(common_ids):
        if base_pass[case_id] and not cand_pass[case_id]:
            regressions.append(_flip_record(baseline, candidate, case_id))
        elif not base_pass[case_id] and cand_pass[case_id]:
            improvements.append(_flip_record(baseline, candidate, case_id))

    overall_delta = candidate["pass_rate"] - baseline["pass_rate"]
    if abs(overall_delta) > critical_threshold:
        severity = "critical"
    elif abs(overall_delta) > warn_threshold:
        severity = "warning"
    else:
        severity = "ok"

    base_cat, cand_cat = _category_accuracy(baseline), _category_accuracy(candidate)
    per_category_delta = {
        cat: round(cand_cat.get(cat, 0.0) - base_cat.get(cat, 0.0), 4) for cat in sorted(set(base_cat) | set(cand_cat))
    }

    return {
        "baseline_prompt_version": baseline.get("prompt_version"),
        "candidate_prompt_version": candidate.get("prompt_version"),
        "baseline_pass_rate": baseline["pass_rate"],
        "candidate_pass_rate": candidate["pass_rate"],
        "overall_delta": round(overall_delta, 4),
        "per_category_delta": per_category_delta,
        "warn_threshold": warn_threshold,
        "critical_threshold": critical_threshold,
        "severity": severity,
        "regressions": regressions,
        "improvements": improvements,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diff two eval reports and gate on regression severity.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--warn-threshold", type=float, default=0.03)
    parser.add_argument("--critical-threshold", type=float, default=0.08)
    parser.add_argument("--out", default="diff_report.json")
    parser.add_argument("--html-out", default=None, help="optional path to also render a static HTML diff report")
    args = parser.parse_args(argv)

    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)

    if not baseline_path.exists():
        candidate_data = json.loads(candidate_path.read_text(encoding="utf-8"))
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(candidate_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"no baseline at {baseline_path} -- bootstrapped from candidate, no diff produced")
        return 0

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    report = diff_reports(baseline, candidate, args.warn_threshold, args.critical_threshold)

    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.html_out:
        from evalkit.report import render_diff_html

        render_diff_html(report, args.html_out)

    print(
        f"severity={report['severity']} overall_delta={report['overall_delta']:+.4f} "
        f"regressions={len(report['regressions'])} improvements={len(report['improvements'])}"
    )
    return 1 if report["severity"] == "critical" else 0


if __name__ == "__main__":
    raise SystemExit(main())
