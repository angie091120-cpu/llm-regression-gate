"""evalkit/diff.py: regression diffing logic + CLI, exercised against the
hand-built fixtures in tests/fixtures/ (SPEC.md §5, AC3). No network calls --
these are pre-generated JSON reports, not live eval runs."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evalkit.diff import diff_reports, main

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).parent.parent


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_diff_reports_unit_logic_matches_known_regressions():
    baseline = _load("baseline_report.json")
    candidate = _load("degraded_report.json")
    known_regressions = set(json.loads((FIXTURES / "known_regressions.json").read_text(encoding="utf-8")))

    report = diff_reports(baseline, candidate, warn_threshold=0.03, critical_threshold=0.08)

    assert report["severity"] == "critical"
    assert report["overall_delta"] == pytest.approx(0.7 - 1.0, abs=1e-6)
    found_ids = {r["id"] for r in report["regressions"]}
    assert known_regressions <= found_ids
    assert report["improvements"] == []


def test_diff_reports_ok_when_no_change():
    baseline = _load("baseline_report.json")
    report = diff_reports(baseline, baseline, warn_threshold=0.03, critical_threshold=0.08)
    assert report["severity"] == "ok"
    assert report["regressions"] == []
    assert report["improvements"] == []


def test_diff_reports_warning_band():
    baseline = _load("baseline_report.json")
    warm = json.loads(json.dumps(baseline))  # deep copy
    warm["pass_rate"] = 0.95  # 5% drop from 1.0 -> between 3% and 8%
    report = diff_reports(baseline, warm, warn_threshold=0.03, critical_threshold=0.08)
    assert report["severity"] == "warning"


def test_diff_main_cli_exits_nonzero_and_writes_known_regressions(tmp_path):
    out_path = tmp_path / "diff_report.json"
    exit_code = main(
        [
            "--baseline",
            str(FIXTURES / "baseline_report.json"),
            "--candidate",
            str(FIXTURES / "degraded_report.json"),
            "--warn-threshold",
            "0.03",
            "--critical-threshold",
            "0.08",
            "--out",
            str(out_path),
        ]
    )
    assert exit_code != 0
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["severity"] == "critical"
    known_regressions = set(json.loads((FIXTURES / "known_regressions.json").read_text(encoding="utf-8")))
    found_ids = {r["id"] for r in written["regressions"]}
    assert known_regressions <= found_ids


def test_diff_main_cli_ac3_subprocess_matches_acceptance_sh(tmp_path):
    """Runs the exact CLI invocation checks/acceptance.sh's AC3 step uses,
    as a real subprocess, so a broken `python -m evalkit.diff` entrypoint
    can't hide behind in-process test calls."""
    out_path = tmp_path / "diff_report.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evalkit.diff",
            "--baseline",
            str(FIXTURES / "baseline_report.json"),
            "--candidate",
            str(FIXTURES / "degraded_report.json"),
            "--warn-threshold",
            "0.03",
            "--critical-threshold",
            "0.08",
            "--out",
            str(out_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, result.stderr
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["severity"] == "critical"


def test_diff_main_bootstraps_baseline_when_missing(tmp_path):
    baseline_path = tmp_path / "eval_reports" / "baseline.json"
    candidate_path = FIXTURES / "baseline_report.json"
    exit_code = main(["--baseline", str(baseline_path), "--candidate", str(candidate_path)])
    assert exit_code == 0
    assert baseline_path.exists()
    assert json.loads(baseline_path.read_text(encoding="utf-8")) == json.loads(candidate_path.read_text(encoding="utf-8"))


def test_diff_main_reads_threshold_env_vars_when_no_cli_flag(tmp_path, monkeypatch):
    """WARN_THRESHOLD/CRITICAL_THRESHOLD (.env.example) must actually be
    read when no --warn-threshold/--critical-threshold flag is passed."""
    baseline = _load("baseline_report.json")
    warm = json.loads(json.dumps(baseline))  # deep copy
    warm["pass_rate"] = 0.95  # 5% drop -- "warning" at the 0.03/0.08 defaults
    baseline_path, candidate_path, out_path = tmp_path / "baseline.json", tmp_path / "warm.json", tmp_path / "diff_report.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(warm), encoding="utf-8")

    monkeypatch.setenv("CRITICAL_THRESHOLD", "0.02")  # below the 5% drop -> should escalate to critical via env alone
    exit_code = main(["--baseline", str(baseline_path), "--candidate", str(candidate_path), "--out", str(out_path)])

    assert exit_code != 0
    assert json.loads(out_path.read_text(encoding="utf-8"))["severity"] == "critical"


def test_diff_main_cli_flag_overrides_threshold_env_var(tmp_path, monkeypatch):
    baseline = _load("baseline_report.json")
    warm = json.loads(json.dumps(baseline))
    warm["pass_rate"] = 0.95  # same 5% drop as above
    baseline_path, candidate_path, out_path = tmp_path / "baseline.json", tmp_path / "warm.json", tmp_path / "diff_report.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(warm), encoding="utf-8")

    # Env says "critical" past 2%, but an explicit CLI flag raises the bar to 50% -- CLI must win.
    monkeypatch.setenv("CRITICAL_THRESHOLD", "0.02")
    exit_code = main(
        [
            "--baseline", str(baseline_path), "--candidate", str(candidate_path),
            "--critical-threshold", "0.5", "--out", str(out_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(out_path.read_text(encoding="utf-8"))["severity"] != "critical"


def test_diff_main_html_out(tmp_path):
    html_path = tmp_path / "report.html"
    out_path = tmp_path / "diff_report.json"
    main(
        [
            "--baseline",
            str(FIXTURES / "baseline_report.json"),
            "--candidate",
            str(FIXTURES / "degraded_report.json"),
            "--out",
            str(out_path),
            "--html-out",
            str(html_path),
        ]
    )
    html = html_path.read_text(encoding="utf-8")
    assert "fix-003" in html
    assert "critical" in html
