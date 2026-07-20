"""evalkit/report.py: static Jinja2 HTML diff report (SPEC.md §5/§6 base
version -- metadata, scorecard, regressed-case table)."""
from __future__ import annotations

import json
from pathlib import Path

from evalkit.report import render_diff_html

FIXTURES = Path(__file__).parent / "fixtures"


def _diff_report_fixture() -> dict:
    from evalkit.diff import diff_reports

    baseline = json.loads((FIXTURES / "baseline_report.json").read_text(encoding="utf-8"))
    candidate = json.loads((FIXTURES / "degraded_report.json").read_text(encoding="utf-8"))
    return diff_reports(baseline, candidate, warn_threshold=0.03, critical_threshold=0.08)


def test_render_diff_html_contains_scorecard_and_regressions(tmp_path):
    report = _diff_report_fixture()
    out_path = tmp_path / "diff.html"
    html = render_diff_html(report, out_path, run_metadata={"model": "claude-haiku-4-5", "git_commit": "abc1234"})

    assert out_path.exists()
    assert "fix-003" in html
    assert "fix-006" in html
    assert "fix-009" in html
    assert "critical" in html
    assert "claude-haiku-4-5" in html
    assert "abc1234" in html


def test_render_diff_html_no_regressions_message(tmp_path):
    from evalkit.diff import diff_reports

    baseline = json.loads((FIXTURES / "baseline_report.json").read_text(encoding="utf-8"))
    report = diff_reports(baseline, baseline, warn_threshold=0.03, critical_threshold=0.08)
    html = render_diff_html(report, tmp_path / "diff.html")
    assert "No regressions." in html
