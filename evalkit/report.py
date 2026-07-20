"""Static Jinja2 HTML diff report (SPEC.md §5/§6). Base version only:
metadata, scorecard, regressed-case table. Trend charts and visual polish
are explicitly cut-able (SPEC.md §3.3) and not implemented here.
"""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATES_DIR = Path(__file__).parent / "templates"


def render_diff_html(diff_report: dict, out_path: str | Path, run_metadata: dict | None = None) -> str:
    """Render diff_report (the dict produced by evalkit.diff.diff_reports) to
    a static HTML file at out_path. Returns the rendered HTML string too,
    mainly so tests can assert on content without re-reading the file."""
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), autoescape=select_autoescape(["html"]))
    template = env.get_template("diff_report.html.j2")
    html = template.render(report=diff_report, meta=run_metadata or {})
    Path(out_path).write_text(html, encoding="utf-8")
    return html
