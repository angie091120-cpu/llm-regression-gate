"""evalkit -- a small, self-built LLM regression-eval toolkit.

Deliberately dependency-light (stdlib + pydantic + jinja2, no RAGAS/DeepEval;
see SPEC.md §5 non-goals) so the whole folder can be vendored into another
project unchanged except for the one import in run_eval.py that points at
the feature-under-test (see run_eval.py's module docstring). dataset.py,
judge.py, cost.py, diff.py and report.py have no dependency on the
classifier and are reusable as-is.
"""
