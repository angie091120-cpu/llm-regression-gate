"""Pytest root conftest.

Makes the flat-layout modules (`llm`, `classifier`, `evalkit`) importable when
pytest is invoked as a bare `pytest -q` — which, unlike `python -m pytest`,
does not put the repo root on sys.path. Keeps AC1's literal command working
identically in local shells and CI.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
