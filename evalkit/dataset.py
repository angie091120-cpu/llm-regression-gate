"""Golden dataset loading (SPEC.md §3).

The eval runner must never silently evaluate against an unconfirmed label --
that's exactly the kind of "quietly swallow bad data" failure mode this
project exists to prevent. load_confirmed_cases() filters to
label_status == "confirmed" and *warns loudly* about everything it skips, so
the boss's human-verification step (see golden_dataset.json's `notes`) stays
the single source of truth for what actually gets evaluated.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class GoldenCase(BaseModel):
    id: str
    language: str
    input_text: str
    draft_category: str
    draft_summary: str
    draft_difficulty: Optional[str] = None
    expected_category: Optional[str] = None
    expected_summary: Optional[str] = None
    expected_difficulty: Optional[str] = None
    notes: Optional[str] = None
    label_status: str
    labeled_by: Optional[str] = None
    labeled_at: Optional[str] = None


def load_all_cases(path: str | Path) -> list[GoldenCase]:
    """Load every case in the dataset file, draft or confirmed."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cases_raw = raw["cases"] if isinstance(raw, dict) else raw
    return [GoldenCase(**c) for c in cases_raw]


def load_confirmed_cases(path: str | Path) -> list[GoldenCase]:
    """Load only label_status == 'confirmed' cases, with a loud UserWarning
    listing everything that was excluded (SPEC.md §3).

    Fails loudly (raises) if a case claims to be confirmed but is missing
    expected_category/expected_summary -- that combination means the record
    was hand-edited incorrectly and must never be evaluated as gold.
    """
    all_cases = load_all_cases(path)
    confirmed = [c for c in all_cases if c.label_status == "confirmed"]
    skipped = [c for c in all_cases if c.label_status != "confirmed"]
    if skipped:
        preview = [c.id for c in skipped[:10]]
        suffix = " ... (truncated)" if len(skipped) > 10 else ""
        warnings.warn(
            f"{len(skipped)} of {len(all_cases)} case(s) in {path} are not "
            f"label_status='confirmed' and were excluded from evaluation: {preview}{suffix}",
            stacklevel=2,
        )
    for c in confirmed:
        if not c.expected_category or not c.expected_summary:
            raise ValueError(
                f"case {c.id} is label_status='confirmed' but missing "
                "expected_category/expected_summary -- refusing to evaluate "
                "against an incomplete gold label."
            )
    return confirmed
