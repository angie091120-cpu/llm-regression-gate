"""Ingest the author's case-by-case review of the 70 reference summaries.

Separate from `import_annotations.py` on purpose, and not a subcommand of it.
The two rounds of `docs/PREREGISTRATION.md` section 9 (entry dated 2026-09-19)
are required not to feed back into each other: the category sheet was sealed
and hashed before this review began, and this review is done on a sheet that
carries no category label of any kind. Two modules that cannot import each
other's data is the cheapest way to make that checkable rather than promised.
This one reads `id`, `input_text` and `expected_summary` from the dataset and
never touches `expected_category`, `draft_category` or `labeled_by`.

    python -m experiments.import_summary_review --reviewer A1 --sheet <file>

What section 9 fixed before the counts were known, and what this module
therefore produces: how many cases are marked `ok`, how many `edit`, and the
ids of the cases marked `edit`. Nothing else is derived from this review. No
judge score is obtained again on account of it, the E0 judge-isolation arm and
E2 are not recomputed, and no API call is sent.

Where the replacements go was left to the import pull request. They go in a
file of their own, `summary_replacements_<reviewer>.json`, and not into a new
`golden_dataset.json` field. A summary is the reference every recorded
`judge_score` was measured against; carrying a competing version inside the
frozen instrument invites a later reader to score against the wrong one, and
the dataset's own version bump exists for the category re-labelling. Writing
`expected_summary` itself, and re-running the judge against the result, takes a
further dated entry in section 9 and is a new measurement rather than a
correction to this one.

Validation is fail-whole, like the category importer: every problem is
reported, nothing is written, exit code 1. A row whose `reference_summary` does
not match the dataset is fatal -- a verdict on an edited summary is a verdict
on something other than the case it is filed under.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from experiments.import_annotations import (
    ANNOTATORS,
    BLANKS,
    DATASET,
    REPO_ROOT,
    dataset_version,
    is_handout_text,
    read_sheet,
)

DATA_DIR = REPO_ROOT / "experiments" / "data"
EXPECTED_ROWS = 70
VALID_VERDICTS = ("ok", "edit")
# The sheet Excel produced names the answer column with the options in it.
VERDICT_COLUMNS = ("verdict(ok/edit)", "verdict")
REQUIRED_COLUMNS = ("case_id", "email_body", "reference_summary")
EDIT_COLUMN = "your_edit"
NOTES_COLUMN = "notes"


def default_out(reviewer: str) -> Path:
    return DATA_DIR / f"summary_review_{reviewer.lower()}.json"


def default_replacements_out(reviewer: str) -> Path:
    return DATA_DIR / f"summary_replacements_{reviewer.lower()}.json"


def load_case_fields() -> dict[str, dict[str, str]]:
    """Case id -> {input_text, expected_summary}. Nothing else is read."""
    with DATASET.open(encoding="utf-8") as handle:
        cases = json.load(handle)["cases"]
    return {
        case["id"]: {
            "input_text": case["input_text"],
            "expected_summary": case["expected_summary"],
        }
        for case in cases
    }


def verdict_column(fieldnames: list[str]) -> str | None:
    for name in VERDICT_COLUMNS:
        if name in fieldnames:
            return name
    return None


def validate(
    fieldnames: list[str],
    rows: list[dict[str, str]],
    cases: dict[str, dict[str, str]],
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    """Returns (errors, warnings, records). Records are meaningful only when
    errors is empty."""
    errors: list[str] = []
    warnings: list[str] = []

    verdict_col = verdict_column(fieldnames)
    missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
    if verdict_col is None:
        missing.append("/".join(VERDICT_COLUMNS))
    if missing:
        errors.append(
            f"missing column(s): {', '.join(missing)} "
            f"(found: {', '.join(fieldnames) or 'nothing'})"
        )
        return errors, warnings, []

    if len(rows) != EXPECTED_ROWS:
        errors.append(f"expected {EXPECTED_ROWS} data rows, found {len(rows)}")

    records: list[dict[str, str]] = []
    seen: dict[str, int] = {}

    for index, row in enumerate(rows, start=2):  # line 1 is the header
        case_id = (row.get("case_id") or "").strip(BLANKS)
        raw_verdict = row.get(verdict_col) or ""
        verdict = raw_verdict.strip(BLANKS).lower()
        replacement = (row.get(EDIT_COLUMN) or "").strip(BLANKS)
        notes = (row.get(NOTES_COLUMN) or "").strip(BLANKS)
        body = row.get("email_body") or ""
        summary = row.get("reference_summary") or ""

        if not case_id:
            errors.append(f"line {index}: case_id is blank")
            continue
        if case_id not in cases:
            errors.append(f"line {index}: unknown case_id {case_id!r}")
            continue
        if case_id in seen:
            errors.append(f"line {index}: case_id {case_id} already appeared on line {seen[case_id]}")
            continue
        seen[case_id] = index

        if not verdict:
            errors.append(f"line {index} ({case_id}): {verdict_col} is blank")
        elif verdict not in VALID_VERDICTS:
            errors.append(
                f"line {index} ({case_id}): {verdict_col} {raw_verdict.strip(BLANKS)!r} is not "
                f"one of {'/'.join(VALID_VERDICTS)}"
            )
        elif verdict == "edit" and not replacement:
            errors.append(
                f"line {index} ({case_id}): marked edit with an empty {EDIT_COLUMN}. Section 9 "
                f"says a case marked edit carries a written replacement; an edit with no "
                f"replacement is a verdict this study cannot report"
            )
        elif verdict == "ok" and replacement:
            errors.append(
                f"line {index} ({case_id}): marked ok but {EDIT_COLUMN} is filled in. Two "
                f"readings and this check cannot tell them apart: the verdict is wrong, or the "
                f"replacement was pasted into the wrong row"
            )

        if summary != cases[case_id]["expected_summary"]:
            errors.append(
                f"line {index} ({case_id}): reference_summary does not match "
                f"golden_dataset.json. The verdict on this row is a verdict on some other text"
            )

        if body != cases[case_id]["input_text"] and not is_handout_text(case_id, body):
            errors.append(
                f"line {index} ({case_id}): email_body matches neither golden_dataset.json nor "
                f"the text this case was handed out with"
            )

        records.append({
            "case_id": case_id,
            "verdict": verdict,
            "replacement": replacement,
            "notes": notes,
        })

    absent = sorted(set(cases) - set(seen))
    if absent:
        errors.append(f"{len(absent)} case(s) missing from the sheet: {', '.join(absent)}")

    return errors, warnings, records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and ingest the reference-summary review sheet.",
        allow_abbrev=False,
    )
    parser.add_argument("--reviewer", required=True, choices=sorted(ANNOTATORS),
                        help="who reviewed the summaries (section 9 names A1, the author)")
    parser.add_argument("--sheet", type=Path, required=True, help="returned CSV")
    parser.add_argument("--out", type=Path, default=None,
                        help="output JSON (default: experiments/data/summary_review_<reviewer>.json)")
    parser.add_argument("--replacements-out", type=Path, default=None,
                        help="replacement text (default: experiments/data/summary_replacements_<reviewer>.json)")
    parser.add_argument("--reviewed-on", default=date.today().isoformat(),
                        help="date the reviewer finished, YYYY-MM-DD (default: today)")
    args = parser.parse_args(argv)
    out_path = args.out or default_out(args.reviewer)
    replacements_path = args.replacements_out or default_replacements_out(args.reviewer)

    try:
        date.fromisoformat(args.reviewed_on)
    except ValueError:
        print(f"ERROR: --reviewed-on {args.reviewed_on!r} is not YYYY-MM-DD", file=sys.stderr)
        return 1

    cases = load_case_fields()
    fieldnames, rows = read_sheet(args.sheet)
    errors, warnings, records = validate(fieldnames, rows, cases)
    sheet_sha256 = hashlib.sha256(args.sheet.read_bytes()).hexdigest()

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    if errors:
        print(f"REJECTED {args.sheet}: {len(errors)} problem(s), nothing written.", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    edits = sorted(record["case_id"] for record in records if record["verdict"] == "edit")
    payload = {
        "reviewer_id": args.reviewer,
        "reviewer": ANNOTATORS[args.reviewer]["rater"],
        "reviewed_on": args.reviewed_on,
        "sheet_file": args.sheet.name,
        "sheet_sha256": sheet_sha256,
        "dataset_version_on_disk": dataset_version(),
        "imported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n": len(records),
        "n_ok": sum(1 for record in records if record["verdict"] == "ok"),
        "n_edit": len(edits),
        "edit_case_ids": edits,
        # Per-case verdicts, no replacement text: the counts and the edit ids
        # are what section 9 permits this review to report.
        "verdicts": [
            {"case_id": record["case_id"], "verdict": record["verdict"], "notes": record["notes"]}
            for record in sorted(records, key=lambda r: r["case_id"])
        ],
    }
    replacements = {
        "reviewer_id": args.reviewer,
        "reviewed_on": args.reviewed_on,
        "source_sheet_sha256": sheet_sha256,
        "note": (
            "proposed replacements for expected_summary, held apart from golden_dataset.json. "
            "Nothing in experiments/ scores against these: every recorded judge_score was "
            "measured against the summaries as shipped (docs/PREREGISTRATION.md section 9, "
            "2026-09-19). Writing them into the dataset takes its own dated entry and a judge "
            "re-run."
        ),
        "n_replacements": len(edits),
        "replacements": [
            {"case_id": record["case_id"], "replacement": record["replacement"]}
            for record in sorted(records, key=lambda r: r["case_id"])
            if record["verdict"] == "edit"
        ],
    }

    for path, body in ((out_path, payload), (replacements_path, replacements)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(body, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    print(f"OK: {len(records)} reviewed summaries from {args.reviewer} -> {out_path}")
    print(f"  ok={payload['n_ok']}  edit={payload['n_edit']}")
    print(f"  cases marked edit: {', '.join(edits) if edits else '(none)'}")
    print(f"  replacements -> {replacements_path} ({len(edits)} entr{'y' if len(edits) == 1 else 'ies'})")
    print(f"  sheet sha256: {sheet_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
