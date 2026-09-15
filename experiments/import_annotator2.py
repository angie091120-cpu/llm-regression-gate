"""Ingest the second annotator's returned sheet for E4 (label reliability).

The sheet handed out (`experiments/data/annotator2_sheet.csv`) carries the case
id and the email text and nothing else: no gold category, no model output, no
difficulty or language stratum. This script is the other half of that
arrangement. It validates the returned file and writes
`experiments/data/annotator2_labels.json`; it never reads, prints or compares a
gold label. Agreement against the gold labels and against the model annotators
belongs to the analysis step, after ingestion, so that nothing here can leak
back into a re-labelling round.

It reports every problem it finds rather than stopping at the first, because
the annotator is doing a favour and a second round trip costs more than a long
error message. On any problem nothing is written and the exit code is 1.

    python -m experiments.import_annotator2 --sheet ~/Downloads/annotator2_sheet.csv
    python -m experiments.import_annotator2 --sheet <file> --annotated-on 2026-09-23

`your_label` is accepted with surrounding whitespace and in any capitalisation
(`Billing ` -> `billing`); every other deviation is an error. The email text is
compared against `golden_dataset.json` and a mismatch is fatal, because a row
whose text was edited in the spreadsheet is a row whose label describes
something other than the case it is filed under. `--allow-body-drift` downgrades
that one check to a warning.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "golden_dataset.json"
DEFAULT_SHEET = REPO_ROOT / "experiments" / "data" / "annotator2_sheet.csv"
DEFAULT_OUT = REPO_ROOT / "experiments" / "data" / "annotator2_labels.json"

ANNOTATOR = "human-2 (non-member)"
SHEET_SEED = 20260916
EXPECTED_ROWS = 70
VALID_LABELS = ("billing", "technical", "account", "general")
REQUIRED_COLUMNS = ("case_id", "email_body", "your_label", "notes")
BLANKS = " \t　​"


def load_case_texts() -> dict[str, str]:
    """Case id -> email text, from the frozen dataset.

    Reads `id` and `input_text` only. `expected_category`, `expected_summary`,
    `expected_difficulty`, `language` and the draft fields are never touched by
    this module -- see the module docstring.
    """
    with DATASET.open(encoding="utf-8") as handle:
        cases = json.load(handle)["cases"]
    return {case["id"]: case["input_text"] for case in cases}


def dataset_version() -> str:
    with DATASET.open(encoding="utf-8") as handle:
        return str(json.load(handle).get("dataset_version", "unknown"))


def read_sheet(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Returns (fieldnames, rows). utf-8-sig so a BOM from Excel is dropped."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise SystemExit(f"ERROR: sheet not found: {path}")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SystemExit(
            f"ERROR: {path} is not UTF-8 ({exc}). Re-save it from Excel as "
            f'"CSV UTF-8 (Comma delimited)" and run this again.'
        )
    reader = csv.DictReader(text.splitlines(True))
    fieldnames = list(reader.fieldnames or [])
    rows = [row for row in reader]
    return fieldnames, rows


def validate(
    fieldnames: list[str],
    rows: list[dict[str, str]],
    case_texts: dict[str, str],
    allow_body_drift: bool,
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    """Returns (errors, warnings, normalised records). Records are only
    meaningful when errors is empty."""
    errors: list[str] = []
    warnings: list[str] = []

    missing_columns = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
    if missing_columns:
        errors.append(
            f"missing column(s): {', '.join(missing_columns)} "
            f"(found: {', '.join(fieldnames) or 'nothing'})"
        )
        return errors, warnings, []

    extra = [name for name in fieldnames if name not in REQUIRED_COLUMNS]
    if extra:
        warnings.append(f"ignoring extra column(s): {', '.join(extra)}")

    if len(rows) != EXPECTED_ROWS:
        errors.append(f"expected {EXPECTED_ROWS} data rows, found {len(rows)}")

    records: list[dict[str, str]] = []
    seen: dict[str, int] = {}
    normalised: list[str] = []

    for index, row in enumerate(rows, start=2):  # line 1 is the header
        case_id = (row.get("case_id") or "").strip(BLANKS)
        raw_label = row.get("your_label") or ""
        label = raw_label.strip(BLANKS).lower()
        notes = (row.get("notes") or "").strip(BLANKS)
        body = row.get("email_body") or ""

        if not case_id:
            errors.append(f"line {index}: case_id is blank")
            continue
        if case_id not in case_texts:
            errors.append(f"line {index}: unknown case_id {case_id!r}")
            continue
        if case_id in seen:
            errors.append(f"line {index}: case_id {case_id} already appeared on line {seen[case_id]}")
            continue
        seen[case_id] = index

        if not label:
            errors.append(f"line {index} ({case_id}): your_label is blank")
        elif label not in VALID_LABELS:
            errors.append(
                f"line {index} ({case_id}): your_label {raw_label.strip(BLANKS)!r} is not one of "
                f"{'/'.join(VALID_LABELS)}"
            )
        elif label != raw_label:
            normalised.append(f"line {index} ({case_id}): {raw_label!r} -> {label!r}")

        if body != case_texts[case_id]:
            message = f"line {index} ({case_id}): email_body differs from golden_dataset.json"
            if allow_body_drift:
                warnings.append(message + " (allowed by --allow-body-drift)")
            else:
                errors.append(message + " -- the row was edited; rerun with --allow-body-drift only if that edit was expected")

        records.append({"case_id": case_id, "label": label, "notes": notes})

    absent = sorted(set(case_texts) - set(seen))
    if absent:
        errors.append(f"{len(absent)} case(s) missing from the sheet: {', '.join(absent)}")

    if normalised:
        warnings.append("normalised label spelling on " + str(len(normalised)) + " row(s): " + "; ".join(normalised))

    return errors, warnings, records


def build_payload(
    records: list[dict[str, str]],
    sheet: Path,
    sheet_sha256: str,
    annotated_on: str,
) -> dict[str, object]:
    return {
        "annotator": ANNOTATOR,
        "annotated_on": annotated_on,
        "sheet_seed": SHEET_SEED,
        "sheet_file": sheet.name,
        "sheet_sha256": sheet_sha256,
        "dataset_version": dataset_version(),
        "n": len(records),
        "imported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "labels": sorted(records, key=lambda record: record["case_id"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and ingest the E4 second annotator's CSV.",
    )
    parser.add_argument("--sheet", type=Path, default=DEFAULT_SHEET, help="returned CSV (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output JSON (default: %(default)s)")
    parser.add_argument(
        "--annotated-on",
        default=date.today().isoformat(),
        help="date the annotator finished, YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--allow-body-drift",
        action="store_true",
        help="downgrade an edited email_body from an error to a warning",
    )
    args = parser.parse_args(argv)

    try:
        date.fromisoformat(args.annotated_on)
    except ValueError:
        print(f"ERROR: --annotated-on {args.annotated_on!r} is not YYYY-MM-DD", file=sys.stderr)
        return 1

    case_texts = load_case_texts()
    fieldnames, rows = read_sheet(args.sheet)
    errors, warnings, records = validate(fieldnames, rows, case_texts, args.allow_body_drift)

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    if errors:
        print(f"REJECTED {args.sheet}: {len(errors)} problem(s), nothing written.", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    sheet_sha256 = hashlib.sha256(args.sheet.read_bytes()).hexdigest()
    payload = build_payload(records, args.sheet, sheet_sha256, args.annotated_on)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    counts = {label: sum(1 for record in records if record["label"] == label) for label in VALID_LABELS}
    with_notes = sum(1 for record in records if record["notes"])
    print(f"OK: {len(records)} labels -> {args.out}")
    print("  annotator's own distribution: " + ", ".join(f"{label}={counts[label]}" for label in VALID_LABELS))
    print(f"  rows carrying a note: {with_notes}")
    print(f"  sheet sha256: {sheet_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
