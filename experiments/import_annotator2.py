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
something other than the case it is filed under.

One mismatch is expected and is not an error: the sheet was handed out while
the dataset was at v1, and `case-007`'s email text changed in v1.1
(`docs/PREREGISTRATION.md` section 9). `HANDOUT_BODY_SHA256` holds the hash of
the text that case had at handout time, so a row returning it is recognised as
the sheet as sent rather than as an edited row. Which dataset version the
labels belong to is decided by `sheet_provenance` from the file itself --
the handout's own sha256 with the answer columns blanked -- and falls back to
`unknown` rather than to whatever is on disk, because one edited row is enough
to make a v1 sheet look like a v1.1 one. Any other mismatch is still an
error, and the message says which of the two faults it cannot rule out.
`--allow-body-drift` remains the escape hatch for a mismatch this file has not
been told about.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
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

# The dataset version `experiments/data/annotator2_sheet.csv` was built from,
# and the sha256 of every email text that has moved since. Hashes rather than
# the strings themselves: the v1 text of case-007 is the third-party address
# this repository took out (PREREGISTRATION section 9) and it is not
# reintroduced here. A returned row matching one of these is the handout, not
# an edit.
SHEET_DATASET_VERSION = "v1"
HANDOUT_BODY_SHA256 = {
    "case-007": "8b20b51d6f57bb239a4d0a2ade6e148028a7a5dddc95da9f5fa936e43b20a79d",
}
# sha256 of `experiments/data/annotator2_sheet.csv` as sent, answer columns
# empty (experiments/README.md records the same value). A returned sheet is
# this file with `your_label` and `notes` filled in, so it is recognised by
# blanking those two columns and rehashing rather than by hashing what comes
# back. Issuing a new sheet means updating these three constants together.
HANDOUT_SHEET_SHA256 = "c7abaf6fe77dc204549f31853ef348c1b370240737a0b1a55e4b758ea2a462ba"
UNKNOWN_VERSION = "unknown"
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


def is_handout_text(case_id: str, body: str) -> bool:
    """True when this row carries the text the sheet was handed out with.

    Distinguishes "the dataset moved after the handout", which is a documented
    fact about this study, from "somebody edited the spreadsheet", which
    invalidates the row.
    """
    expected = HANDOUT_BODY_SHA256.get(case_id)
    if expected is None:
        return False
    return hashlib.sha256(body.encode("utf-8")).hexdigest() == expected


def blanked_sheet_sha256(rows: list[dict[str, str]]) -> str:
    """sha256 of these rows rewritten with the answer columns empty.

    Reproduces the handout byte for byte when the returned file is the handout
    with `your_label` and `notes` filled in, because it rebuilds the CSV the
    way the handout was written (utf-8-sig, CRLF, minimal quoting, the four
    columns in order). A spreadsheet that re-quotes fields on save will not
    match; that costs an attribution, never a wrong one.
    """
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(REQUIRED_COLUMNS))
    writer.writeheader()
    for row in rows:
        writer.writerow({
            "case_id": row.get("case_id") or "",
            "email_body": row.get("email_body") or "",
            "your_label": "",
            "notes": "",
        })
    return hashlib.sha256(buffer.getvalue().encode("utf-8-sig")).hexdigest()


def sheet_provenance(
    raw_sha256: str,
    rows: list[dict[str, str]],
    handout_era_rows: list[str],
) -> tuple[str, str, str | None]:
    """Returns (dataset version the sheet is keyed to, how that was decided,
    warning or None).

    Decided from the file first and from row contents second, because only one
    case differs between v1 and v1.1: a v1 sheet with that one row pasted over
    is indistinguishable from a v1.1 sheet by rows alone, and answering `v1.1`
    there would be a guess wearing a fact's clothes. That case returns
    `unknown`.
    """
    on_disk = dataset_version()
    if raw_sha256 == HANDOUT_SHEET_SHA256:
        return SHEET_DATASET_VERSION, "the file is byte-identical to the handout", None
    if blanked_sheet_sha256(rows) == HANDOUT_SHEET_SHA256:
        return (
            SHEET_DATASET_VERSION,
            "identical to the handout once your_label and notes are blanked",
            None,
        )
    if handout_era_rows:
        return (
            SHEET_DATASET_VERSION,
            "carries text that exists only in dataset "
            f"{SHEET_DATASET_VERSION}: {', '.join(sorted(handout_era_rows))}",
            "this file is not the handout with only the answer columns filled -- a "
            "case_id or an email_body was changed as well. The labels are still "
            f"attributed to dataset {SHEET_DATASET_VERSION}, on the rows that say so",
        )
    if on_disk != SHEET_DATASET_VERSION:
        return (
            UNKNOWN_VERSION,
            "neither the file nor any row identifies a dataset version",
            f"cannot tell which dataset this sheet was filled in against: it is not the "
            f"handout, and no row carries text unique to {SHEET_DATASET_VERSION}. "
            f"Recorded as {UNKNOWN_VERSION} rather than assuming the {on_disk} on disk",
        )
    return on_disk, "the dataset has not moved since the handout", None


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
) -> tuple[list[str], list[str], list[dict[str, str]], list[str]]:
    """Returns (errors, warnings, normalised records, handout-era case ids).

    Records are only meaningful when errors is empty. The last value lists the
    rows carrying text that only exists in the dataset version the sheet was
    handed out at; `sheet_provenance` turns it into a version.
    """
    errors: list[str] = []
    warnings: list[str] = []
    handout_era_rows: list[str] = []

    missing_columns = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
    if missing_columns:
        errors.append(
            f"missing column(s): {', '.join(missing_columns)} "
            f"(found: {', '.join(fieldnames) or 'nothing'})"
        )
        return errors, warnings, [], []

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
            if is_handout_text(case_id, body):
                handout_era_rows.append(case_id)
                warnings.append(
                    f"line {index} ({case_id}): email_body is this case's text in dataset "
                    f"{SHEET_DATASET_VERSION}, which is what the sheet was handed out with; "
                    f"golden_dataset.json has moved on since (PREREGISTRATION section 9). "
                    f"The row was not edited, so it is accepted."
                )
            elif allow_body_drift:
                warnings.append(
                    f"line {index} ({case_id}): email_body matches neither golden_dataset.json "
                    f"nor the handout (allowed by --allow-body-drift)"
                )
            else:
                errors.append(
                    f"line {index} ({case_id}): email_body matches neither golden_dataset.json "
                    f"nor the text this case was handed out with. Two different faults and this "
                    f"check cannot tell them apart: the row was edited in the spreadsheet, or "
                    f"the dataset changed after the handout and HANDOUT_BODY_SHA256 was not "
                    f"updated. Compare the row against the sheet that was sent before reaching "
                    f"for --allow-body-drift"
                )

        records.append({"case_id": case_id, "label": label, "notes": notes})

    absent = sorted(set(case_texts) - set(seen))
    if absent:
        errors.append(f"{len(absent)} case(s) missing from the sheet: {', '.join(absent)}")

    if normalised:
        warnings.append("normalised label spelling on " + str(len(normalised)) + " row(s): " + "; ".join(normalised))

    return errors, warnings, records, handout_era_rows


def build_payload(
    records: list[dict[str, str]],
    sheet: Path,
    sheet_sha256: str,
    annotated_on: str,
    sheet_dataset_version: str,
    sheet_dataset_version_basis: str,
) -> dict[str, object]:
    return {
        "annotator": ANNOTATOR,
        "annotated_on": annotated_on,
        "sheet_seed": SHEET_SEED,
        "sheet_file": sheet.name,
        "sheet_sha256": sheet_sha256,
        "sheet_dataset_version": sheet_dataset_version,
        "sheet_dataset_version_basis": sheet_dataset_version_basis,
        "dataset_version_on_disk": dataset_version(),
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
    errors, warnings, records, handout_era_rows = validate(
        fieldnames, rows, case_texts, args.allow_body_drift
    )
    raw_sha256 = hashlib.sha256(args.sheet.read_bytes()).hexdigest()
    sheet_dataset_version, version_basis, version_warning = sheet_provenance(
        raw_sha256, rows, handout_era_rows
    )
    if version_warning:
        warnings.append(version_warning)

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    if errors:
        print(f"REJECTED {args.sheet}: {len(errors)} problem(s), nothing written.", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    sheet_sha256 = raw_sha256
    payload = build_payload(
        records, args.sheet, sheet_sha256, args.annotated_on,
        sheet_dataset_version, version_basis,
    )
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
    print(f"  sheet is keyed to dataset {sheet_dataset_version} "
          f"(on disk: {dataset_version()}) -- {version_basis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
