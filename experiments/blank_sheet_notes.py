"""Make the version of a returned sheet that goes into the repository.

A2 and A3 are two people outside the project. What they were promised is
anonymity and a published category answer; nothing told them their free-text
`notes` would be published, so the copy that enters version control has that
column emptied and the original stays with the author
(`docs/PREREGISTRATION.md` section 9, entry dated 2026-09-19 headed "A2's and
A3's sheets are registered under two hashes").

Run it on receipt, before anything is committed, and put both printed hashes in
section 9:

    python -m experiments.blank_sheet_notes --sheet ~/received/a2.csv \\
        --out experiments/data/a2_category_labels_2026-09-23.csv

It does that and nothing else. It does not validate the sheet, does not read
`golden_dataset.json`, and prints no label -- ingestion is
`import_annotations.py`, and it runs after this, on the copy.

What "nothing else touched" means here, exactly. Every value in `case_id`,
`email_body` and `your_label` comes through unchanged, cell for cell, in the
order the rows arrived. The file itself is re-serialised rather than edited in
place: CSV UTF-8 with a BOM, CRLF line endings, minimal quoting, the four
columns in handout order. So a sheet whose spreadsheet quoted a field that did
not need quoting comes back with that field unquoted, and the copy is not
guaranteed byte-identical to the original even where the notes were already
empty -- that it was, for A1, is a measured fact about A1's file and not a
property of this tool.

The column set has to be exactly `case_id`, `email_body`, `your_label`,
`notes`, in that order. A sheet carrying a fifth column is rejected and the
columns are printed. Re-serialising would drop that column silently, and a
tool whose whole contract is "one column changes" cannot be the thing that
quietly removes another.

Because the three other columns are preserved, the copy still reduces to the
handout when `import_annotations.py` blanks the answer columns and rehashes, so
it passes the provenance check; and the residue category the section 9 entries
describe is unchanged in what it contains -- the sheet still carries
`case-007`'s v1 wording, because the email column is untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
from pathlib import Path

from experiments.import_annotations import BLANKS, REQUIRED_COLUMNS, read_sheet

NOTES_COLUMN = "notes"


def blanked_bytes(rows: list[dict[str, str]]) -> bytes:
    """The sheet rewritten with `notes` empty, in the handout's own format."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(REQUIRED_COLUMNS))
    writer.writeheader()
    for row in rows:
        writer.writerow({
            "case_id": row.get("case_id") or "",
            "email_body": row.get("email_body") or "",
            "your_label": row.get("your_label") or "",
            NOTES_COLUMN: "",
        })
    return buffer.getvalue().encode("utf-8-sig")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write the notes-blanked copy of a returned sheet and print both hashes.",
        allow_abbrev=False,
    )
    parser.add_argument("--sheet", type=Path, required=True, help="the file as received")
    parser.add_argument("--out", type=Path, required=True, help="where to write the copy")
    parser.add_argument("--force", action="store_true", help="overwrite an existing --out")
    args = parser.parse_args(argv)

    if args.out.exists() and not args.force:
        print(f"ERROR: {args.out} already exists. Pass --force to replace it", file=sys.stderr)
        return 1

    fieldnames, rows = read_sheet(args.sheet)
    if fieldnames != list(REQUIRED_COLUMNS):
        print(
            f"ERROR: {args.sheet} does not have the handout's columns.\n"
            f"  expected: {', '.join(REQUIRED_COLUMNS)}\n"
            f"  found   : {', '.join(fieldnames) or 'nothing'}\n"
            f"This step re-serialises the file, so any column outside that list would be "
            f"dropped and any reordering would be silently applied. Fix the sheet, or hash "
            f"and file it by hand and say so in PREREGISTRATION section 9",
            file=sys.stderr,
        )
        return 1

    original = args.sheet.read_bytes()
    copy = blanked_bytes(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(copy)

    with_notes = sum(1 for row in rows if (row.get(NOTES_COLUMN) or "").strip(BLANKS))
    original_sha = hashlib.sha256(original).hexdigest()
    copy_sha = hashlib.sha256(copy).hexdigest()
    print(f"rows: {len(rows)}   rows carrying a note: {with_notes}")
    print(f"original sha256      : {original_sha}")
    print(f"notes-blanked sha256 : {copy_sha}")
    print(f"the two are {'identical' if original_sha == copy_sha else 'different'}")
    print(f"wrote {args.out}")
    print("Record both hashes in docs/PREREGISTRATION.md section 9. Keep the original "
          "outside the repository.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
