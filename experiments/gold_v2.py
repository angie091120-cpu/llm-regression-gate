"""Produce gold v2 from the returned annotator sheets, by the rules only.

`docs/PREREGISTRATION.md` section 9 (entry dated 2026-09-19) fixed every rule
in here before any sheet was compared with anything, and this module is that
entry turned into code and nothing more. Where the entry is silent, this
module refuses and names the cases rather than choosing: a rule invented after
the sheets are open is exactly what registering the protocol was meant to stop.
Those refusals are `ProtocolGap` and they exit 4.

    python -m experiments.gold_v2 --as-of 2026-09-27
    python -m experiments.gold_v2 --as-of 2026-09-27 --rulings <author_rulings.csv>

Exit codes, because "wrote nothing" has two very different meanings here:

    0  gold v2 written (sealed if no case is waiting on a ruling)
    1  bad input -- a file that will not parse, a ruling for a case not queued
    3  no gold v2 by rule: fallback 2, or the deadline has not passed yet
    4  a situation section 9 does not cover; the case ids are printed

What it reads: `experiments/data/annotations/*_labels.json`, written by
`import_annotations.py`, and `golden_dataset.json`. What it never puts in front
of the author: the adjudication sheet carries the email text and that case's
human labels, and not `draft_category`, not the 2026-07-20 rulings in `notes`,
not the shipped v1.1 label, and no model output from any arm of this study.

The rules, in the order they are applied:

* All three sheets in. Three valid votes: the majority decides (`majority`);
  all three different goes to the author (`adjudicated`). Two valid votes --
  that sheet omitted the case, or its entry failed validation and no
  correction came back -- two agreeing decide it, two differing go to the same
  author on the same restricted sheet. Fewer than two valid votes is wider
  than the rule covers, and is a `ProtocolGap`.
* Fallback 1, in force if by the deadline the returns are A1 plus exactly one
  of A2 and A3. Where those two humans agree the case is final (`majority`);
  where they split, the shipped v1.1 label is a third vote and breaks the tie
  (`fallback_v1.1`); where A1, the other human and v1.1 are all different the
  author rules (`adjudicated`). A case left with only A1's vote keeps its v1.1
  label (`fallback_v1.1`). A case left with only the *other* human's vote is
  not a case section 9 names, and is a `ProtocolGap`.
* Fallback 2, if by the deadline only A1 has returned: gold stays at v1.1 and
  A1's sheet is a reliability check used for nothing else. Nothing is written.
* A sheet that arrives after a fallback has been applied does not reopen gold
  v2. If the output file already records a fallback and more annotators are
  present now, this refuses to overwrite it.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

from experiments.import_annotations import (
    ANNOTATION_DIR,
    ANNOTATORS,
    BLANKS,
    DATASET,
    REPO_ROOT,
    VALID_LABELS,
    dataset_version,
)

DATA_DIR = REPO_ROOT / "experiments" / "data"
DEFAULT_OUT = DATA_DIR / "gold_v2.json"
DEFAULT_QUEUE = DATA_DIR / "adjudication_queue.csv"
# Section 9: "if by 2026-09-27 the returns are A1 plus exactly one of A2 and A3".
RETURN_DEADLINE = date(2026, 9, 27)
ROSTER = ("A1", "A2", "A3")
ROUTE_MAJORITY = "majority"
ROUTE_ADJUDICATED = "adjudicated"
ROUTE_FALLBACK = "fallback_v1.1"
REGIME_MAIN = "main"
REGIME_FALLBACK_1 = "fallback_1"
REGIME_FALLBACK_2 = "fallback_2"
QUEUE_HEADER_FIXED = ("case_id", "email_body")
RULING_COLUMNS = ("case_id", "ruling")


class ProtocolGap(Exception):
    """A case the section 9 entry does not decide. Never resolved in code."""


def load_annotations(annotation_dir: Path) -> dict[str, dict]:
    """annotator id -> the payload `import_annotations.py` wrote."""
    found: dict[str, dict] = {}
    for annotator in ROSTER:
        path = annotation_dir / f"{annotator.lower()}_labels.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        recorded = payload.get("annotator_id")
        if recorded != annotator:
            raise SystemExit(
                f"ERROR: {path} says annotator_id={recorded!r} but is filed under {annotator}"
            )
        found[annotator] = payload
    return found


def labels_by_case(payload: dict) -> dict[str, str]:
    return {record["case_id"]: record["label"] for record in payload["labels"]}


def load_shipped() -> tuple[dict[str, str], dict[str, str]]:
    """(case id -> v1.1 expected_category, case id -> input_text).

    The shipped label is read for one purpose only: fallback 1's third vote.
    It never reaches the adjudication sheet.
    """
    with DATASET.open(encoding="utf-8") as handle:
        cases = json.load(handle)["cases"]
    return (
        {case["id"]: case["expected_category"] for case in cases},
        {case["id"]: case["input_text"] for case in cases},
    )


def decide_regime(present: list[str], as_of: date) -> tuple[str, str]:
    """Returns (regime, why). Raises ProtocolGap for rosters section 9 does not
    name, and SystemExit(3) through the caller for "not yet"."""
    present = sorted(present)
    if present == list(ROSTER):
        return REGIME_MAIN, "all three sheets have arrived"
    if "A1" not in present:
        raise ProtocolGap(
            "A1 has not returned. Section 9's fallbacks are both written around A1's sheet "
            f"being present (returns so far: {', '.join(present) or 'none'}), so which rule "
            "applies is not fixed anywhere"
        )
    if as_of < RETURN_DEADLINE:
        return (
            "",
            f"only {', '.join(present)} of {', '.join(ROSTER)} have returned and the deadline "
            f"{RETURN_DEADLINE.isoformat()} has not passed ({as_of.isoformat()}). A fallback is "
            f"chosen at the deadline and not before -- section 9 fixes both fallbacks in advance "
            f"so that neither is picked after a disagreement rate is known",
        )
    if len(present) == 2:
        other = [a for a in present if a != "A1"][0]
        return REGIME_FALLBACK_1, f"fallback 1: the returns at the deadline are A1 and {other}"
    return REGIME_FALLBACK_2, "fallback 2: only A1 returned by the deadline"


def decide_case(
    case_id: str,
    votes: dict[str, str],
    regime: str,
    shipped_label: str,
) -> tuple[str | None, str, bool]:
    """Returns (label or None while it waits on a ruling, route, needs_ruling)."""
    voters = sorted(votes)
    values = [votes[a] for a in voters]

    if regime == REGIME_MAIN:
        if len(values) == 3:
            counts = Counter(values)
            winner, top = counts.most_common(1)[0]
            if top >= 2:
                return winner, ROUTE_MAJORITY, False
            return None, ROUTE_ADJUDICATED, True
        if len(values) == 2:
            if values[0] == values[1]:
                return values[0], ROUTE_MAJORITY, False
            return None, ROUTE_ADJUDICATED, True
        raise ProtocolGap(
            f"{case_id}: {len(values)} valid vote(s) with all three sheets in. Section 9's "
            f"two-vote rule 'covers per-case gaps and nothing wider'"
        )

    # Fallback 1.
    if len(values) == 2:
        if values[0] == values[1]:
            return values[0], ROUTE_MAJORITY, False
        if shipped_label in values:
            return shipped_label, ROUTE_FALLBACK, False
        return None, ROUTE_ADJUDICATED, True
    if voters == ["A1"]:
        return shipped_label, ROUTE_FALLBACK, False
    raise ProtocolGap(
        f"{case_id}: under fallback 1 the only valid vote is {voters[0] if voters else 'none'}. "
        f"Section 9 names the case where 'fallback 1 leaves a case with only A1's vote valid' "
        f"and no other"
    )


def build_decisions(
    annotations: dict[str, dict],
    regime: str,
    shipped: dict[str, str],
) -> tuple[list[dict], list[str]]:
    """Returns (per-case decisions, ProtocolGap messages)."""
    per_annotator = {annotator: labels_by_case(payload) for annotator, payload in annotations.items()}
    decisions: list[dict] = []
    gaps: list[str] = []
    for case_id in sorted(shipped):
        votes = {
            annotator: labels[case_id]
            for annotator, labels in per_annotator.items()
            if case_id in labels
        }
        try:
            label, route, needs_ruling = decide_case(case_id, votes, regime, shipped[case_id])
        except ProtocolGap as gap:
            gaps.append(str(gap))
            continue
        decisions.append({
            "case_id": case_id,
            "label": label,
            "labeled_by": route,
            "votes": dict(sorted(votes.items())),
            "needs_ruling": needs_ruling,
        })
    return decisions, gaps


def write_adjudication_queue(
    path: Path,
    decisions: list[dict],
    texts: dict[str, str],
    annotators_present: list[str],
) -> int:
    """The restricted sheet. Columns: the case, its email, one column per human
    who labelled it. Nothing else -- see the module docstring and section 9."""
    pending = [d for d in decisions if d["needs_ruling"]]
    header = [*QUEUE_HEADER_FIXED, *(f"label_{a}" for a in annotators_present)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for decision in pending:
            writer.writerow([
                decision["case_id"],
                texts[decision["case_id"]],
                *(decision["votes"].get(a, "") for a in annotators_present),
            ])
    return len(pending)


def read_rulings(path: Path, queued: set[str]) -> dict[str, str]:
    text = path.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(text.splitlines(True))
    fieldnames = list(reader.fieldnames or [])
    missing = [name for name in RULING_COLUMNS if name not in fieldnames]
    if missing:
        raise SystemExit(
            f"ERROR: {path} is missing column(s): {', '.join(missing)} "
            f"(found: {', '.join(fieldnames) or 'nothing'})"
        )
    rulings: dict[str, str] = {}
    errors: list[str] = []
    for index, row in enumerate(reader, start=2):
        case_id = (row.get("case_id") or "").strip(BLANKS)
        ruling = (row.get("ruling") or "").strip(BLANKS).lower()
        if not case_id:
            errors.append(f"line {index}: case_id is blank")
            continue
        if case_id not in queued:
            errors.append(f"line {index}: {case_id} is not on the adjudication queue")
            continue
        if case_id in rulings:
            errors.append(f"line {index}: {case_id} appears twice")
            continue
        if ruling not in VALID_LABELS:
            errors.append(f"line {index} ({case_id}): ruling {ruling!r} is not one of {'/'.join(VALID_LABELS)}")
            continue
        rulings[case_id] = ruling
    if errors:
        print(f"REJECTED {path}: {len(errors)} problem(s), nothing written.", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        raise SystemExit(1)
    return rulings


def refuse_reopen(out_path: Path, present: list[str]) -> None:
    """Section 9: a sheet that arrives after a fallback has been applied is one
    more reliability check against the gold that exists."""
    if not out_path.exists():
        return
    try:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    regime = existing.get("regime")
    already = set(existing.get("annotators_present") or [])
    if regime in (REGIME_FALLBACK_1, REGIME_FALLBACK_2) and set(present) - already:
        late = ", ".join(sorted(set(present) - already))
        raise SystemExit(
            f"REFUSED: {out_path} already records gold v2 under {regime} from "
            f"{', '.join(sorted(already))}, and {late} has arrived since. Section 9: a sheet "
            f"that arrives after a fallback has been applied is reported as one more "
            f"reliability check against the gold that exists, and does not reopen gold v2. "
            f"Report it as a reliability check; do not regenerate this file"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Turn the returned annotator sheets into gold v2, by the section 9 rules.",
        allow_abbrev=False,
    )
    parser.add_argument("--as-of", default=date.today().isoformat(),
                        help="date the regime is decided on, YYYY-MM-DD (default: today)")
    parser.add_argument("--annotation-dir", type=Path, default=ANNOTATION_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--queue-out", type=Path, default=DEFAULT_QUEUE,
                        help="where the restricted adjudication sheet is written")
    parser.add_argument("--rulings", type=Path, default=None,
                        help="the author's rulings, CSV with columns case_id,ruling")
    args = parser.parse_args(argv)

    try:
        as_of = date.fromisoformat(args.as_of)
    except ValueError:
        print(f"ERROR: --as-of {args.as_of!r} is not YYYY-MM-DD", file=sys.stderr)
        return 1

    annotations = load_annotations(args.annotation_dir)
    present = sorted(annotations)
    if not present:
        print(f"ERROR: no annotator files under {args.annotation_dir}", file=sys.stderr)
        return 1

    try:
        regime, why = decide_regime(present, as_of)
    except ProtocolGap as gap:
        print(f"PROTOCOL GAP: {gap}", file=sys.stderr)
        print("Nothing written. This takes a dated line in PREREGISTRATION section 9, not a "
              "decision here.", file=sys.stderr)
        return 4

    if not regime:
        print(f"NOT YET: {why}")
        print(f"Returned so far: {', '.join(present)}. Nothing written.")
        return 3

    if regime == REGIME_FALLBACK_2:
        print(f"FALLBACK 2: {why}")
        print("Section 9: gold stays at v1.1 and A1's sheet is reported as a reliability check "
              "and used for nothing else. No gold v2 written.")
        return 3

    refuse_reopen(args.out, present)
    shipped, texts = load_shipped()
    decisions, gaps = build_decisions(annotations, regime, shipped)
    if gaps:
        print(f"PROTOCOL GAP: {len(gaps)} case(s) section 9 does not decide. Nothing written.",
              file=sys.stderr)
        for gap in gaps:
            print(f"  - {gap}", file=sys.stderr)
        return 4

    pending = write_adjudication_queue(args.queue_out, decisions, texts, present)
    queued = {d["case_id"] for d in decisions if d["needs_ruling"]}
    rulings: dict[str, str] = {}
    if args.rulings is not None:
        rulings = read_rulings(args.rulings, queued)
        for decision in decisions:
            if decision["case_id"] in rulings:
                decision["label"] = rulings[decision["case_id"]]
    unresolved = sorted(d["case_id"] for d in decisions if d["label"] is None)

    payload = {
        "gold_version": "v2",
        "regime": regime,
        "regime_basis": why,
        "as_of": as_of.isoformat(),
        "return_deadline": RETURN_DEADLINE.isoformat(),
        "annotators_present": present,
        "annotator_display_names": {a: ANNOTATORS[a]["rater"] for a in present},
        "annotator_sheet_sha256": {a: annotations[a]["sheet_sha256"] for a in present},
        "annotator_sheet_dataset_version": {
            a: annotations[a]["sheet_dataset_version"] for a in present
        },
        "dataset_version_on_disk": dataset_version(),
        "protocol": "docs/PREREGISTRATION.md section 9, entry dated 2026-09-19",
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n": len(decisions),
        "n_adjudicated_pending": len(unresolved),
        "adjudication_pending_case_ids": unresolved,
        "adjudicated_case_ids": sorted(d["case_id"] for d in decisions if d["needs_ruling"]),
        "route_counts": dict(sorted(Counter(d["labeled_by"] for d in decisions).items())),
        "sealed": not unresolved,
        "labels": decisions,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"{regime}: {why}")
    print(f"  {len(decisions)} cases -> {args.out}")
    for route, count in payload["route_counts"].items():
        print(f"    {route}: {count}")
    print(f"  adjudication queue: {pending} case(s) -> {args.queue_out}")
    if unresolved:
        print(f"  NOT SEALED: {len(unresolved)} case(s) still waiting on a ruling: "
              f"{', '.join(unresolved)}")
    else:
        print("  sealed: every case has a label")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
