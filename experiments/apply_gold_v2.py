"""Write gold v2 into `golden_dataset.json` and move the envelope to v2.0.

Not run by the import pull request. The pull request builds the tool, the
tests and the sensitivity analysis; the dataset itself moves when A2's and A3's
sheets are in, gold v2 is sealed, and `docs/PREREGISTRATION.md` section 9
carries their arrival hashes. Until then this is a dry run by default and
refuses to write an unsealed gold.

    python -m experiments.apply_gold_v2                 # dry run, prints the diff
    python -m experiments.apply_gold_v2 --write         # writes the dataset

What moves, from section 9's "Dataset version" paragraph, and nothing else:

    expected_category   the gold v2 label
    labeled_by          the decision route: majority, adjudicated, fallback_v1.1
    dataset_version     v1.1 -> v2.0

`expected_summary`, `expected_difficulty`, every `draft_*` field, `notes`,
`label_status` and `input_text` are left exactly as they are. `labeled_at` is
also left as it is, and that is a gap rather than a decision: section 9 lists
the three fields above and does not say what a case's `labeled_at` should read
once a September panel has decided a label the entry dated 2026-07-20 records.
This prints that as a warning every run rather than picking a date.

Two consequences this script does not carry out, both recorded in section 9:
`tests/test_golden_dataset.py` asserts the version and moves with it, and the
CI baseline is rebuilt once on v2.0 by one real-API `evalkit` run that gets its
own `docs/DECISIONS.md` entry the way D-011 recorded the v1.1 rebuild.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from experiments.gold_v2 import DEFAULT_OUT as DEFAULT_GOLD_V2
from experiments.import_annotations import DATASET, VALID_LABELS

NEW_DATASET_VERSION = "v2.0"
EXPECTED_FROM_VERSION = "v1.1"
LABELED_AT_WARNING = (
    "labeled_at is left at its current value on every case. Section 9's dataset paragraph "
    "names expected_category, labeled_by and dataset_version and says nothing about the date "
    "a case was labelled, so this script does not invent one. If the dates should move, that "
    "takes its own dated line in section 9"
)


def load_gold(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"ERROR: {path} does not exist. Run `python -m experiments.gold_v2` first; if it "
            f"refused, the reason it printed is the reason the dataset does not move yet"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply gold v2 to golden_dataset.json (dry run unless --write).",
        allow_abbrev=False,
    )
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_V2)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--write", action="store_true", help="actually write the dataset")
    args = parser.parse_args(argv)

    gold = load_gold(args.gold)
    if not gold.get("sealed"):
        pending = gold.get("adjudication_pending_case_ids") or []
        print(
            f"REFUSED: {args.gold} is not sealed -- {len(pending)} case(s) still wait on a "
            f"ruling: {', '.join(pending)}. A half-decided gold is not a label set",
            file=sys.stderr,
        )
        return 1

    labels = {entry["case_id"]: entry for entry in gold["labels"]}
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    on_disk = dataset.get("dataset_version")
    if on_disk == NEW_DATASET_VERSION:
        print(f"Nothing to do: {args.dataset} is already at {NEW_DATASET_VERSION}.")
        return 0
    if on_disk != EXPECTED_FROM_VERSION:
        print(
            f"REFUSED: {args.dataset} is at {on_disk!r}; this step is written for "
            f"{EXPECTED_FROM_VERSION} -> {NEW_DATASET_VERSION}",
            file=sys.stderr,
        )
        return 1

    problems: list[str] = []
    cases = dataset["cases"]
    dataset_ids = {case["id"] for case in cases}
    missing = sorted(dataset_ids - set(labels))
    extra = sorted(set(labels) - dataset_ids)
    if missing:
        problems.append(f"{len(missing)} case(s) have no gold v2 label: {', '.join(missing)}")
    if extra:
        problems.append(f"{len(extra)} gold v2 label(s) name no case: {', '.join(extra)}")
    for case_id, entry in sorted(labels.items()):
        if entry.get("label") not in VALID_LABELS:
            problems.append(f"{case_id}: gold v2 label {entry.get('label')!r} is not a category")
    if problems:
        print(f"REFUSED: {len(problems)} problem(s).", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    changed: list[tuple[str, str, str]] = []
    for case in cases:
        entry = labels[case["id"]]
        if case["expected_category"] != entry["label"]:
            changed.append((case["id"], case["expected_category"], entry["label"]))
        case["expected_category"] = entry["label"]
        case["labeled_by"] = entry["labeled_by"]
    dataset["dataset_version"] = NEW_DATASET_VERSION

    print(f"gold v2 regime: {gold['regime']} ({gold['regime_basis']})")
    print(f"routes: " + ", ".join(f"{route}={n}" for route, n in sorted(Counter(
        entry["labeled_by"] for entry in labels.values()).items())))
    print(f"{len(changed)} of {len(cases)} expected_category values change:")
    for case_id, before, after in changed:
        print(f"  {case_id}: {before} -> {after}")
    print(f"WARNING: {LABELED_AT_WARNING}", file=sys.stderr)

    if not args.write:
        print(f"\nDry run. Nothing written. Re-run with --write to move {args.dataset.name} "
              f"to {NEW_DATASET_VERSION}.")
        return 0

    with args.dataset.open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"\nwrote {args.dataset} at {NEW_DATASET_VERSION}")
    print("Next, and not done here: update the version assertion in "
          "tests/test_golden_dataset.py, and rebuild the CI baseline once with one real-API "
          "evalkit run recorded in docs/DECISIONS.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
