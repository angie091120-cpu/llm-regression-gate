# Raw data

One JSONL line per API call, written by `experiments/runner.py`,
`runner_pairwise.py` and `runner_annotator.py` and never edited afterwards. A
raw file is evidence here, so nothing in this tree is back-filled, re-run or
scrubbed; `experiments/results/MANIFEST.json` records the line counts and
`checks/experiments_acceptance.sh` C2 checks them against the files.

That rule has one visible consequence. Six of these files -- `e0_noise`, the
four `e1_v2*` arms and the 2026-09-16 `e2_pairwise` batch -- still contain,
inside model outputs that quoted the email they were handed, the `case-007`
address `golden_dataset.json` replaced when it moved to v1.1 on 2026-09-17: an
invented account holder at a domain that turned out to belong to a real
company. The address is not reprinted here and the dataset no longer carries
it. Why these files are deliberately left alone, what the scope of the residue
is, and which published artifact did move instead, are in the
`docs/PREREGISTRATION.md` section 9 entry headed "one email address in
`golden_dataset.json` belonged to a real company; the dataset moves to v1.1".
