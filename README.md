# llm-regression-gate

CI-style regression testing for LLM prompt changes — every prompt edit is evaluated against a human-verified golden dataset on the pull request that changes it.

> Spec'd, directed, and reviewed by me; implemented with an AI engineering team I orchestrate.

**Status:** MVP built — 114 tests pass (fully mocked, no API key needed): 46 cover the tool itself and 68 the experiment tooling under `experiments/`, and the acceptance script in [`checks/acceptance.sh`](checks/acceptance.sh) runs the machine-checkable criteria from SPEC.md §7. The regression baseline (`eval_reports/baseline.json`) is a real-API run of `prompts/v1` over all 70 confirmed cases: rebuilt on dataset v2.0 from the run of 2026-09-22 16:24 UTC (2026-09-23 00:24 +08:00), pass rate 0.8714 (61/70). The two runs it replaced are kept and read by nothing — `eval_reports/baseline-2026-09-17-dataset-v1.1.json` (dataset v1.1, 63/70) and `eval_reports/baseline-2026-09-14-dataset-v1.json` (dataset v1, 64/70). [docs/DECISIONS.md D-014](docs/DECISIONS.md) says why the baseline follows the dataset version and which six cases changed verdict between the two newest; D-011 does the same for the rebuild before it. See *CI gate* below.

## Why

Most teams ship prompt changes blind: edit a string, deploy, hope. This project treats prompts as versioned artifacts and runs a full evaluation suite (exact-match scoring + LLM-as-judge + latency + token cost) against a human-verified golden dataset, diffs the results against a baseline, and fails the check with a scorecard comment on the PR when a critical regression is detected.

- **Bilingual golden dataset** — Traditional Chinese / English / mixed-language customer support emails, human-verified labels (SPEC.md §3).
- **Regression diffing** — per-case pass/fail flips, per-category deltas, configurable warning (3%) and critical (8%) thresholds.
- **CI gate** — every PR touching `prompts/**`, and manual `workflow_dispatch` runs, is designed to evaluate the prompt against the real API, diff the result against the committed baseline, post a scorecard comment on the PR, and fail the check on a critical regression. See [SPEC.md §6](SPEC.md#6-ci-gate).

  **Today this check does not block a merge.** Making it one that does requires all of the following:

  - ~~an `ANTHROPIC_API_KEY` repository secret is provisioned~~ — done 2026-09-14; before that, `eval-gate.yml` failed loud on every run with an explicit "ANTHROPIC_API_KEY secret not configured" error rather than passing silently;
  - ~~a committed `eval_reports/baseline.json` exists~~ — done 2026-09-14 (SPEC.md §5, [docs/DECISIONS.md D-009](docs/DECISIONS.md)); runs now diff against it instead of re-bootstrapping;
  - the repository's branch protection lists this check as required — still not the case, and now a deliberate choice rather than a pending step; the paragraph below says why.

  The secret and the baseline are both in place, and the scorecard comment is no longer hypothetical: [PR #4](https://github.com/angie091120-cpu/llm-regression-gate/pull/4) gets one from `eval-gate.yml` on every push, each after a real-API run of `prompts/v1` against the dataset on that commit. Runs on dataset v1 came back 64/70 with a delta of 0; runs on dataset v1.1 against that v1 baseline came back 63/70. Only one of them was a local run with a per-case report, and it names what moved — `case-056`, a W-9 request the classifier answered `billing` in every earlier observation and `general` there, with the judge scoring it 5 either way. It is not `case-007`, the case the v1.1 privacy fix edited; that one answers `account` and passes in both. The baseline is now rebuilt on v1.1 (D-011), so the gate is again diffing prompts rather than prompts plus a dataset edit. The gate posts a fresh scorecard on every push, so read the newest comment rather than any figure or count quoted here. `main` is protected: strict mode (a branch must be up to date before merging), force pushes and branch deletion refused, and two required status checks — `pytest` and `gitleaks`, both jobs of `test.yml`, which runs on every push and pull request. Administrators are exempt from the rule (`enforce_admins` is off) and this repository's only maintainer is an administrator, so those two required checks do not compel the account that merges; `allow_force_pushes` and `allow_deletions` are off in the rule either way. The `eval` job is deliberately not one of them: `eval-gate.yml` is filtered on `paths: prompts/**`, and a required check that a pull request never triggers sits at Expected forever, which would leave every pull request touching no prompt unmergeable. A critical regression therefore turns the `eval` check red on a pull request that edits `prompts/**` and does not by itself block the merge.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # includes requirements.txt
pytest -q                             # fully mocked, no API key needed
```

`checks/acceptance.sh` runs on `python3` from `PATH` unless told otherwise, so
activate `.venv` first or name an interpreter -- `EXP_PYTHON=$HOME/.venvs/lrg-exp/bin/python bash checks/acceptance.sh`.
On a bare system `python3` the dependencies are missing and AC1 fails on the
import rather than on a test, which is a wrong interpreter and not a failing
suite; the script prints the interpreter it used on its first line.

Running the real-API paths (`python -m evalkit.run_eval`, `python -m evalkit.judge`)
needs `ANTHROPIC_API_KEY` in the environment. Copy `.env.example` to `.env` and fill
it in locally, or export the variables directly. Key ones:

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | -- | required for real-API runs; not read at all by `pytest -q` |
| `LLM_PROVIDER` | `anthropic` | set to `openai` to switch providers (needs `OPENAI_API_KEY` + `pip install openai`) |
| `LLM_CLASSIFIER_MODEL` | `claude-haiku-4-5` (or `gpt-4o-mini` on OpenAI) | model for the classifier tier |
| `LLM_JUDGE_MODEL` | `claude-sonnet-5` (or `gpt-4o` on OpenAI) | model for the LLM-as-judge tier |
| `EVAL_CONCURRENCY` | `5` | max concurrent requests during `run_eval` |

Full list in [`.env.example`](.env.example).

**Golden dataset status:** `golden_dataset.json` ships with all 70 cases at
`label_status: confirmed`. The labels are human-verified as SPEC.md §3 defines
it: drafted text and suggested labels may be machine-generated, the
`expected_category` and `expected_summary` fields are written only after human
review, and `label_status: confirmed` is required before a case enters
evaluation. `evalkit.run_eval` only ever evaluates `label_status: confirmed`
cases and warns loudly about (and refuses to run against) anything that reverts
to draft or is missing an expected label (SPEC.md §3). The cases are fictional:
every company, person, domain and email address in them is invented, and any
resemblance to a real one is coincidence. The file has changed twice since it
was frozen. v1.1 (2026-09-17) moved `case-007`'s email address to a domain
reserved for documentation, the one it carried having turned out to belong to a
real company; reason, scope and both sha256 are in `docs/PREREGISTRATION.md`
section 9, and no label, no category and no other case moved. v2.0
(2026-09-22) is the relabelling: three people labelled the same 70 emails
blind, each case takes the majority of the three, the author ruled on the one
case all three split, and five of the 70 `expected_category` values changed
([PR #11](https://github.com/angie091120-cpu/llm-regression-gate/pull/11),
[docs/DECISIONS.md D-013](docs/DECISIONS.md)). The pre-registered tables in
`experiments/` stay the confirmatory result on the v1.1 labels; the versions
recomputed on v2.0 sit beside them as a sensitivity analysis.

See [SPEC.md](SPEC.md) for the frozen specification and acceptance criteria, and [docs/DECISIONS.md](docs/DECISIONS.md) for design decisions.

Part of a three-project series on LLM quality engineering: **evaluation (this repo) → retrieval quality (hybrid-rag-engine) → observability (pipeline-lens)**.
