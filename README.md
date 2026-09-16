# llm-regression-gate

CI-style regression testing for LLM prompt changes — every prompt edit is evaluated against a human-verified golden dataset on the pull request that changes it.

> Spec'd, directed, and reviewed by me; implemented with an AI engineering team I orchestrate.

**Status:** MVP built — 46 tests pass (fully mocked, no API key needed) and the acceptance script in [`checks/acceptance.sh`](checks/acceptance.sh) runs the machine-checkable criteria from SPEC.md §7. The regression baseline (`eval_reports/baseline.json`) was bootstrapped on 2026-09-14 from a real-API run of `prompts/v1` over all 70 confirmed cases (pass rate 0.9143), so the gate now has a fixed reference to diff against — see *CI gate* below.

## Why

Most teams ship prompt changes blind: edit a string, deploy, hope. This project treats prompts as versioned artifacts and runs a full evaluation suite (exact-match scoring + LLM-as-judge + latency + token cost) against a human-verified golden dataset, diffs the results against a baseline, and fails the check with a scorecard comment on the PR when a critical regression is detected.

- **Bilingual golden dataset** — Traditional Chinese / English / mixed-language customer support emails, hand-verified labels (never LLM-generated ground truth).
- **Regression diffing** — per-case pass/fail flips, per-category deltas, configurable warning (3%) and critical (8%) thresholds.
- **CI gate** — every PR touching `prompts/**`, and manual `workflow_dispatch` runs, is designed to evaluate the prompt against the real API, diff the result against the committed baseline, post a scorecard comment on the PR, and fail the check on a critical regression. See [SPEC.md §6](SPEC.md#6-ci-gate).

  **Today this check does not block a merge.** Making it one that does requires all of the following:

  - ~~an `ANTHROPIC_API_KEY` repository secret is provisioned~~ — done 2026-09-14; before that, `eval-gate.yml` failed loud on every run with an explicit "ANTHROPIC_API_KEY secret not configured" error rather than passing silently;
  - ~~a committed `eval_reports/baseline.json` exists~~ — done 2026-09-14 (SPEC.md §5, [docs/DECISIONS.md D-009](docs/DECISIONS.md)); runs now diff against it instead of re-bootstrapping;
  - the repository's branch protection lists this check as required — a repository-owner setting, not configured yet.

  Today it only turns the check red; the scorecard comment is posted once the secret and baseline are in place.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # includes requirements.txt
pytest -q                             # fully mocked, no API key needed
```

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
`label_status: confirmed` -- every `expected_category`/`expected_summary`/
`expected_difficulty` was written by a human reviewer, not the agent-drafted
`draft_category`/`draft_summary`/`draft_difficulty` fields. `evalkit.run_eval`
only ever evaluates `label_status: confirmed` cases and warns loudly about
(and refuses to run against) anything that reverts to draft or is missing an
expected label (SPEC.md §3).

See [SPEC.md](SPEC.md) for the frozen specification and acceptance criteria, and [docs/DECISIONS.md](docs/DECISIONS.md) for design decisions.

Part of a three-project series on LLM quality engineering: **evaluation (this repo) → retrieval quality (hybrid-rag-engine) → observability (pipeline-lens)**.
