# llm-regression-gate

CI-style regression testing for LLM prompt changes — every prompt edit is evaluated against a human-verified golden dataset before it can merge.

> Spec'd, directed, and reviewed by me; implemented with an AI engineering team I orchestrate.

**Status:** Work in progress — MVP targeted for early August 2026.

## Why

Most teams ship prompt changes blind: edit a string, deploy, hope. This project treats prompts as versioned artifacts and runs a full evaluation suite (exact-match scoring + LLM-as-judge + latency + token cost) against a human-verified golden dataset, diffs the results against a baseline, and can block the merge when a critical regression is detected.

- **Bilingual golden dataset** — Traditional Chinese / English / mixed-language customer support emails, hand-verified labels (never LLM-generated ground truth).
- **Regression diffing** — per-case pass/fail flips, per-category deltas, configurable warning (3%) and critical (8%) thresholds.
- **CI gate** — every PR touching `prompts/**`, and manual `workflow_dispatch` runs, evaluate the prompt against the real API, diff the result against the committed baseline, post a scorecard comment on the PR, and fail the check on a critical regression. See [SPEC.md §6](SPEC.md#6-ci-gate).

  **Current status: not yet enforcing.** The repo has no `ANTHROPIC_API_KEY` secret configured, so `eval-gate.yml` fails loud (explicit "ANTHROPIC_API_KEY secret not configured" error, not a silent pass) on every run until that secret is provisioned. Branch protection also isn't yet set to require this check. Neither has been done as part of this fix — provisioning the secret is a key-management step, out of scope here. Separately, `eval_reports/baseline.json` has never been committed, so once the secret is in place, the gate will keep *bootstrapping* the baseline (SPEC.md §5) instead of actually diffing until someone runs a real eval once and commits that file — see [docs/DECISIONS.md D-009](docs/DECISIONS.md).

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
