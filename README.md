# llm-regression-gate

CI-style regression testing for LLM prompt changes — every prompt edit is evaluated against a human-verified golden dataset before it can merge.

> Spec'd, directed, and reviewed by me; implemented with an AI engineering team I orchestrate.

**Status:** Work in progress — MVP targeted for early August 2026.

## Why

Most teams ship prompt changes blind: edit a string, deploy, hope. This project treats prompts as versioned artifacts and runs a full evaluation suite (exact-match scoring + LLM-as-judge + latency + token cost) on every pull request that touches `prompts/**`, diffs the results against a baseline, and blocks the merge when a critical regression is detected.

- **Bilingual golden dataset** — Traditional Chinese / English / mixed-language customer support emails, hand-verified labels (never LLM-generated ground truth).
- **Regression diffing** — per-case pass/fail flips, per-category deltas, configurable warning (3%) and critical (8%) thresholds.
- **CI gate** — GitHub Actions posts a scorecard comment on the PR and fails the check on critical regressions.

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

**Golden dataset status:** `golden_dataset.json` currently ships as a **draft** --
70 cases with agent-drafted `draft_category`/`draft_summary`/`draft_difficulty`,
all `label_status: draft`. The eval engine only evaluates `label_status: confirmed`
cases (and warns loudly about everything it skips), so `evalkit.run_eval` will
correctly refuse to run against an empty confirmed set until a human reviewer
fills in `expected_category`/`expected_summary`/`expected_difficulty` and flips
`label_status` to `confirmed` per case (SPEC.md §3.1(c)).

See [SPEC.md](SPEC.md) for the frozen specification and acceptance criteria, and [docs/DECISIONS.md](docs/DECISIONS.md) for design decisions.

Part of a three-project series on LLM quality engineering: **evaluation (this repo) → retrieval quality (hybrid-rag-engine) → observability (pipeline-lens)**.
