# llm-regression-gate

CI-style regression testing for LLM prompt changes — every prompt edit is evaluated against a human-verified golden dataset before it can merge.

> Spec'd, directed, and reviewed by me; implemented with an AI engineering team I orchestrate.

**Status:** Work in progress — MVP targeted for early August 2026.

## Why

Most teams ship prompt changes blind: edit a string, deploy, hope. This project treats prompts as versioned artifacts and runs a full evaluation suite (exact-match scoring + LLM-as-judge + latency + token cost) on every pull request that touches `prompts/**`, diffs the results against a baseline, and blocks the merge when a critical regression is detected.

- **Bilingual golden dataset** — Traditional Chinese / English / mixed-language customer support emails, hand-verified labels (never LLM-generated ground truth).
- **Regression diffing** — per-case pass/fail flips, per-category deltas, configurable warning (3%) and critical (8%) thresholds.
- **CI gate** — GitHub Actions posts a scorecard comment on the PR and fails the check on critical regressions.

See [SPEC.md](SPEC.md) for the frozen specification and acceptance criteria, and [docs/DECISIONS.md](docs/DECISIONS.md) for design decisions.

Part of a three-project series on LLM quality engineering: **evaluation (this repo) → retrieval quality (hybrid-rag-engine) → observability (pipeline-lens)**.
