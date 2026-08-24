# SPEC — llm-regression-gate (frozen)

Regression-detection system for an LLM-powered feature: a customer-support email classifier for a fictional SaaS ("NebulaDesk"). When a prompt changes, the system re-evaluates it against a human-verified golden dataset, diffs the results against the baseline, and gates the merge in CI.

This spec is frozen. Implementation that deviates from the acceptance criteria below is treated as a blocking review finding.

## 1. Feature under test

`classify_email(email_text, prompt_config) -> ClassificationResult`

- Categories: `billing | technical | account | general`, plus a one-sentence summary.
- Model: `claude-haiku-4-5` via a provider-agnostic `llm.py` interface (default Anthropic; switchable to OpenAI through environment variables).
- Structured output enforced via tool-use + Pydantic — no regex-parsing of free text.

## 2. Prompt versioning

Prompts live in `/prompts`, one YAML file per version: `version`, `model`, `created_at`, `system_prompt`, `few_shot_examples`, `output_schema`. The prompt file is the unit of change the CI gate protects.

## 3. Golden dataset

- 60–80 cases. Language mix: ~50% Traditional Chinese, ~40% English, ~10% mixed.
- Deliberate edge cases: typos/zhuyin artifacts, mixed currencies, sarcasm, ambiguous two-category emails, very short emails; each case tagged `expected_difficulty` (`easy | ambiguous | edge`).
- **Labels are human-verified.** Drafted text and suggested labels may be machine-generated, but `expected_category` / `expected_summary` are only written after human review; `label_status: confirmed` is required for a case to enter evaluation. The eval runner skips (and warns about) unconfirmed cases.
- Dataset is versioned via an internal `dataset_version` field in `golden_dataset.json` (the filename stays stable — it is part of the AC2 contract); adding confirmed cases bumps the minor version. *(Revised 2026-07-20: an earlier draft named the file `golden_dataset_v1.json`, contradicting AC2; the AC wins — see docs/DECISIONS.md D-001.)*

## 4. Evaluation engine

`run_eval(prompt_config, golden_dataset) -> eval_report.json`, async-batched. Per case:

| Dimension | Definition |
|---|---|
| `category_match` | exact match against `expected_category` (binary) |
| `judge_score` | 1–5, `claude-sonnet-5` as judge, scoring the produced summary against the expected one |
| `latency_ms` | wall-clock per request |
| `tokens` | input + output tokens |

Case **pass** = `category_match` AND `judge_score >= 3` (threshold configurable). `pass_rate` = passes / total.

## 5. Regression diffing

Compare current run vs. baseline (`eval_reports/baseline.json`, the last successful run on `main`; first run bootstraps the baseline and exits 0):

- overall `pass_rate` delta, per-category accuracy delta,
- flip lists: pass→fail (regressions), fail→pass (improvements),
- thresholds: `abs(delta) > 3%` → warning (exit 0, flagged), `abs(delta) > 8%` → critical (**exit non-zero**). Both configurable.

An HTML diff report (Jinja2, static) renders run metadata, a scorecard vs. baseline, and a side-by-side table of regressed cases.

## 6. CI gate

- `test.yml` — on every push/PR: `pytest -q`, fully mocked, no API key required; plus secret scanning (gitleaks).
- `eval-gate.yml` — on PRs touching `prompts/**`: runs `pytest -q` (mocked) plus an offline regression-diff smoke test against pre-generated fixtures, and posts a scorecard comment on the PR — no API key required. On `workflow_dispatch`: runs the real-API eval, diffs it against the committed baseline, and fails the check on critical regressions. Requires the repo's branch protection to mark the workflow as a required status check for a `workflow_dispatch` run to actually gate a merge. *(Revised 2026-08-24: an earlier draft of this bullet described both triggers as unconditionally running the real-API eval; that conflicted with the cost-control bullet below and made the PR-triggered gate unrunnable without a provisioned `ANTHROPIC_API_KEY` repo secret — see docs/DECISIONS.md D-006.)*
- Cost control: regression-detection logic is exercised for free on every `prompts/**` PR via pre-generated fixture reports (`test.yml` and the `eval-gate.yml` PR smoke test); real-API runs are reserved for manually-dispatched gate runs and milestones.

## 7. Acceptance criteria

| # | Command | Assertable outcome |
|---|---|---|
| AC1 | `pytest -q` with no `ANTHROPIC_API_KEY` set | exit 0; all LLM calls mocked/fixtured |
| AC2 | `python -m evalkit.run_eval --prompt-version v1 --dataset golden_dataset.json --out eval_report.json` (real API) | report exists; `pass_rate` ∈ [0,1]; `cases` ≥ 60; each case has `category_match`/`judge_score`/`latency_ms`/`tokens` |
| AC3 | `python -m evalkit.diff --baseline <fixture> --candidate <degraded-fixture> --warn-threshold 0.03 --critical-threshold 0.08` | exit non-zero; `regressions` contains the 3 planted known-regression case ids; `severity: critical` |
| AC4 | open a test PR touching `prompts/**` | `eval-gate` workflow green; PR comment contains pass rate, delta vs. baseline, severity |
| AC5 | `docker build -t llm-regression-gate:test .` then `docker run --rm llm-regression-gate:test pytest -q` | both exit 0 |

`checks/acceptance.sh` runs the machine-checkable subset in one shot.

## 8. Non-goals (v1)

No Slack alerts (PR comments instead), no dashboard/Streamlit (static HTML), no rolling-average drift detection, no third-party eval frameworks (a small self-built `evalkit/` is the point), no statistical significance testing, no multi-model cost routing.
