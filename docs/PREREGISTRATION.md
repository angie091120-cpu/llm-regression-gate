# Pre-registration: LLM-as-judge reliability and regression-gate sensitivity

**Written 2026-09-14, before any E1 data existed.** The git history of this
branch is the evidence for that ordering: this file and `experiments/analyze.py`
are committed before the first degraded-prompt run, so the analysis could not
have been chosen after seeing the result.

Only the 10-case smoke run (`experiments/results/raw/smoke/`, 20 calls, cost
calibration only) exists at the time of writing. It is not part of any
hypothesis test and its cases are re-used in the confirmatory runs.

## 1. Question

On Traditional Chinese and code-switched enterprise text, how far does an LLM
judge's verdict agree with human gold labels, how sensitive is that agreement
to the order in which candidates are shown, and how large a quality
degradation can an automated regression gate actually detect at n = 70?

## 2. Dataset

`golden_dataset.json`, 70 cases, all `label_status: confirmed`, labels decided
by one human annotator (the author) in July 2026. Composition: 35 zh-tw,
28 en, 7 mixed; 45 easy, 17 ambiguous, 8 edge. The file is frozen for the
whole study; its sha256 is recorded in every `experiments/results/MANIFEST.json`.

A second independent human annotator labels the same 70 cases (E4). That
annotator sees the email text and the four category definitions only -- not
the existing labels, and not any model output.

## 3. Metrics

- **Primary: `category_match`** -- the classifier's category equals the gold
  category. Binary, one value per (case, repeat). Contains no LLM judge, so
  it is not contaminated by the thing E2 and E4 are measuring.
- **Secondary: `passed`** -- `category_match` AND `judge_score >= 3`, the
  production gate definition (`evalkit/run_eval.py: DEFAULT_PASS_THRESHOLD`).
- Descriptive, never used as an outcome: latency, input/output tokens, cost.

Rationale for the split: the whole point of E2/E4 is that the judge may be
unreliable, so the confirmatory claim must not depend on it. Any result that
holds on `passed` but not on `category_match` is reported as exploratory.

## 4. Confirmatory family

Exactly four tests. Each compares one degraded prompt version against the v1
baseline on the **primary** metric, paired by case, using **exact McNemar**:

| # | Version | Degradation |
|---|---------|-------------|
| H1 | v2a | few-shot examples 4 -> 2 |
| H2 | v2b | few-shot examples 4 -> 0 |
| H3 | v2c | category definitions and the tie-break rule removed |
| H4 | v2d | bilingual / typo handling rules removed |

- Direction: all four are one-directional expectations (degraded <= baseline),
  tested two-sided and reported with the sign of the risk difference.
- Multiplicity: **Holm** across these four p-values, family-wise alpha = 0.05.
- Everything else in this study is **exploratory**: Benjamini-Hochberg within
  each exploratory family, and every exploratory table carries the word
  `exploratory` in its `family` column.

## 5. Design and sample size

| Exp | Design | Calls |
|-----|--------|-------|
| E0 | v1 x 70 cases x 5 repeats, plus a judge-isolation arm (run-1 classifier outputs frozen, judge re-scores 4 more times) | 980 |
| E1 | 4 degraded versions x 70 cases x 3 repeats | 1,680 |
| E2 | pairwise judge (same email, summary A, summary B, tie allowed), order swapped, two judge models | 560 |
| E4 | second human annotator (70), plus Haiku and Sonnet each labelling 70 | 140 |
| E5 | stratified re-analysis of E0/E1 data | 0 |

n = 70 cases is fixed by the dataset and is not increased after looking at any
result. Repeat counts are fixed in advance and are not extended to chase
significance. There is no interim analysis and no optional stopping: E1's
confirmatory tests are computed once, after all 1,680 calls have completed.

Sample size was not chosen by a power calculation -- the dataset existed
first. The detectable-effect question is therefore reported as an outcome
(the E1 power curve), not as a justification.

## 6. Analysis plan

- Proportions: Wilson score interval (primary display) and Clopper-Pearson
  (conservative cross-check), both at 95%.
- Paired comparisons: exact McNemar on discordant pairs; risk difference
  reported as a point estimate. A Newcombe method 10 interval is added only
  after it is validated against the published worked example; until then the
  cell counts (a, b, c, d) are printed so any reader can recompute.
- Repeated measurements of the same case are dependent. Pooled rates carry a
  case-level cluster bootstrap interval (10,000 resamples, seed 20260920).
- Run-to-run variation (E0) is reported as the spread of the five per-run
  rates against the 3% warning / 8% critical thresholds the production gate
  uses, so that "detected a regression" can be read against the noise floor.
- Agreement (E4): Cohen's kappa for each annotator pair, bootstrap CI.
- Position bias (E2): proportion of first-position wins, tested against 0.5
  with an exact binomial test.
- All estimators are implemented in `experiments/stats.py` against the Python
  standard library and are checked against published worked examples and
  against scipy by `python -m experiments.stats --cross-check`.

## 7. Exclusions and failures

- An API failure (429/5xx/timeout/schema violation) produces a row with
  `ok: false` and is excluded from rate denominators. Every table's `n` is the
  number of successful observations; failure counts are reported next to it.
  A failed call is never scored as a wrong answer.
- No case is dropped for being "unrepresentative", and no run is dropped for
  looking bad. If a run has to be discarded for a technical reason, both the
  raw file and the reason stay in the repository.
- Prices are pinned (Haiku 4.5 $1/$5, Sonnet 5 $2/$10 per 1M tokens); the
  runner refuses to start if the effective price table differs.

## 8. What these numbers will not be able to show

1. **Nothing about other datasets.** 70 hand-built cases about a fictional SaaS
   help desk are a deliberately stratified sample, not a random sample of any
   real support queue. No population inference follows.
2. **Nothing about model versions in general.** One classifier (Haiku 4.5) and
   one judge (Sonnet 5) at one point in time. `response.model` returns a dated
   snapshot for Haiku but only the alias for Sonnet, so silent version drift on
   the judge side cannot be fully excluded -- it can only be disclosed.
3. **Nothing about whether the judge is right.** Kappa measures agreement with
   a human label, not correctness. High agreement with a single annotator can
   also mean both share the same blind spot.
4. **No causal claim about real degradations.** The four degraded prompts are
   an intervention chosen by the author; how their effect sizes map to the
   ways a prompt degrades in production is unknown.
5. **No claim about production detection power.** The power curve at n = 30/50
   is simulated from this dataset's dependence structure; simulated and
   measured points are drawn with different markers and are never merged.
6. **A non-significant result is not evidence of no difference.** With n = 70
   and a binary outcome the interval is wide; the CI width is reported instead
   of accepting the null.
7. **No claim that a re-run reproduces these numbers.** LLM sampling cannot be
   seeded. The reproducibility claim is narrower and exact: *raw data
   published, analysis fully reproducible from that raw data* -- the same
   `--seed` on the same JSONL yields byte-identical CSVs
   (`checks/experiments_acceptance.sh` C1).
8. **No claim about production cost or latency.** Figures come from one
   machine, one concurrency setting, no prompt caching and no batch API, at
   2026-09 list prices.

Additionally: E4 has one second annotator, so annotator variance is not
estimable -- the kappa interval reflects sampling of cases only.

## 9. Deviations

Any departure from this document is recorded here, dated, with the reason,
and stays visible in the git history.

**2026-09-14, E0 judge-isolation arm, one call short.** 280 calls planned,
279 succeeded. `case-003` repeat 3 returned a tool input with no `score`
field and failed Pydantic validation, so that case has 3 re-scores instead of
4. Handled by section 7 as written: the row stays in the raw file with
`ok: false`, it is excluded from the denominator rather than scored as a
disagreement, and the run was not repeated -- repeat counts are fixed in
advance and re-rolling a bad draw is exactly what section 5 forbids.
Reason: a model formatting error, not a rate limit or a timeout.

**2026-09-14, descriptive tables added to `analyze.py` after this document
was written.** `rates_run_spread.csv`, `per_case_instability.csv`,
`judge_rescore_stability.csv`, `judge_rescore_summary.csv` and
`figures/noise_floor_vs_gate_thresholds.png` did not exist when section 6 was
committed; they implement the sentence in section 6 that promised E0's
run-to-run variation would be reported against the 3% / 8% thresholds. All of
them are counts, a mean, a sample standard deviation and a range. No
hypothesis test, no interval and no estimator was added, and the confirmatory
family in section 4 is untouched. The threshold values drawn on the figure
are read out of `evalkit/diff.py` at analysis time rather than retyped, so
the annotation cannot go stale.

**2026-09-14, cost accounting.** `runner.cost_for()` records $0 for a failed
call even when the API billed it, so the isolation arm's ledger entry
understates real spend by $0.005338. Cost is descriptive here and is never an
outcome (section 3), so no result changes; the amount is documented in
`experiments/COST_CALIBRATION.md` section 3 and the runner fix is listed as
due before E1.
