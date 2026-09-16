# Pre-registration: LLM-as-judge reliability and regression-gate sensitivity

Status: FROZEN 2026-09-15 23:24 (+0800) by the author's decision, recorded in the office decision log. Any later change is an entry in section 9, dated; the body above section 9 is not edited after this stamp.

**Written 2026-09-14, before any E1 data existed.** The git history of this
branch is the evidence for that ordering: this file and `experiments/analyze.py`
are committed before the first degraded-prompt run, so the analysis could not
have been chosen after seeing the result.

**Last substantive revision 2026-09-14**, adding section 10 (E1 in full) and
the deviations dated that day in section 9. Still before the first E1 call.
Once this document is signed off, changes go in section 9 and nowhere else.

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

Section 10 carries the rest of E1: the file names, what "paired by case" means
when a case is measured three times, which baseline run the pairs come from,
and how the power curve is produced. `docs/DEGRADATION_DESIGN.md` carries the
diffs and the predictions.

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
  reported as a point estimate with a Newcombe method 10 interval. The
  validation that interval actually got is not the one this section originally
  promised -- see the 2026-09-14 entry in section 9, and the `note` column that
  travels with every published interval. The cell counts (a, b, c, d) are
  printed either way, so any reader can recompute.
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

**2026-09-15, what "repeat 3" means in the entry above.** `repeat 3` is a
zero-based repeat index: `repeat_idx` takes the values 0, 1, 2 and 3 in
`experiments/results/raw/e0_judge_iso/`, 70 rows each, so the failed call is
the fourth re-score of `case-003` rather than the third. Clarification only --
no number, no rule and no raw row changes.

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

**2026-09-14, cost accounting, closed.** `runner.cost_for()` now charges any
call that reported token usage, failed or not, and `analyze.py` recomputes
every published cost from the raw `usage` objects at the pinned prices instead
of summing the `cost_usd` field the runner wrote. The E0 raw files are not
back-filled -- a raw file is evidence -- so MANIFEST.json now carries both
totals and their difference, and the $0.005338 appears in the recomputed one.
`experiments/COST_CALIBRATION.md` section 3 states which number is which.

**2026-09-14, Newcombe method 10 shipped with a different validation than
section 6 promised.** Section 6 said the interval would be added "only after it
is validated against the published worked example" in Newcombe (1998). The
paper was not reachable from the machine this was written on, and E1 needs the
interval, so the estimator ships validated three other ways instead: the
formula is written out line by line in the `newcombe_paired_diff_ci`
docstring; `python -m experiments.stats` asserts its structural invariants
(symmetry when b == c, containment of the point estimate, the continuity
correction widening rather than narrowing, degenerate tables staying inside
[-1, 1]); and `python -m experiments.stats --coverage` runs a Monte-Carlo
coverage study over five paired-outcome distributions at n = 30 and n = 70.
Measured on 2026-09-14 at 20,000 replicates per cell: coverage of the shipped
form (continuity correction on) ranged 0.9520 to 0.9941 against a nominal
0.95, and the uncorrected form dipped to 0.9341, which is why the corrected
form is the default. The check against the paper's printed example remains
open and is listed in `experiments/README.md`. Every interval this function
produces carries `experiments.stats.NEWCOMBE_VALIDATION` in its own table row
and figure footnote, so the caveat cannot be separated from the number.

**2026-09-14, confirmatory unit of analysis stated explicitly.** Section 4 says
the four tests are "paired by case", and `analyze.py` was pairing every
(case, repeat) observation, which would have treated 210 dependent
observations as 210 independent pairs and made the exact McNemar p-value
anticonservative. Section 10.4 now fixes the rule the tests actually use --
one value per case, repeats collapsed by majority vote -- and the
(case, repeat) version stays in `paired_mcnemar.csv` labelled
`sensitivity_per_repeat_pairing`, with no interval and no multiplicity
adjustment. No data existed for either version when this was written.

**2026-09-14, one evaluation case is inside the baseline prompt.** Few-shot
example 4 of `prompts/v1.yaml` is case-043 verbatim, label included. Found
while building the degraded prompts, after E0 had run. The dataset is frozen
(section 2) so nothing is edited; instead section 10.10 pre-specifies the
sensitivity analysis (H1 and H2 recomputed with case-043 dropped) and
`docs/DEGRADATION_DESIGN.md` states the consequence for v2a and v2b, which
remove that example along with the degradation being tested.

**2026-09-15, Newcombe method 10 now checked against the paper's printed
example; the 2026-09-14 deviation above is closed.** The open item that entry
left -- and that `experiments/README.md` listed as due before any submitted
document quoted the interval -- was the check against Newcombe (1998)
itself. Table III of that paper was located and transcribed on 2026-09-15:
e=20, f=12, g=2, h=16, n=50, theta-hat = (f - g)/n = 0.2000, printed 95%
interval 0.0562 to 0.3292 for method 10 and 0.0618 to 0.3242 for the
uncorrected method 8. `newcombe_paired_diff_ci` returns 0.056156 to 0.329207
and, with `continuity=False`, 0.061805 to 0.324162 -- both agree with the
paper at every digit it prints, so the continuity correction is pinned as
well as the formula. Nothing was changed to reach this agreement: the diff on
`experiments/stats.py` removes no line of arithmetic, only comment and
docstring text. The edits are the example constant, the assertions over it,
the rewritten validation notes, and this entry.

Three assertions now run in `python -m experiments.stats` (28/28 on
2026-09-15) at a tolerance of 1e-4, the precision the paper prints: method 10
against the printed interval, `continuity=False` against the printed method 8
row, and a guard that transposing f and g negates the interval and therefore
fails the printed endpoints. The last one exists because the paper's e/f/g/h
and this codebase's (a, b, c, d) are the same 2x2 table in a different order
(a=e, b=g, c=f, d=h, documented on `NEWCOMBE_TABLE3_EXAMPLE`); a transposition
there would leave every interval the right width and the wrong sign, which no
structural invariant catches. Both negative controls were run before this was
written: transposing the mapping, and moving a printed constant by 2e-4, each
turn the self-check red.

Two notes on what this check is not. It does not supersede the Monte-Carlo
coverage study -- one table at one alpha cannot show coverage, so `--coverage`
stays as the evidence for that. And R's `PropCIs` and `DescTools` were not
used as the reference: their paired-proportion functions cite Agresti & Min
(2005) and Tango (1998), and `DescTools::BinomDiffCI(method="scorecc")` cites
Newcombe's *independent*-samples paper, Stat Med 17(8):873-890, not this one.
The source is the printed paper.

`experiments.stats.NEWCOMBE_VALIDATION`, which travels in the `note` column of
every row and figure footnote that carries a Newcombe interval, has been
updated to state the new status. Section 6 and section 10.7 point at this
section for validation status and so need no edit; the caveat they refer to
is now the paragraph above rather than the 2026-09-14 one.

**2026-09-15, section 10.10.1 had no implementation when this document was
frozen.** `analyze.py` shipped the pairing-unit view of section 10.10.2 --
`paired_mcnemar.csv`, family `sensitivity_per_repeat_pairing` -- but nothing
in the code dropped case-043 and recomputed H1 and H2. The gap was found while
the E1 calls were still being made and closed before `analyze.py` had been run
over any E1 raw file: at that point the only E1 data read were row counts, the
success and failure counts, `response_model` and latency, none of which is an
outcome. The rule implemented is this document's sentence unchanged -- H1 and
H2 recomputed with case-043 dropped, n = 69, reported next to the primary rows
-- and it calls the same `mcnemar_exact` and `newcombe_paired_diff_ci` the
confirmatory rows call. No new arithmetic was added. The new path was checked
first on a synthetic pairing in which dropping the marked case is known to
move b from 2 to 1.

Three choices this document did not fix, recorded here so that they are not
read as pre-registered: the sensitivity rows sit in `e1_main.csv` and are told
apart by the `family` column (`sensitivity_drop_leaked_case`); they carry no
adjusted p-value, because Holm is fixed across the four confirmatory tests and
enlarging that family would change the confirmatory result a sensitivity
analysis is supposed to leave alone; and both E1 figures filter on the
confirmatory family, so `e1_forest.png` and `e1_power.png` show the four lines
they would have shown without this change. This is the first entry written
after the freeze stamp at the head of the document.

**2026-09-16, E5's implementation choices, none of which this document
fixed.** Section 5 gives E5 one line ("stratified re-analysis of E0/E1 data")
and section 6 gives it the Wilson interval; everything else was decided while
writing `table_e5_strata` and `table_e5_logit`, after the E0 and E1 data
existed, and is therefore recorded here rather than read as pre-registered.
Five choices. **Unit:** one outcome per case, repeats collapsed by majority
vote -- section 10.4's rule, reused, so a stratum's n is a count of emails
(mixed = 7) and not of calls (mixed = 35). **Arms:** the arms are named in
`analyze.py` (`e0_noise` v1, `e1_v2a..d`) rather than globbed, because the raw
directory also holds a 10-case smoke run tagged v1 and a judge-isolation arm
with no classifier call. **Cross-stratum test:** each stratum against the rest
of its own arm, Fisher exact, one p-value per row -- not all pairwise, which
would put three p-values on a three-level kind and none on the row. **BH
families:** two, the baseline arm's ten strata and the four degraded arms'
forty, named in the `family` column; pooling them would dilute the baseline
rows, and every row is labelled exploratory either way. **Stratum membership
notes** are derived from the case ids at analysis time, so the observation
that the seven `mixed` cases are the consecutive block case-064..case-070 --
one quota fill, not a draw spread across the dataset -- travels in the table
instead of in a document that can go stale.

**2026-09-16, the E5 logistic model is not fitted on the arm it was specified
for.** `pass ~ language + difficulty + category` with case-clustered standard
errors cannot be estimated on the baseline arm: `language[mixed]` is 35/35 and
`category[account]` is 80/80 there, so those coefficients are unbounded and
the fit would return a large number with a large standard error and no
meaning. Firth's penalised likelihood is the standard remedy; no validated
implementation is available here (statsmodels 0.15.0 has none, `firthlogist`
is not installed), and this package does not ship an unvalidated one, so the
row stays in `e5_logit.csv` marked `fitted = no` with the separated levels
named, and the Firth sensitivity is a second row marked not run. What is
reported instead is the same formula fitted on every arm that ran, E0 plus the
four E1 arms (1,190 observations, 70 clusters), where no level has a constant
outcome -- plus a sensitivity fit that adds a prompt-version term, because
pooling five prompts into a model with no version term is a real
mis-specification and not one to hide in a footnote. The pooled fit is a
different quantity from the one section 5 implied, and nothing written from it
may be described as the baseline system's stratum effects.

**2026-09-16, E2 ran 94 of 560 calls and stopped: the API account hit its
spend limit.** The run began 17:30:48 UTC and every call after 17:32:01 UTC
came back `400 invalid_request_error`, body: "You have reached your specified
API usage limits. You will regain access on 2026-10-01 at 00:00 UTC." All 560
rows are in `experiments/results/raw/e2_pairwise/`, 466 of them with
`ok: false`, the error text and the request id -- section 7 as written: a
failed call is a row, never a wrong answer, and the file stays in the
repository. It was a billing limit on the account, not a rate limit, a schema
violation or a bug in the runner; the 16-call `e2_smoke` calibration run three
minutes earlier succeeded 16/16 on both judge models, and the four cells that
returned nothing are the ones that had not been reached yet.

What survives is one cell of four, partially: 24 of 70 easy-layer pairs judged
in both orders by `claude-sonnet-5`, and zero pairs for `claude-haiku-4-5` in
either layer. The tables report it with the coverage rows first
(`e2_consistency.csv`, `row_type = coverage`: 0/140, 0/140, 94/140, 0/140) and
`analyze.py` refuses to draw `figures/e2_consistency.png` from part of the
design, printing the missing cells instead. **E2's pre-registered questions are
unanswered.** 24 pairs is not the 70 the design calls for, one judge model is
not two, the easy layer alone cannot compare easy against hard, and no number
in those two CSVs is a result of this study. Re-running the 560 calls needs
account access, which returns 2026-10-01 unless the limit is raised; the design
is unchanged and no part of it was altered in response to the failure.

**2026-09-16, E2 was completed in a second batch: 466 calls, 21 hours after the
first 94.** The entry above records the run that stopped. This one records how
it was finished and what the gap costs the reading of the numbers.

`experiments/runner_pairwise.py` gained `--resume-failed-from`, which re-sends
only the design cells -- (judge model, layer, order, pair) -- that an earlier
run of the same experiment recorded as failures. 466 of them, and no part of
the design changed: the pairs are rebuilt from the same frozen E0 and E1 raw
files, the judge prompt is the same file with the same sha256
(`judge_pairwise_v1`, hashed into MANIFEST.json), and the runner refuses to
start if the resumed file carries a different `exp_id` or a different prompt
version, or if any cell would be judged against different summaries than the
failed call used -- the source call ids on the old row are compared against the
ones about to be sent rather than assumed equal. Cells that already succeeded
are subtracted, so the re-run cannot produce a second answer to a question that
already has one.

Batch 1: 2026-09-15 17:30:48 to 17:32:45 UTC, 94 answered calls, all
`claude-sonnet-5` on the easy layer. Batch 2: 2026-09-16 14:52:11 to 14:58:52
UTC, 466 answered calls, none failed, covering the rest of all four cells.
Every E2 number therefore comes from two sessions 21 hours apart. Only the
`claude-sonnet-5` easy cell mixes them, 94 calls against 46; the other three
cells are batch 2 alone.

What the gap does not allow this study to exclude is judge-side drift. Section
8.2 already fixes that `response.model` comes back as the bare alias for
Sonnet, so "the same judge answered both batches" is an assumption and not a
fact in the raw data. Haiku's dated snapshot is recorded per row and reads
`claude-haiku-4-5-20251001` on every call of batch 2, and the Haiku arm ran
inside batch 2 in any case. The batches are separable from the raw data
(`run_id`, `timestamp_utc`) and are deliberately not compared as a test: 94
against 46 in the one cell that broke, chosen after seeing where it broke, is
not a hypothesis this document registered.

**Duplicate cells, and the rule that resolves them.** A cell can now appear
twice in the raw data -- refused in batch 1, answered in batch 2.
`analyze.py`'s `e2_cell_rows` takes the answered row and leaves the refused one
in the file as the record of what happened. Two *answered* rows for one cell is
not resolved by a rule: the analysis stops and names the cell, because choosing
between two answers to the same question is exactly the decision this package
exists to avoid. Both branches were exercised before the re-run, on copies of
the raw tree with the duplicate built by hand.

**One analysis defect, found after the full run and corrected.** With all four
cells populated, `e2_consistency.csv`'s paired easy-against-hard sensitivity row
for the pooled `(both)` view was keyed on the case id alone, so the two judge
models overwrote each other and a row labelled `(both)` reported one model's 70
pairs (b = 10, c = 11, n = 70, identical to the `claude-sonnet-5` row). It is
now keyed on (judge model, case) and reads n = 140, b = 22, c = 23, p = 1.000.
The per-model rows were correct before and after, the row carries no
multiplicity adjustment and belongs to no family, and the defect could only
show itself once more than one model had data -- which is why the partial run
did not reveal it.

**2026-09-16, E4's model-annotation arm: what the models were given.** Section 5
allots E4 140 calls ("Haiku and Sonnet each labelling 70") and section 6 gives
the estimator (Cohen's kappa per rater pair, bootstrap interval). The prompt was
not specified, and the prompt is the instrument, so the choices are recorded
here rather than read as pre-registered.

`experiments/prompts/annotator_model_v1.yaml` hands a model the same material
the second human annotator gets and nothing else: the four category definitions
(`prompts/v1.yaml` lines 10-19) and the rule for an email that two categories
both fit (lines 24-29), quoted verbatim, no few-shot examples, no statement that
a gold label exists, one call per case. `experiments/runner_annotator.py`
re-reads those line ranges from `prompts/v1.yaml` at startup and refuses to run
if the prompt has drifted from them, so three raters cannot end up working from
three versions of the definitions without the run stopping.

Three differences from the human handout, none of which better wording would
remove:

- the model's quote of the tie-break rule starts one clause earlier. Lines
  24-29 begin mid-sentence, at "be only a single short sentence. Classify based
  on the underlying intent, not the language or politeness of the wording.";
  `experiments/data/annotator2_instructions_zh.md` starts its quote of the same
  rule at "If an email could plausibly".
- the handout is in Traditional Chinese with the English definitions quoted
  inside it and a gloss marked non-authoritative; the model prompt is English
  throughout.
- the handout has a free-text `notes` column; the model returns a category and
  nothing else, because asking a rater to write its reasoning makes it a
  different rater.

The unit is one label per case per rater. A failed call leaves that case out of
the pairing and is counted in `n_excluded`, never scored as a disagreement
(section 7); on 2026-09-16 no call failed. `e4_kappa.csv` reports every pair
available -- the gold labels against each model, and the models against each
other -- all marked descriptive: there is no null hypothesis in that table and
no p-value in it. The second human's rows appear in the same table as soon as
`experiments/data/annotator2_labels.json` exists, a path exercised on a
synthetic labels file and not on real data, because on 2026-09-16 the file does
not exist. The rank-flip check the sprint plan lists under E4 -- does the
system's ranking change under another annotator's labels -- is still not
implemented and is still listed as such in `experiments/README.md`.

## 10. E1 in full

Written 2026-09-14, before the first degraded-prompt call. Section 4 fixes
the four hypotheses and the multiplicity rule; this section fixes everything
else that could otherwise be decided after seeing a number.

### 10.1 The four versions

| # | File | Edit | sha256 |
|---|------|------|--------|
| H1 | `prompts/v2a.yaml` | few-shot examples 4 -> 2 (account and general removed) | recorded in `experiments/results/MANIFEST.json` under `inputs.prompts` |
| H2 | `prompts/v2b.yaml` | few-shot examples 4 -> 0 | same |
| H3 | `prompts/v2c.yaml` | the four category definitions and the tie-break rule removed | same |
| H4 | `prompts/v2d.yaml` | the bilingual / typo / zhuyin / sarcasm paragraph removed | same |

Every line each file keeps is byte-identical to `prompts/v1.yaml`; the diffs
and the predicted effects are in `docs/DEGRADATION_DESIGN.md`. The prompt
files are frozen from this point: a change to any of them after E1 starts is a
new experiment, not a revision of this one.

### 10.2 Runs

Each version: 70 cases x 3 repeats, classifier and judge, `exp_id = e1_<version>`,
420 calls per version and 1,680 in total. Repeat counts are fixed here and are
not extended to chase significance; there is no interim look and no optional
stopping. All four versions are run before any confirmatory test is computed.

### 10.3 Which baseline the pairs come from (frozen: E0, no new baseline run)

**The paired baseline is the E0 main arm, `experiments/results/raw/e0_noise/e0_noise_20260914T115913Z.jsonl`,
repeats 0, 1 and 2.** No fresh v1 arm is run alongside E1.

Three reasons, in the order they matter:

1. **The primary metric can verify the thing a time gap threatens.** The
   classifier is Haiku 4.5, and `response.model` comes back as a dated
   snapshot (`claude-haiku-4-5-20251001`) on every call, recorded per row.
   Whether E0 and E1 ran against the same model version is therefore a fact
   in the raw data, not an assumption. The judge has no such handle -- Sonnet
   echoes the bare alias (section 8.2) -- which is one more reason the
   confirmatory family is `category_match` only.
2. **The baseline has no measurable run-to-run variance to re-estimate.**
   Across E0's five repeats the per-run rate was 0.914286 every time, sample
   SD 0, and not one of the 70 cases changed its verdict between repeats
   (`rates_run_spread.csv`, `per_case_instability.csv`). A second baseline of
   the same size is expected to return the same 64/70.
3. **Cost.** A concurrent baseline is 420 more calls, about $2.09 at the
   measured per-case price. E2 and E4 have no cheaper fallback; E1's baseline
   does.

A fourth, smaller reason: one baseline number means every figure in the study
refers to the same v1 rate. Running a second one would put two v1 rates in the
write-up and force every caption to say which.

**What this choice cannot exclude:** a change in the serving stack behind an
unchanged snapshot id, and judge-side drift of any kind. Both are disclosed
rather than ruled out, in the terms section 8.2 already fixes. A concurrent
baseline would remove the first; nothing available here removes the second.

**Pre-specified reversal.** The frozen choice is void, and a concurrent
baseline arm (`exp_id = e1_v1_concurrent`, v1 x 70 x 3, run in the same
session as the degraded versions) becomes the paired baseline, if any of the
following is true when E1 runs:

- (a) any E1 classifier call returns a `response_model` other than
  `claude-haiku-4-5-20251001`;
- (b) the E1 runs execute after 2026-09-28, fourteen days after E0;
- (c) E1 runs with a different `LLM_CLASSIFIER_MODEL`, a different `anthropic`
  major version, or a different price table than E0 did.

If the fallback triggers, the E0-paired results are still computed and
reported, labelled exploratory, next to the concurrent-baseline ones. The
trigger is a property of the raw data, so it is checked by reading files, not
by judgement at analysis time.

### 10.4 Unit of analysis

One outcome per case per version. A case measured three times is collapsed by
**majority vote** across its repeats -- for the candidate over its three E1
repeats, for the baseline over E0 repeats 0-2. n = 70 pairs.

- The baseline uses three repeats rather than all five so that both sides of
  every pair are the same kind of estimate. On the observed E0 data the choice
  is inert anyway: no case changed its verdict across the five repeats.
- A tied vote can only arise if a failed call leaves an even number of
  observations. Those cases are excluded from the pairing and counted in
  `n_excluded_tie`; no tie-breaking rule is invented after the fact.
- Cases missing from either side (all repeats failed) are excluded and counted
  in `n_excluded_missing`.

Treating each (case, repeat) as a separate pair would inflate n to 210 and
make the exact test anticonservative, because three repeats of one email are
not three independent observations. That view is still produced, as
`paired_mcnemar.csv` with `family = sensitivity_per_repeat_pairing`, without
an interval or a multiplicity adjustment.

### 10.5 The test

Exact McNemar on the discordant pairs of each 2x2 table:

| | candidate correct | candidate wrong |
|---|---|---|
| **baseline correct** | a | **b** |
| **baseline wrong** | **c** | d |

b = the baseline got it right and the degraded version did not; c = the
reverse. The test is the two-sided exact binomial on b out of b + c against
0.5, computed once, after all 1,680 calls are in. Tables report a, b, c and d
so any reader can recompute every number in the row.

### 10.6 Multiplicity

Holm across exactly four p-values -- the four primary-metric tests of section
4 -- at family-wise alpha 0.05. The secondary metric `passed` gets
Benjamini-Hochberg within its own family and is labelled `exploratory_bh` in
every row; its adjusted value never appears in the `p_holm` column. Output:
`experiments/results/tables/e1_main.csv`, one row per (metric, version).

### 10.7 Effect size

- Risk difference `rd = (c - b) / n`, signed so that a degradation is negative.
- 95% interval: Newcombe (1998) method 10 for paired proportions, with the phi
  continuity correction on. Validation status travels in the row's `note`
  column; see the 2026-09-14 entry in section 9.
- Odds ratio `c / b`, left empty when b = 0 rather than patched with a
  continuity constant, with the reason in the `note` column.

### 10.8 Power simulation

`experiments/results/tables/e1_power.csv` and `figures/e1_power.png`.

For each primary-metric comparison and each n in {30, 50, 70}: draw n paired
cases from that comparison's 70 observed pairs **with replacement**, recompute
the exact McNemar test, repeat B = 2000 times, and report the share that
rejects. Analysis seed 20260920; the generator is seeded from (seed, version,
n) so the result does not depend on evaluation order. Two levels are reported
per point: alpha = 0.05 (nominal) and alpha = 0.0125 (what Holm charges the
first test in a family of four -- the worst case for this design).

This is **conditional power**: it answers what a study of size n would find if
the true effect were the one observed here, and it inherits every peculiarity
of these 70 cases. It is not a design power calculation, and section 5 already
records that the sample size was fixed by the dataset before any of this ran.

Figure rules, fixed here so they cannot be relaxed later: simulated points are
open markers on a dashed line; the single realized decision at n = 70 -- did
the test that actually ran reject under Holm -- is a filled diamond at 0 or 1.
Simulated and measured points are never averaged, never joined by a line, and
the footnote states that one realized decision is not a rate.

### 10.9 Failures

Per section 7, unchanged. A call that fails produces a row with `ok: false`,
is excluded from the denominator of its case, and is never scored as a wrong
answer. If failures leave a case with no usable observation on either side,
that case leaves the pairing and is counted; the counts sit next to n in
`e1_main.csv`.

### 10.10 Pre-specified sensitivity analyses

Both are computed whatever the primary result says, and neither can replace it.

1. **The leaked case.** `prompts/v1.yaml`'s fourth few-shot example is case-043
   verbatim, so v2a and v2b remove a leak at the same time as they remove
   examples. H1 and H2 are recomputed with case-043 dropped (n = 69) and
   reported next to the primary rows.
2. **The pairing unit.** The (case, repeat) pairing of `paired_mcnemar.csv`
   is reported as a sensitivity view of section 10.4's case-level test.

### 10.11 Outputs fixed in advance

`e1_main.csv` (one row per metric and version: n, exclusions, a/b/c/d, p_raw,
p_holm or p_bh, rd, CI, odds ratio, family, methods), `e1_power.csv`,
`figures/e1_forest.png`, `figures/e1_power.png`. Before E1 raw data exists the
two CSVs are written with their headers and no rows, and the two figures are
skipped -- `python -m experiments.analyze --self-check` exercises the same
estimators on E0 repeat 1 against repeat 2, a pairing whose true difference is
zero by construction.
