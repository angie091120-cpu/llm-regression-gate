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

**2026-09-17, section 10.1's "byte-identical" sentence, and the five lines it
does not cover.** Section 10.1 says "Every line each file keeps is
byte-identical to `prompts/v1.yaml`". That holds for every kept line, and it
is not the whole account of what the files contain: three of the four also
carry lines that appear nowhere in v1. A line-by-line diff against v1, run
on 2026-09-17 under the normalisation `docs/DEGRADATION_DESIGN.md` prints
(drop `#` comments, the `version:` line and the `created_at:` line), gives
-8 / +0 for v2a, -17 / +1 for v2b, -15 / +3 for v2c and -5 / +1 for v2d. The
five added lines are:

- v2c, two: `Classify each email into exactly one category: billing,
  technical, account` and `or general.` -- a condensed lead-in written for
  that file, keeping the four category labels and dropping the nine definition
  lines that stood under them;
- v2c, one: `not the language or politeness of the wording.` -- v1's line of
  that text with the tie-break clause cut off, so a truncation rather than a
  copy;
- v2d, one: `If an email could plausibly` -- the other half of that same v1
  line, kept because v2d removes the paragraph in front of the tie-break rule
  and not the rule;
- v2b, one: `few_shot_examples: []`, the key with an empty list where four
  example blocks were.

The counts have been in `docs/DEGRADATION_DESIGN.md`'s table as `+3` and `+1`
since the files were built on 2026-09-14; the prose in that file also said
"nothing was re-typed", which is false for v2c, and that sentence has been
replaced by the list above. No analysis number moves: the four confirmatory
tests read model outputs, the prompt files are inputs, and their sha256 are in
`MANIFEST.json` either way. Section 10.1 is frozen text and is not edited;
this entry is the correction to it.

**2026-09-17, section 10.3's reversal condition (c) cannot be checked by
reading the raw data, and what the SDK-version claim rests on instead.**
Section 10.3 ends with "the trigger is a property of the raw data, so it is
checked by reading files, not by judgement at analysis time". That is true of
(a), which is `response_model` on every E1 classifier row, and of (b), which
is `timestamp_utc`. Condition (c) names three things: `LLM_CLASSIFIER_MODEL`
is in each meta file as `classifier_model_requested` and the price table is
there as `prices_per_mtok`, both checkable, and the `anthropic` version is
**in no meta file and in no raw row this study produced.**

What the SDK version rests on instead: `requirements.txt` pins
`anthropic==0.117.0`, and `git show <commit>:requirements.txt` returns that
pin at every commit the meta files record -- 434fc31 (E0 main arm and
judge-isolation arm), 786f32a (all four E1 arms), 12ffbff (E2 calibration and
batch 1), 957d936 (E2 batch 2 and E4). The repository interpreter `.venv`
carries 0.117.0 today -- a property of this machine, checkable by whoever has
it and by nobody else, since a virtualenv is not in the repository -- and
`experiments/README.md` fixes the convention that study runs use it rather
than the analysis venv, which carries 1.5.0. A pin
plus a convention is not a record of what executed. The probe files are the
nearest artifact and they are separate processes:
`probes_sdk0.117.0_20260914T113405Z.json` and
`probes_sdk1.5.0_20260914T113518Z.json` were both written on the E0 day, so
"only one SDK was installed on that machine" is not available as an argument;
`probes_sdk0.117.0_20260916T143354Z.json` and `..._20260916T143923Z.json` sit
18 and 13 minutes before the E2 re-run; and there is no probe file at all on
2026-09-15, the E1 day.

Per arm, in the strongest form the evidence supports: E0, the four E1 arms,
both E2 batches and E4 ran with `anthropic==0.117.0` pinned at their recorded
commit, and not one of them carries a field showing that the interpreter
obeyed the pin. `temperature` was null on every call in the study, so the one
row field that reacts to the SDK version -- `temperature_transport`, which
resolves to `param` on 0.117.0 and `extra_body` on 1.5.0 -- reads `none`
throughout and separates nothing. The 2026-09-14 smoke run sits outside that
list: it predates the pin (its commit 2727d9e has `anthropic>=0.40`) and
predates `experiments/runner.py` being committed at all, so the runner that
wrote it was uncommitted at run time.

Nothing in E1 turns on this. The two auditable parts of (c) did not fire, and
neither did (a) or (b). The gap is closed going forward only: `runner.py`,
`runner_pairwise.py` and `runner_annotator.py` now write
`anthropic_sdk_version`, read from `anthropic.__version__` at run time, into
the `.meta.json` of every run from 2026-09-17. No existing meta or raw file is
back-filled, for the reason the 2026-09-14 cost entry above gives: a raw file
is evidence.

**2026-09-17, one email address in `golden_dataset.json` belonged to a real
company; the dataset moves to v1.1.** `case-007`'s `input_text` carried
an invented account name at a domain that is not invented. The dataset is
fictional by design (section 2: NebulaDesk and everyone in it are invented),
and that domain belongs to a live Taiwanese CDN and DDoS-mitigation provider
-- checked on 2026-09-17, it resolves and serves that company's site. The
address itself is not reprinted here. It is still readable in seven files
that are deliberately not rewritten: the six raw JSONL files described below,
and `experiments/data/annotator2_sheet.csv`, the human-readable handout built
from the dataset at v1 and left at v1 for the reason given later in this
entry. Those seven are the whole residue, and it stays. Publishing a support-email
corpus that places an invented account holder at a real company's domain is a
privacy and impersonation risk with no analytical benefit on the other side
of it, so the address is now `lin.admin@example.com`, in the domain RFC 2606
reserves for documentation.

What changed: one string, in one case. `case-007`'s category, summary,
difficulty, language and `label_status` are untouched, the other 69 cases are
untouched byte for byte, and the envelope's `dataset_version` moves from `v1`
to `v1.1` so that the edit is visible in the file and not only in this entry.
sha256 before
`af33d77841d9c17f55f4acd3abbe241d237333e433cf4545263a6e4d6fa1c3d9`, after
`1590949032efbf123ca11394b3c12576aa3b2315a9edb03d30f01d0d6cdf2d5e`; the file
goes from 62,562 to 62,560 bytes. `tests/test_golden_dataset.py` pins the
shipped envelope version and now pins `v1.1` -- the assertion is moved, not
relaxed.

What did not change is any result. Every number in this study was computed
from model outputs that already exist under `experiments/results/raw/`, and
those files are evidence: the old address survives in 30 places across six
raw files, inside model summaries that quoted the email they were handed, and
not one of them is rewritten. Each run's `.meta.json` names the dataset it
read and belongs to the day it ran, so none is back-filled.
`MANIFEST.json` is regenerated from the file on disk and now carries the
v1.1 sha256, which is the only published artifact that moves.
`experiments/data/annotator2_sheet.csv` deliberately does not move with it:
it is the sheet a person outside the project was handed, built from the
dataset as it stood at v1, and its sha256
`c7abaf6fe77dc204549f31853ef348c1b370240737a0b1a55e4b758ea2a462ba` is
recorded in `experiments/README.md` as the handout. Regenerating it would
make the record describe a file that annotator never saw, so the sheet stays
at v1 and `experiments/import_annotator2.py` knows that `case-007`'s body
will come back in its v1 form. A future run of
any arm will send the v1.1 text; its raw file was never going to be
byte-comparable with these ones in any case, because the calls cannot be
seeded.

**2026-09-17, what the gate did after v1.1, and what E0's five repeats do not
cover.** End-to-end `evalkit` runs of `prompts/v1` over dataset v1 returned
64/70 -- E0's five repeats, the 2026-09-14 baseline bootstrap and the gate
runs on PR #4 before the dataset moved. Runs over dataset v1.1 against that
same v1 baseline have returned 63/70. `eval-gate.yml` posts a scorecard on
every push, so counting those runs in this document would date it within a
day; what follows is the one run with a per-case report, made locally at 2026-09-16 19:10 UTC (2026-09-17 03:10 +08:00) ($0.419296, 70 cases).

That run names the case a scorecard cannot: **`case-056`**, a request for a
signed W-9 form. Baseline: predicted `billing`, judge 5, passed. This run:
predicted `general`, `category_match` false, judge 5, not passed. The category
moved; the judge score did not, and the summary still describes a W-9
request. `case-007`, the case whose email address v1.1 edited, answers
`account` with judge 5 and passes in both runs, so the string edit did not
move it. Gate runs produce no per-case artifact, so whether any of them lost
the same case is not recoverable from the scorecard.

`case-056` is a billing/general boundary case, and this study's own raw data
says where its instability sits. The classifier answered `billing` on all 17
of its calls -- E0's five repeats and three each under v2a, v2b, v2c and v2d
-- and both E4 model annotators also said `billing`. The wobble in the raw
data is on the judge instead: of the 21 judge scores that case carries, twenty
are 5 and one is a 2, on `e1_v2b` repeat 0.

The judge moved on far more than one case between the two baseline runs, and
only the verdicts hid it. Comparing them case by case: 51 of the 70 cases
carry the same `judge_score` in both, **19 carry a different one** -- fifteen
by a single point and four by two, eight up and eleven down (`case-012`
5 -> 3, `case-048` 5 -> 3, `case-016` 3 -> 5, `case-044` 3 -> 5, and fifteen
more) --
and not one of those 19 changed its verdict, because the pass threshold is 3
and the movement stayed on one side of it. So the single flipped verdict is
the visible part of a scoring surface that was moving underneath it on more
than a quarter of the dataset.

What this does not license is calling the 63/70 runs a noise floor. Section
6's run-to-run figures are five repeats inside one session on 2026-09-14, and
`rates_run_spread.csv` reports their sample SD as 0. A classifier that never
changed a verdict across those five repeats has still changed one two days
later, on a different session, which means the five repeats measure
within-session stability and are not a bound on anything wider. No number in
this document is revised on the strength of these runs; the observation is
recorded because the opposite -- a stability claim resting on five repeats --
would be.

**2026-09-19, the 70 category labels are being re-labelled blind by three
people; this entry fixes the protocol before any returned sheet is compared
with anything.** Section 2 records the decision step -- "labels decided by one
human annotator (the author) in July 2026" -- and not the whole provenance. An
agent drafted every case together with `draft_category`, `draft_summary` and
`draft_difficulty`, and the author confirmed all 70 on 2026-07-20. What this
repository records of that step: every case carries `labeled_at` 2026-07-20 and
`label_status: confirmed`; 69 of 70 `expected_category` equal `draft_category`,
the exception being `case-031` (`account` -> `billing`, unified with
`case-052`); every `expected_summary` and every `expected_difficulty` equals its
draft; and six cases carry a dated 2026-07-20 ruling in `notes` (`case-012`,
`case-020`, `case-031`, `case-043`, `case-052`, `case-068`), in three pairs. The
case-by-case review sheet those rulings were made on is held privately by the
author and is not published here, so what it contained, and how much of it A1
had read, are statements in this document and not artifacts a reader can open.
What is auditable about that sheet is its hash, committed here before any
returned sheet is read: sha256
`3fac2e1393ea3cee61577f4c192a27e5f44f5afae2217926f80f268b6a38e451`, 26,925
bytes. Agent-drafted, author-confirmed labels are not an independent human
annotation of these emails. `e4_kappa.csv`'s two
`human-1 (gold)` rows (0.904658 against Haiku, 0.904632 against Sonnet)
therefore measure a model against a label set that began as a model's draft --
a sharper statement than section 8.3's shared blind spot, and not a
human-model agreement rate. How far that origin is shared with the two model
annotators cannot be stated either way: the model identity of the drafting agent
is recorded nowhere in this repository, so whether it was the same family as
Haiku 4.5 or Sonnet 5, or a different one, is not recoverable, and that unknown
is itself a limit on reading those two rows. Three people now label the same 70
emails blind, so that a label set exists which no model drafted.

**The three annotators.** A1 is the author, who labelled on 2026-09-19 from the
same handout an outside annotator was already given --
`experiments/data/annotator2_sheet.csv` at v1, sha256
`c7abaf6fe77dc204549f31853ef348c1b370240737a0b1a55e4b758ea2a462ba`, with
`experiments/data/annotator2_instructions_zh.md`. One limit on A1 belongs next
to the number rather than after it: A1 read the 2026-07-20 review sheet two
months earlier and ruled on the contested groups, so A1 is blind to the label
column and not to the dataset. A2 and A3 are two people outside the project
working from that same sheet and those same instructions, each alone, with no
AI assistance and no discussion with each other or with A1. That last pair is an
undertaking A2 and A3 give, not a property of anything in the record: no field
of a returned sheet distinguishes a label written unaided from one that was not,
and this document does not claim otherwise. No annotator sees the shipped labels
or any model output. A2 and A3 are expected back by 2026-09-23.

All three work from the sheet as it stood at dataset v1, which differs from v1.1
in one string -- `case-007`'s email address, changed for the privacy reason the
2026-09-17 entry above gives. Gold v2's label for `case-007` is therefore a
label on the v1 wording of that email, and the two wordings differ by one
address.

**Hashes are recorded on arrival, before comparison.** Each returned file's
sha256 goes into this section when the file is received and before it is read
against anything else. A1's category file, received 2026-09-19:
`774e62bc6c8337b188b87f249b66f1991196c6d616ef4d6d2e1cae1cde02cfbb`. The
author's reference-summary review file, received 2026-09-19:
`48b1d1d3c5becab83c6554920499eeced3a643774e77c6ffcb655d8f4d6a7759`, 20,750
bytes; at the moment that hash was registered the file's contents had been
neither read nor validated. The labelling files enter the repository in the
import pull request; the pull request carrying this entry changes documentation
only and opens no labelling file.

A sheet that fails validation -- a case missing, a category outside the four, a
file that will not parse -- goes back to its annotator for correction and is not
repaired here. The corrected file's sha256 is recorded as its own dated line in
this section, and the hash of the version that failed stays above it, so the
record shows that a file was replaced and which one it was.

**Gold v2 and the rule that produces it.** Categories only. Per case, the
majority of A1, A2 and A3. Where all three differ, the author rules on that
case. The adjudication sheet carries the email text and that case's existing
human labels -- three of them on the main path, two under the two-vote rule and
under fallback 1 -- and nothing else: it does not carry `golden_dataset.json`'s
`draft_*` fields, the dated 2026-07-20 rulings in `notes`, the v1.1
`expected_category`, or model output from any arm of this study. Every
adjudicated case id is listed.

Where all three sheets have arrived and one case is short a valid vote -- that
sheet omitted the case, or its entry failed validation and no correction came
back -- two agreeing votes decide it and two differing votes go to the same
adjudication rule on the same restricted sheet. That rule covers per-case gaps
and nothing wider. A sheet that never arrives is not a gap but a fallback: once
fallback 1 is in force, every case is decided by fallback 1's rule and the
two-vote rule applies to none of them.

Two fallbacks, fixed now so that neither is chosen after a disagreement rate is
known. **Fallback 1**, if by 2026-09-27 the returns are A1 plus exactly one of
A2 and A3: the cases those two humans agree on are final, and the cases they
split on take the shipped v1.1 label as a third vote. Where that third vote
leaves A1, the other human and v1.1 all different from one another, the author
rules under the restrictions above and the case is recorded as `adjudicated`.
Where fallback 1 leaves a case with only A1's vote valid -- the other human
omitted that case, or its entry failed validation and no correction arrived by
2026-09-27 -- the case keeps its v1.1 label and is recorded as `fallback_v1.1`.
Fallback 1 pulls labels of agent-draft origin back into gold on exactly the
cases where the two humans disagreed; those cases stay individually
identifiable by their `fallback_v1.1` route, so any reading of gold v2 can be
redone with them excluded. **Fallback 2**, if only A1 has returned: gold stays
at v1.1 and A1's sheet is reported as a reliability check and used for nothing
else.

One disclosure about the fallback 1 adjudication path: its trigger is itself
informative. Handing the author a case at all says that A1, the other human and
v1.1 are three different labels, which over four categories narrows v1.1 to the
two the sheet does not show. The main path carries no such leak -- three humans
all different is a fact about three human sheets, and no shipped label takes
part in the trigger. What the adjudication sheet may show is unchanged either
way; what cannot be withheld on this path is the inference its trigger
permits.

A sheet that arrives after a fallback has been applied is reported as one more
reliability check against the gold that exists. It does not reopen gold v2.
Revising a finished label set on the strength of a late arrival is the kind of
choice this section exists to take off the table.

**What is not re-labelled.** `expected_difficulty` stays as it is,
agent-drafted and author-confirmed. `expected_summary` is not rewritten either,
and it is reviewed case by case in the second pass described next, whose result
is reported under the limits fixed there. `passed` is the secondary metric of
section 3 and the E2 pairwise arm is an exploratory analysis under section 4;
both keep scoring against the reference summaries as they stand, and
re-labelling categories does not change the text they score against.

**A second pass over the reference summaries, and what it may be used for.**
From 2026-09-19 the author reviews all 70 `expected_summary` values one case at
a time, marking each `ok` or `edit` and writing out a replacement wherever the
mark is `edit`. A summary passes on four criteria: it is faithful to the email
and adds no fact the email does not carry; it covers the main request; its tone
is neutral; and its language matches the email's, which means Chinese for the
code-switched cases. The review sheet carries the full email text and the
current reference summary, and no category label of any kind. This pass starts
only after A1's category sheet is sealed and its sha256 recorded above, and the
two rounds do not feed back into each other: nothing seen while reviewing
summaries may change a category label already submitted, and the category sheet
is not consulted while reviewing summaries. The review file's sha256 goes into
this section when it arrives and before its contents are read.

What the review may be used for, fixed here before its counts are known: this
study reports how many cases are marked `ok`, how many are marked `edit`, and
the ids of the cases marked `edit`. No judge score is obtained again on account
of this review: every `judge_score` already in the raw files stands as written,
the E0 judge-isolation arm and E2 are not recomputed at all, and no API call is
sent. Those scores were produced against the summaries that exist, and scoring
them again against rewritten ones is a different measurement rather than a
correction to this one.

One quantity does move under gold v2, and it moves for the category
re-labelling and not for this review. `passed` is `category_match` AND
`judge_score >= 3` (section 3), so a case whose gold category changes can change
its `passed` value. That recomputation is arithmetic over judge scores already
recorded -- the same numbers, recombined with the new `category_match` -- and it
obtains no new score and sends no call. E2 carries no `category_match` and is
untouched by it.

The author's replacements are stored in a new dataset field or in a file of
their own -- which of the two is a choice for the import pull request -- and do
not overwrite `expected_summary`. Overwriting it, and re-running the judge
against the result, takes a further dated entry in this section and is reported
as the new measurement it would be.

**What gets reported.** Agreement: Cohen's kappa for each human pair, Fleiss'
kappa and Krippendorff's alpha (nominal) across the three, each annotator
against the shipped v1.1 labels, and each of the two model annotators against
gold v2 and against each human annotator. Intervals are the case-level
bootstrap section 6 already fixes, B = 10,000, seed 20260920. Results: the E0,
E1 and E5 tables are recomputed on gold v2 and published beside their
pre-registered v1.1 versions, with the differences listed table by table. That
recomputation re-runs the analysis over the raw files that already exist -- no
call is re-sent and no raw file changes. No case is dropped for having been
re-labelled, the confirmatory family of section 4 and the multiplicity rules of
sections 4 and 10.6 are untouched, and section 10.10's `case-043` leakage
sensitivity analysis runs as written.

**Which of the two sets is confirmatory.** The pre-registered v1.1 results stay
the confirmatory result of section 4. The gold v2 recomputation is a
sensitivity analysis: reported beside the v1.1 version with the differences
listed table by table, raising no second confirmatory family, and yielding no
Holm-adjusted confirmatory claim. Every gold v2 row carries its own sensitivity
family, `sensitivity_gold_v2`, and takes no multiplicity correction of any
kind: `p_holm` and `p_bh` are left empty and the `note` column says to read
`p_raw`. That is the form `sensitivity_drop_leaked_case` already ships in
`e1_main.csv`, and it is reused here rather than invented. The family is
pooled with none of the v1.1 families, so no published v1.1 `p_holm` or `p_bh`
changes value. Holm stays fixed across exactly the four tests of section 4
computed on the v1.1 labels. The gold v2 versions of the E5 strata rows and of
`passed` follow the same rule.

E5's logistic model keeps the specification the 2026-09-16 entry above records:
the pooled fit over E0 and the four E1 arms, with the baseline-arm fit still
marked `fitted = no` for the separation documented there. If a re-labelled case
breaks one of the constant-outcome levels and the baseline arm becomes
estimable under gold v2, that fit is reported as an addition, marked
exploratory, and does not replace the pooled specification.

The import pull request adds a `note` to `e4_kappa.csv`'s two `human-1 (gold)`
rows pointing back at this entry, so the provenance of that rater travels in
the table beside the number. The pull request carrying this entry changes no
results table.

**Dataset version.** A category change is a label change, so the envelope moves
to `dataset_version: v2.0`. `labeled_by` stops carrying a personal name and
carries the decision route for that case instead (`majority`, `adjudicated` or
`fallback_v1.1`). Under fallback 1 those routes read: `majority` where the two
humans agree, `fallback_v1.1` where they split and the shipped label breaks the
tie, and `adjudicated` where the two humans and the shipped label are all
different and the author ruled. The CI baseline is rebuilt once on v2.0 -- one
real-API `evalkit` run, recorded in `docs/DECISIONS.md` as its own entry the
way D-011 recorded the v1.1 rebuild.

**Ordering.** This entry is written and merged before any returned sheet is
compared with the shipped labels or with another sheet. The evidence is the git
history of this file, the arrival hash of each labelling file recorded above,
and the handout hash already published in `experiments/README.md`, which pins
that every annotator worked from one sheet. Sections 1 to 8 and section 10 are
frozen text and are not edited; this entry is the correction to section 2's
provenance sentence, to how section 8.3 is read, and to section 8's closing
line -- "E4 has one second annotator, so annotator variance is not estimable"
stops holding once three people have labelled these 70 emails, and the
three-rater agreement numbers are where it stops. `docs/DECISIONS.md` D-012
records the choice and the options it was chosen over.

Not every ordering in this entry is equally auditable. That the entry precedes
any comparison is a property of the git history. That the summary review began
only after A1's category sheet was sealed is a statement: both fall on
2026-09-19, and the commit sequence supports the order in which the two hashes
were registered, not the order in which the author worked.

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
