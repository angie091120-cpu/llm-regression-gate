# experiments/ -- pilot study on LLM-as-judge reliability and gate sensitivity

Additive package. Nothing here modifies `evalkit/`, `llm.py`, `classifier.py`,
`prompts/v1.yaml`, `golden_dataset.json` or `tests/`; the production suite
still runs 46 green tests
(`checks/experiments_acceptance.sh` C3 asserts both the result and the count).
Where the experiment and the production system must agree, production code is
imported rather than copied: `classifier.load_prompt_config`,
`classifier._few_shot_messages`, `evalkit.judge.JUDGE_SYSTEM_PROMPT`,
`evalkit.dataset.load_confirmed_cases`, `evalkit.cost`, `llm._model_for`.

Design, hypotheses and the limits of what these numbers can support:
[`docs/PREREGISTRATION.md`](../docs/PREREGISTRATION.md).
Measured costs and API behaviour: [`COST_CALIBRATION.md`](COST_CALIBRATION.md).

## Layout

```
experiments/
  client.py        one instrumented Anthropic call: response.model, full usage,
                   raw response, per-call temperature, structured error capture
  runner.py        evaluation loop -> one JSONL line per API call
  runner_pairwise.py
                   E2 only: pairs of already-produced summaries, each judged in
                   both orders by both judge models; same raw-row contract,
                   imported from runner.py rather than copied.
                   --resume-failed-from re-sends only the cells an earlier run
                   recorded as failures
  runner_annotator.py
                   E4 only: one category label per (model, case) from the
                   category definitions alone, no few-shot examples
  analyze.py       raw JSONL -> CSV tables + PNG figures + MANIFEST.json (no API)
  stats.py         estimators, standard library only, self-checked
  probes.py        four API-behaviour probes quoted in COST_CALIBRATION.md
  import_annotations.py
                   validate and ingest one E4 annotator's returned sheet, A1,
                   A2 or A3 (no API, reads no gold label). Was
                   import_annotator2.py until 2026-09-19
  import_summary_review.py
                   validate and ingest the author's ok/edit review of the 70
                   reference summaries (no API, reads no category label)
  gold_v2.py       turn the returned sheets into gold v2 by the
                   PREREGISTRATION section 9 rules, or refuse and say which
                   rule is missing
  apply_gold_v2.py write a sealed gold v2 into golden_dataset.json and move it
                   to v2.0 (dry run unless --write; not run by the import PR)
  prompts/         experiment-only prompts (judge_pairwise_v1.yaml,
                   annotator_model_v1.yaml). Kept out of the repo's prompts/,
                   which is the classifier's version directory; analyze.py
                   hashes these into MANIFEST.json under "experiment_prompts"
  data/            E4 handout (annotator2_sheet.csv,
                   annotator2_instructions_zh.md), the returned sheets exactly
                   as received (a1_*.csv, and A2's and A3's when they arrive),
                   annotations/<id>_labels.json from import_annotations.py,
                   the summary review, and gold_v2.json once the panel
                   produces it
  results/
    raw/<exp_id>/<run_id>.jsonl        raw calls, committed
    raw/<exp_id>/<run_id>.meta.json    run provenance, committed
    probes/                            probe output, committed
    tables/*.csv  figures/*.png        derived, committed
    MANIFEST.json                      sha256 + line counts + recomputed cost
    cost_ledger.json                   appended by every runner run
    cache/                             local only, gitignored
```

## Environment

WSL Ubuntu ships Python 3.14 without `ensurepip`, so `python3 -m venv` fails
at the pip step. What works:

```bash
python3 -m venv --without-pip ~/.venvs/lrg-exp
PYTHONPATH=$HOME/.local/lib/python3.14/site-packages \
  ~/.venvs/lrg-exp/bin/python -m pip install --ignore-installed pip
~/.venvs/lrg-exp/bin/python -m pip install \
  anthropic pydantic PyYAML Jinja2 pytest numpy scipy statsmodels matplotlib
```

All nine packages installed from cp314 wheels on 2026-09-14: numpy 2.5.3,
scipy 1.18.1, statsmodels 0.15.0 (with pandas 3.0.5), matplotlib 3.11.2,
anthropic 1.5.0. Nothing had to be built from source and no fallback was
needed. The fallbacks exist anyway, because a cp314 wheel that exists today
may not exist for the next release:

- every estimator in `stats.py` is standard library only; scipy is a
  cross-check (`python -m experiments.stats --cross-check`), never the source
  of a published number;
- `analyze.py` writes all CSV tables before it touches matplotlib, and prints
  a warning instead of failing if matplotlib is missing.

**Two interpreters, on purpose.** The repo's own `.venv` pins
`anthropic 0.117.0`; the analysis venv has 1.5.0, which removed
`temperature`/`top_p`/`top_k` from `messages.create`. `client.py` detects this
and switches transport, so the runner works on both, but runs for the study
use the repo interpreter so that the SDK matches what the repository declares:

```bash
# runner (API calls)
/path/to/llm-regression-gate/.venv/bin/python -m experiments.runner ...
# analysis (numpy/scipy/matplotlib)
~/.venvs/lrg-exp/bin/python -m experiments.analyze ...
```

Environment variables:

| Name | Where | Note |
|------|-------|------|
| `ANTHROPIC_API_KEY` | `.env` at the repo root (gitignored, mode 600) | loaded by `runner.py`; names are logged, values never |
| `PRICE_CLAUDE_SONNET_5_INPUT=2.00` | export before a run on any checkout that predates D-010 | this branch was cut from a `main` whose `evalkit/cost.py` still had $3/$15; the runner aborts unless the effective price is $2/$10, whichever way it gets there |
| `PRICE_CLAUDE_SONNET_5_OUTPUT=10.00` | same | same |
| `LLM_CLASSIFIER_MODEL` / `LLM_JUDGE_MODEL` | optional | same contract as production `llm.py` |
| `EXP_PYTHON` | optional | interpreter for `checks/experiments_acceptance.sh`; unset, it takes `~/.venvs/lrg-exp/bin/python` if that exists, then the repo `.venv`, then `python3` |

## Running

```bash
export PRICE_CLAUDE_SONNET_5_INPUT=2.00 PRICE_CLAUDE_SONNET_5_OUTPUT=10.00

# see the plan and the call count without spending anything
python -m experiments.runner --exp-id e0_noise --repeats 5 --dry-run

# E0 noise floor
python -m experiments.runner --exp-id e0_noise --repeats 5 --max-cost-usd 3.00

# E0 judge-isolation arm: freeze repeat 0's classifier outputs, re-judge 4x
python -m experiments.runner --exp-id e0_judge_iso --repeats 4 \
  --judge-only-from experiments/results/raw/e0_noise/<run_id>.jsonl

# E1 degradation arm
python -m experiments.runner --exp-id e1_v2a --prompt-version v2a --repeats 3

# E2 pairwise judge: 70 pairs x 2 layers x 2 orders x 2 judge models
python -m experiments.runner_pairwise --exp-id e2_pairwise \
  --baseline-from experiments/results/raw/e0_noise/<run_id>.jsonl \
  --contrast-from experiments/results/raw/e1_v2b/<run_id>.jsonl \
  --max-cost-usd 3.00

# E2 re-run: only the cells an earlier run failed on, into a new raw file
python -m experiments.runner_pairwise --exp-id e2_pairwise \
  --baseline-from experiments/results/raw/e0_noise/<run_id>.jsonl \
  --contrast-from experiments/results/raw/e1_v2b/<run_id>.jsonl \
  --resume-failed-from experiments/results/raw/e2_pairwise/<run_id>.jsonl \
  --max-cost-usd 3.00

# E4 model-annotation arm: 70 cases x 2 models, definitions only
python -m experiments.runner_annotator --exp-id e4_annot --max-cost-usd 1.00

# analysis (free, re-runnable)
python -m experiments.analyze --seed 20260920

# acceptance (its first line names the interpreter and whether scipy is there)
bash checks/experiments_acceptance.sh
EXP_PYTHON=$HOME/.venvs/lrg-exp/bin/python bash checks/experiments_acceptance.sh
```

Useful flags: `--cases 10` or `--cases case-001,case-064` or `--cases @ids.txt`;
`--concurrency` (default 4); `--skip-judge`; `--classifier-temperature`;
`--max-cost-usd` (hard stop, default $1.00); `--temperature-transport`.

Three free checks that call nothing:

```bash
# the E1 estimators on E0 repeat 1 vs repeat 2, where the answer must be zero
python -m experiments.analyze --self-check
# coverage evidence for the paired risk-difference interval
python -m experiments.stats --coverage
# the standard-library estimators against scipy (analysis venv only)
~/.venvs/lrg-exp/bin/python -m experiments.stats --cross-check
```

`--self-check` belongs to `analyze.py`. `stats.py` takes `--coverage`,
`--cross-check` and `--n-sim`, and nothing else: both entry points exit 2 on an
unrecognised flag instead of falling through to their default run, so a typo
such as `--corss-check` fails loudly. C7 of
`checks/experiments_acceptance.sh` asserts that.

### The cache is off by default and refuses to run with repeats

`--cache` stores responses by a hash of the full request. Serving a repeated
identical request from cache would report a noise floor of exactly zero, which
is the opposite of what E0 measures, so `--cache` together with `--repeats > 1`
is rejected with an error rather than a warning. Cache hits, when used, are
written as rows with `cache_hit: true` and `cost_usd: 0`.

## Raw data contract

One JSONL line per API call, including failures. Fields consumed by
`analyze.py`: `schema_version, call_id, run_id, exp_id, repeat_idx, case_id,
prompt_version, tier, request_model, response_model, temperature,
temperature_transport, max_tokens, ok, input_tokens, output_tokens, usage,
latency_ms, cost_usd, price_key, timestamp_utc, cache_hit, request_hash,
stop_reason, has_thinking_block, parsed, error_type, error_message,
status_code, request_id, raw_response_json`.

Three properties are load-bearing:

1. **`response_model` comes from the API response, never from the request
   string.** For Haiku that resolves the alias to a dated snapshot; for Sonnet
   the API currently echoes the alias (see COST_CALIBRATION.md 4.2).
2. **A failed call is a row, not an exception.** 429/5xx/timeout/schema
   violations are recorded with the error and the batch continues, so
   "1,680 planned, 1,676 succeeded" is visible in the data instead of being a
   crashed job.
3. **Cost is recomputed from `usage`,** by `analyze.py`, from the raw files at
   the prices pinned in `experiments/__init__.py` -- not carried over from the
   `cost_usd` field the runner wrote, and not affected by a `PRICE_*` override
   in whatever shell the analysis happens to run in. A call that returned
   tokens and then failed validation is charged; `MANIFEST.json` reports the
   recomputed total, the runner's total and the difference between them
   (COST_CALIBRATION.md section 3).

## Runs completed

| Date | exp_id | Design | Calls (ok/planned) | Cost | Raw file |
|------|--------|--------|--------------------|------|----------|
| 2026-09-14 | `smoke` | 10 cases x 1, cost calibration only | 20/20 | $0.059746 | `results/raw/smoke/smoke_2026-09-14.jsonl` |
| 2026-09-14 | `e0_noise` | E0 main arm: v1 x 70 cases x 5 repeats, classifier + judge | 700/700 | $2.080951 | `results/raw/e0_noise/e0_noise_20260914T115913Z.jsonl` |
| 2026-09-14 | `e0_judge_iso` | E0 judge-isolation arm: repeat 0 classifier output frozen, judge re-scores 4x | 279/280 | $1.084140 | `results/raw/e0_judge_iso/e0_judge_iso_20260914T120557Z.jsonl` |
| 2026-09-15 | `e1_v2a` | E1 H1: v2a (few-shot 4 -> 2) x 70 cases x 3 repeats | 420/420 | $1.226163 | `results/raw/e1_v2a/e1_v2a_20260915T153333Z.jsonl` |
| 2026-09-15 | `e1_v2b` | E1 H2: v2b (few-shot 4 -> 0) x 70 x 3 | 420/420 | $1.185492 | `results/raw/e1_v2b/e1_v2b_20260915T153711Z.jsonl` |
| 2026-09-15 | `e1_v2c` | E1 H3: v2c (category definitions and tie-break rule removed) x 70 x 3 | 420/420 | $1.189659 | `results/raw/e1_v2c/e1_v2c_20260915T154039Z.jsonl` |
| 2026-09-15 | `e1_v2d` | E1 H4: v2d (bilingual / typo / zhuyin / sarcasm paragraph removed) x 70 x 3 | 420/420 | $1.237953 | `results/raw/e1_v2d/e1_v2d_20260915T154407Z.jsonl` |
| 2026-09-15 | `e2_smoke` | E2 calibration: 2 cases x 2 layers x 2 orders x 2 judge models, to price the pairwise prompt | 16/16 | $0.061943 | `results/raw/e2_smoke/e2_smoke_20260915T172942Z.jsonl` |
| 2026-09-15 | `e2_pairwise` | E2 main run: 70 pairs x 2 layers x 2 orders x 2 judge models. **Stopped by the account's API spend limit after 94 calls** | 94/560 | $0.443440 | `results/raw/e2_pairwise/e2_pairwise_20260915T173047Z.jsonl` |
| 2026-09-16 | `e2_pairwise` | E2 re-run of the 466 cells the limit refused: same design, same pairs, same prompt | 466/466 | $1.553728 | `results/raw/e2_pairwise/e2_pairwise_20260916T145210Z.jsonl` |
| 2026-09-16 | `e4_annot` | E4 model-annotation arm: Haiku 4.5 and Sonnet 5 each label the 70 cases once, from the definitions alone | 140/140 | $0.312869 | `results/raw/e4_annot/e4_annot_20260916T145907Z.jsonl` |

The isolation arm's cost is $0.005338 higher than the figure the runner
printed on the day: the one failed call was billed and recorded at $0, and
cost is now recomputed from `usage` (COST_CALIBRATION.md section 3).

Neither E0 arm used `--cache`; `cache_hits` is 0 in both meta files.
`response_model` came back as `claude-haiku-4-5-20251001` for every classifier
call and as the bare alias `claude-sonnet-5` for every judge call, which is
what section 4.2 of COST_CALIBRATION.md predicted.

The one failed call is `case-003` repeat 3 in the isolation arm: a schema
violation, not a network error -- the judge returned a tool input with no
`score` field. Per PREREGISTRATION section 7 the row stays in the raw file,
is excluded from that case's denominator (3 scores instead of 4), and the run
was not repeated to replace it.

What E0 produced: `results/tables/rates_by_run.csv`,
`rates_run_spread.csv`, `per_case_instability.csv`,
`judge_rescore_stability.csv`, `judge_rescore_summary.csv` and
`results/figures/noise_floor_vs_gate_thresholds.png`. The `fleiss_kappa` and
`krippendorff_alpha` columns of the two judge tables are still empty, and
since 2026-09-19 for a different reason than before: both estimators now exist
(they were built for the three-annotator panel described below), and both are
nominal-scale coefficients, while these repeats are ordinal 1-5 judge scores.
No ordinal agreement coefficient is specified anywhere in this study, so the
cells stay empty and the note in the table says that rather than the older
"not implemented".

Still to run: nothing that costs an API call. A2's and A3's sheets are the
inputs this study is still waiting on, due 2026-09-23.

## What E1 found

The four arms ran on 2026-09-15 between 15:33 and 15:48 UTC: 1,680 calls,
1,680 successful, no failures, no `--cache`, `cache_hits` 0 in all four meta
files. `response_model` was `claude-haiku-4-5-20251001` on all 840 classifier
calls and the bare alias `claude-sonnet-5` on all 840 judge calls, so none of
PREREGISTRATION section 10.3's reversal conditions fired and the paired
baseline is the frozen E0 one (repeats 0-2, 64/70 on both metrics).

| Version | What it removes | base -> degraded | b | c | RD | 95% CI | p | p Holm |
|---------|-----------------|------------------|---|---|----|--------|---|--------|
| v2a | 2 of the 4 few-shot examples | 64/70 -> 64/70 | 1 | 1 | 0.000 | -0.064 to +0.064 | 1.000 | 1.000 |
| v2b | all 4 few-shot examples | 64/70 -> 64/70 | 1 | 1 | 0.000 | -0.064 to +0.064 | 1.000 | 1.000 |
| v2c | the category definitions and the tie-break rule | 64/70 -> 56/70 | 10 | 2 | -0.114 | -0.219 to -0.015 | 0.039 | 0.154 |
| v2d | the bilingual / typo / zhuyin / sarcasm paragraph | 64/70 -> 65/70 | 0 | 1 | +0.014 | -0.041 to +0.075 | 1.000 | 1.000 |

**Nothing is significant under the pre-registered correction.** One version
moves at all. v2c drops the gate from 64/70 to 56/70, and its exact McNemar
p of 0.0386 would clear a nominal 0.05 if it were the only test in the study;
Holm across the four pre-registered tests puts it at 0.154. The other three
sit on top of the baseline: v2b, the version DEGRADATION_DESIGN predicted
would fall the furthest, ends on the same 64/70 with one case flipping each
way, and v2d ends one case above the baseline. Those predictions were written
down before the runs and stay where they are, wrong.

The secondary metric `passed` returns exactly the same four tables as
`category_match`: on these 280 case-version outcomes, no case matched the gold
category and then failed the judge's score-3 threshold, so the judge added no
discrimination beyond the category comparison. That is a fact about this
dataset and this threshold, not a general property of the judge.

Both pre-specified sensitivity analyses (section 10.10) agree with the primary
result:

- dropping case-043, the case that sits verbatim in v1's own prompt, leaves
  H1 and H2 at b = 1, c = 1, RD = 0 on n = 69 (rows with
  `family = sensitivity_drop_leaked_case` in `e1_main.csv`);
- pairing every (case, repeat) instead of collapsing repeats, n = 210, gives
  the same picture: v2c -0.110, the other three within one flip of zero
  (`paired_mcnemar.csv`, `family = sensitivity_per_repeat_pairing`).

`e1_power.csv` answers what the 70-case gate can catch. Conditional on v2c's
observed effect, a study of this size rejects 55% of the time at alpha 0.05
and 32% at the 0.0125 Holm charges the first of four tests; at n = 50, 38% and
17%; at n = 30, 12% and 2%. For the other three versions the observed effect
is zero, so simulated power is zero at every n -- the resampling cannot invent
an effect that is not in the pairs. Each of those percentages is a proportion
of B = 2,000 resamples and carries simulation error of its own: sqrt(p(1-p)/B)
is at most 0.011 at this B, so power is read to two decimal places and no
further. The per-row value is the `mc_se` column of `e1_power.csv`.

The blunt version of the same arithmetic: with no cases flipping the other
way, an exact McNemar test at n = 70 needs 6 discordant pairs to clear a
nominal 0.05 (8.6 points of gate rate) and 8 to clear Holm's worst case (11.4
points). v2c had 10 flips down but also 2 up, and 10-versus-2 is not enough;
it would have needed 13 down against those 2. A 70-case gate of this shape
does not see a 5-point regression, and the interval widths in the table above
say so directly rather than by a non-significant p-value.

Exploratory, not part of any test: the two cases DEGRADATION_DESIGN named in
advance as the ones v2c should *improve* -- case-018, where v1's "SSO/2FA
setup" line pulls an SSO failure into `account`, and case-009, where v1's
tie-break example anchors a failed card charge to account access -- are
exactly the two that moved up. The 10 that moved down are mostly not the
at-risk cases the same document named: 8 of them are `easy` cases in the
`account` and `general` categories, while the ambiguous, currency and
export/sync cases it flagged largely held. Removing the definitions hurt the
cases nobody expected to be carrying them.

The design is frozen in
[`docs/PREREGISTRATION.md` section 10](../docs/PREREGISTRATION.md) and the
prompts in [`docs/DEGRADATION_DESIGN.md`](../docs/DEGRADATION_DESIGN.md).

## What E5 found

E5 re-reads the E0 and E1 raw files by stratum and calls no API. Unit: one
outcome per case, repeats collapsed by majority vote, so n counts emails.
Everything in it is exploratory -- the dataset was built to a quota, not
sampled -- and the tables say so in a `family` column.

The baseline arm, 64/70 overall, by stratum
(`results/tables/e5_strata.csv`, `figures/e5_forest.png`):

| Stratum | n | k | rate | 95% Wilson | width | Fisher vs rest | BH |
|---------|---|---|------|-----------|-------|----------------|-----|
| language zh-tw | 35 | 31 | 0.886 | 0.740-0.955 | 0.21 | 0.673 | 1.000 |
| language en | 28 | 26 | 0.929 | 0.774-0.980 | 0.21 | 1.000 | 1.000 |
| language mixed | 7 | 7 | 1.000 | 0.646-1.000 | 0.35 | 1.000 | 1.000 |
| difficulty easy | 45 | 44 | 0.978 | 0.884-0.996 | 0.11 | 0.020 | 0.092 |
| difficulty ambiguous | 17 | 16 | 0.941 | 0.730-0.990 | 0.26 | 1.000 | 1.000 |
| difficulty edge | 8 | 4 | 0.500 | 0.215-0.785 | 0.57 | 0.001 | **0.010** |
| category billing | 19 | 18 | 0.947 | 0.754-0.991 | 0.24 | 1.000 | 1.000 |
| category technical | 18 | 17 | 0.944 | 0.742-0.990 | 0.25 | 1.000 | 1.000 |
| category general | 17 | 13 | 0.765 | 0.527-0.904 | 0.38 | 0.028 | 0.092 |
| category account | 16 | 16 | 1.000 | 0.806-1.000 | 0.194 | 0.325 | 0.812 |

One stratum survives BH: `edge`, 4 of 8, against 60 of 62 everywhere else.
`general` and `easy` move at a nominal 0.05 and not after correction.

The widths are the point, and they are the reason the mixed-language row is
not a headline. `mixed` is 7 for 7, and its interval runs from 0.646 to 1.000:
this dataset cannot tell "the classifier never misses a code-switched email"
from "it misses a third of them". Those seven cases are also case-064 to
case-070, a consecutive block written in one sitting to fill the 10%
bilingual quota rather than seven draws spread across the dataset, which is
derived from the case ids at analysis time and printed in the table's `note`
column. The same holds, less dramatically, everywhere below n = 20: nine of
the ten strata have an interval wider than 19 points -- `account` at 0.194,
printed to three places above because it is the one that rounds onto the
threshold, is the narrowest of the nine, and `easy` at 0.112 is the only
stratum below it.

`e5_logit.csv` carries `pass ~ language + difficulty + category` with
standard errors clustered on case. **The pre-specified fit -- baseline arm
only -- does not exist**, and the file says so: `language[mixed]` is 35/35 and
`category[account]` is 80/80 in that arm, the maximum-likelihood coefficients
for those levels are unbounded, and Firth's penalised likelihood, the standard
remedy, has no validated implementation here (see the table below). The fit
that is reported pools every arm that ran (E0 plus the four E1 arms, 1,190
observations, 70 clusters), where no level has a constant outcome:

| Term | coefficient | cluster-robust SE | p | odds ratio | 95% CI |
|------|------------|-------------------|---|-----------|--------|
| const (zh-tw, easy, billing) | 3.981 | 0.982 | 0.00005 | 53.6 | 7.8-366.7 |
| language[en] | 0.079 | 0.521 | 0.880 | 1.08 | 0.39-3.01 |
| language[mixed] | 1.322 | 0.831 | 0.112 | 3.75 | 0.74-19.13 |
| difficulty[ambiguous] | -1.422 | 0.930 | 0.126 | 0.24 | 0.04-1.49 |
| difficulty[edge] | -2.718 | 0.681 | 0.00007 | 0.066 | 0.017-0.251 |
| category[technical] | 0.009 | 1.309 | 0.994 | 1.01 | 0.08-13.13 |
| category[general] | -2.100 | 0.787 | 0.008 | 0.122 | 0.026-0.572 |
| category[account] | -0.773 | 0.980 | 0.431 | 0.46 | 0.07-3.15 |

Difficulty is doing the work, and the two coefficients that move are the two
strata the Fisher rows flag. Language is not: `en` sits on the reference and
`mixed` is positive with an interval from 0.74 to 19, which is the same "we
cannot tell" as its Wilson row, in odds-ratio units.

The clustering is not decoration. Fitting the same model without it gives
standard errors 1.65x to 3.36x smaller (`const` 0.346 against 0.982,
`category[technical]` 0.389 against 1.309): 1,190 observations of 70 emails
carry roughly 70 emails' worth of information, and a naive fit would report
`category[general]` as more than twice as precise as it is.

That model pools five prompt versions, four of them degraded on purpose, and
has no version term -- a real mis-specification, so the sensitivity fit that
adds one is in the same file (`m3_pooled_plus_version_term`). It moves nothing
in the stratum coefficients and puts v2c at -1.333 (p = 0.039) with the other
three versions within 0.09 of zero, which is E1's answer arrived at a second
way. Both fits and the absent one are in `e5_logit.csv` with a `fitted`
column; the design choices E5 made after the data existed are listed in
`docs/PREREGISTRATION.md` section 9, dated 2026-09-16.

## What E2 found

E2 asks whether this judge's verdict survives swapping the two candidates. The
design is 70 pairs x 2 layers (baseline against the zero-shot degradation;
baseline against itself on a second repeat) x 2 orders x 2 judge models, and
all 560 calls are now in: 94 answered on 2026-09-15 before the account hit its
spend limit, and the remaining 466 on 2026-09-16, 466/466 successful. The two
batches are 21 hours apart; the `claude-sonnet-5` easy cell is the only one
that mixes them (94 calls from the first batch, 46 from the second) and the
other three cells are entirely from the second. What that allows and what it
does not is `docs/PREREGISTRATION.md` section 9, dated 2026-09-16: the design,
the prompt and the pair texts are unchanged and verified per cell, and
judge-side version drift across a day is disclosed rather than excluded,
because `response.model` returns the bare alias for Sonnet.

Order consistency -- the same summary source winning in both orders, a tie
counting as a verdict (`e2_consistency.csv`, `figures/e2_consistency.png`):

| Judge | Layer | consistent | rate | 95% Wilson |
|-------|-------|-----------|------|-----------|
| claude-haiku-4-5 | easy | 52/70 | 0.743 | 0.630-0.831 |
| claude-haiku-4-5 | hard | 52/70 | 0.743 | 0.630-0.831 |
| claude-sonnet-5 | easy | 58/70 | 0.829 | 0.724-0.899 |
| claude-sonnet-5 | hard | 59/70 | 0.843 | 0.740-0.910 |

**About one verdict in five changes when the two candidates change places**:
221 of 280 pairs held, 0.789 overall. Neither comparison the design was built
for separates. Between models the paired McNemar is b = 14, c = 27, p = 0.060
pooled across layers (BH 0.358) and further from any threshold within a layer
(easy p = 0.286, hard p = 0.167), so "Sonnet is the steadier judge" is what the
point estimates say and not what the test supports. Between layers there is
nothing at all: Fisher exact p = 1.000 for both judges (haiku 52/70 against
52/70, sonnet 58/70 against 59/70), and the paired sensitivity row, which
exists because the two layers are built from the same 70 emails, agrees
(b = 22, c = 23, p = 1.000 pooled). Comparing a summary with a degraded
summary is no more order-stable, on this dataset, than comparing a summary with
another draw of itself.

Position preference, ties excluded (`e2_position_pref.csv`):

| Judge | Layer | first position wins | rate | exact binomial vs 0.5 | BH | case-cluster 95% |
|-------|-------|--------------------|------|----------------------|----|------------------|
| claude-haiku-4-5 | easy | 51/94 | 0.543 | 0.470 | 0.941 | 0.495-0.596 |
| claude-haiku-4-5 | hard | 27/46 | 0.587 | 0.302 | 0.941 | 0.476-0.702 |
| claude-sonnet-5 | easy | 15/33 | 0.455 | 0.728 | 0.971 | 0.333-0.567 |
| claude-sonnet-5 | hard | 11/23 | 0.478 | 1.000 | 1.000 | 0.333-0.619 |
| both | both | 104/196 | 0.531 | 0.432 | -- | 0.482-0.580 |

**No cell shows a position effect**, and the two judges do not even lean the
same way -- Haiku above 0.5 in both layers, Sonnet below it in both. The
pre-registered test is the exact binomial (section 6); it treats calls as
independent and they are not, so the case-cluster bootstrap interval sits in
the same row and is the width to quote. Wang et al. (2024) report a large
position bias in reference-free pairwise judging; this prompt keeps the human
reference summary in front of the judge, as production does, and on 196
decisive calls there is no sign of one. That is a result about this setup, not
a contradiction of theirs.

The tie rate is what makes those denominators small, and it is the one number
that carried over from the partial run: 364 of 560 calls came back `tie`,
0.650 overall, and it is not evenly spread -- 0.329 for Haiku on the easy
layer, 0.836 for Sonnet on the hard one. Sonnet ties more than Haiku in both
layers, and both tie more when the two summaries are two draws of the same
prompt than when one of them is degraded, which is the direction a working
judge should move. The cost of it is arithmetic: a 560-call design bought 196
usable observations for the position question. Anything of this shape should be
budgeted at four calls per decisive answer.

Two controls worth their line. Four `hard` pairs have byte-identical summaries
(the classifier returned the same string on both repeats), which makes 16 calls
whose only correct answer is `tie` -- and all 16 came back `tie`, on both
judges. And on the easy layer, where the pair is the baseline summary against
the zero-shot one, the decisive consistent pairs split 24/37 to the baseline
under Haiku and 3/10 under Sonnet: small, descriptive, and one more reading of
E1's result that removing the four few-shot examples did not visibly damage the
output.

## E4 second annotator

A person outside the project labels the same 70 cases from the email text and
the four category definitions alone. What that person is given, and what is
deliberately withheld, is the measurement, so it is written down here rather
than left to whoever hands the files over.

The handout, both in `experiments/data/`:

| File | Contents |
|------|----------|
| `annotator2_sheet.csv` | UTF-8 with BOM, CRLF, header plus 70 rows, columns `case_id, email_body, your_label, notes`. The last two ship empty. |
| `annotator2_instructions_zh.md` | One page, Traditional Chinese. The four category definitions (`prompts/v1.yaml` lines 10-19) and the tie-break rule (lines 24-29) quoted verbatim, with a Chinese gloss marked non-authoritative. |

Withheld: `expected_category`, `expected_summary`, `expected_difficulty`,
`language`, the four few-shot examples, every model output and every E0/E1
number. The instructions carry no example email at all. `golden_dataset.json`
has no subject field, so the sheet has no `email_subject` column and
`email_body` is the whole email as the classifier saw it.

Row order is shuffled, so that position in the sheet says nothing about
difficulty or category. The shuffle is seeded and the sheet regenerates byte
for byte **from dataset v1**, the version it was built from; run against v1.1
the same snippet returns
`200aa335ca1a5eb9e1e76c9951511ea194b5a4d53659b60db7df62c754660f44` instead,
because `case-007`'s email text changed (`docs/PREREGISTRATION.md` section 9).
The handout is not regenerated -- reproducing it means running this against
the v1 file:

```python
import csv, json, random

cases = json.load(open("golden_dataset.json", encoding="utf-8"))["cases"]
rows = [{"case_id": c["id"], "email_body": c["input_text"]} for c in cases]
rows.sort(key=lambda r: r["case_id"])
random.Random(20260916).shuffle(rows)
with open("experiments/data/annotator2_sheet.csv", "w", encoding="utf-8-sig", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["case_id", "email_body", "your_label", "notes"])
    writer.writeheader()
    for row in rows:
        writer.writerow({**row, "your_label": "", "notes": ""})
```

Seed 20260916; sha256 of the sheet as handed out,
`c7abaf6fe77dc204549f31853ef348c1b370240737a0b1a55e4b758ea2a462ba`. Two of the
70 rows land on their `golden_dataset.json` index by chance, which is what
shuffling 70 items does and not a seed that failed to apply.

Ingestion, once a sheet comes back. Three people work from this one handout
(`docs/PREREGISTRATION.md` section 9, entry dated 2026-09-19), so the importer
takes the annotator as an argument and writes one file per annotator:

```bash
python -m experiments.import_annotations --annotator A1 \
    --sheet experiments/data/a1_category_labels_2026-09-19.csv \
    --annotated-on 2026-09-19
python -m experiments.import_annotations --annotator A2 \
    --sheet ~/Downloads/annotator2_sheet.csv --annotated-on 2026-09-23
```

`import_annotations.py` validates the whole file before it writes anything: 70
rows, every case id present exactly once, `your_label` one of the four values,
and `email_body` matching either `golden_dataset.json` or the text the sheet
was handed out with. Surrounding whitespace and capitalisation in the label are
normalised and the normalisation is printed; nothing else is repaired silently.
An edited `email_body` is an error rather than a warning, because such a row
carries a label for text the case does not contain. Every problem is listed in
one pass; exit 1 writes no file.

One row will come back not matching the dataset: `case-007`, whose email text
changed when the dataset moved to v1.1 after the handout
(`docs/PREREGISTRATION.md` section 9). `HANDOUT_BODY_SHA256` in the importer
holds the sha256 of that case's v1 text, so a row returning it is recognised as
the sheet as sent, reported as a warning and accepted; a `case-007` row
carrying anything else is still an error. The rejection message names both
faults it cannot tell apart -- an edited spreadsheet, or a dataset that moved
without the importer being told -- and `--allow-body-drift` remains the escape
hatch for the second. It was the only warning A1's import printed on
2026-09-19, and no flag was needed for it.

One flag is new and is off by default: `--allow-missing-cases` accepts a sheet
that is short a case and records which, because section 9's two-vote rule is
written for exactly that -- "that sheet omitted the case, or its entry failed
validation and no correction came back". Without it a short sheet is rejected
whole and goes back to its annotator, which is the normal path.

The importer reads `id` and `input_text` from the dataset and nothing else. No
gold label is printed or compared at ingestion time, so the decision to accept
or reject a returned sheet cannot be made after seeing how well it agrees.
Agreement is computed later, by the analysis step.

`experiments/data/annotations/<annotator>_labels.json` holds `annotator_id`
(`A1`, `A2` or `A3`), `annotator` (the role, e.g. `A1 (author)` -- no name is
recorded anywhere), `annotated_on`, `sheet_seed`, `sheet_sha256`,
`sheet_dataset_version`, `sheet_dataset_version_basis`,
`dataset_version_on_disk`, `missing_case_ids` and 70 `{case_id, label, notes}`
records sorted by case id. The version fields are separate on purpose: the
labels describe the text the annotator read, which is dataset v1, not whatever
the dataset says on the day of the import.

`sheet_dataset_version` is decided from the file before it is decided from any
row, because only one case differs between v1 and v1.1 and a v1 sheet with
that row pasted over would otherwise pass for a v1.1 one. In order: the
returned file's own sha256 against the handout's; the same hash with
`your_label` and `notes` blanked, which is what a filled-in handout reduces
to; then rows carrying text unique to v1, which also warns that the file is
not the handout with only the answer columns filled. When none of those
decides it, the field is `unknown` and a warning says so -- the importer does
not fall back to the version on disk, because that is a guess in exactly the
case where the guess is wrong. `sheet_dataset_version_basis` records which of
those four it was.

Status on 2026-09-19: A1, the author, returned a completed sheet, imported at
sha256 `774e62bc6c8337b188b87f249b66f1991196c6d616ef4d6d2e1cae1cde02cfbb` and
committed to `experiments/data/` exactly as received; A2 and A3 are due back
2026-09-23. The kappa estimator and its bootstrap interval run against A1 and
the model annotators today, and each further annotator's rows appear in the
same `e4_kappa.csv` the moment `experiments/data/annotations/<id>_labels.json`
exists. The three-rater rows -- Fleiss' kappa, Krippendorff's alpha and the
exact three-way agreement rate -- are written only once all three human sheets
are in, and are absent rather than approximated until then. The rank-flip check
-- does the system's ranking change when the gold labels are swapped for
another annotator's -- is still a `NotImplementedError` item in the table below.

**A1 against the shipped v1.1 labels, the one agreement number this study can
report before A2 and A3 return:** raw agreement 0.800 (56 of 70), Cohen's kappa
0.732, 95% bootstrap 0.600 to 0.846 (B = 10,000, seed 20260920). It is one of
the quantities section 9 lists in advance -- "each annotator against the
shipped v1.1 labels" -- and it changes no label: gold v2 needs a majority, and
`experiments/gold_v2.py` refuses to produce one from a single sheet. Read it
next to the two model rows below, where the same shipped labels score 0.905.
A1 is also the one annotator with a documented contamination: A1 read the
2026-07-20 review sheet two months earlier and ruled on its contested groups,
so A1 is blind to the label column and not to the dataset.

## What the E4 model annotators agreed with

Haiku 4.5 and Sonnet 5 each labelled the same 70 emails once, from the four
category definitions and the two-categories-fit rule alone: the material the
human second annotator gets, quoted verbatim from the same lines of
`prompts/v1.yaml` (`experiments/prompts/annotator_model_v1.yaml`, quotes
re-checked against the source at startup). No few-shot examples, nothing saying
a label already exists, one call per case. 140 calls, none failed.

| Rater A | Rater B | n | raw agreement | kappa | 95% bootstrap |
|---------|---------|---|---------------|-------|---------------|
| human-1 (gold) | claude-haiku-4-5 | 70 | 0.929 | 0.905 | 0.809-0.981 |
| human-1 (gold) | claude-sonnet-5 | 70 | 0.929 | 0.905 | 0.810-0.981 |
| claude-haiku-4-5 | claude-sonnet-5 | 70 | 0.971 | 0.962 | 0.902-1.000 |
| human-1 (gold) | A1 (author) | 70 | 0.800 | 0.732 | 0.600-0.846 |
| A1 (author) | claude-haiku-4-5 | 70 | 0.757 | 0.675 | 0.536-0.805 |
| A1 (author) | claude-sonnet-5 | 70 | 0.757 | 0.675 | 0.535-0.806 |

Five disagreements out of 70 for each model, and **the two models agree with
each other (0.962) more than either agrees with `human-1 (gold)` -- which is
not the independent human rater its name suggests.** Section 9's entry of
2026-09-19 records what that label set is: an agent drafted every case, the
author confirmed all 70 on 2026-07-20, and 69 of the 70 `expected_category`
values equal `draft_category`. So the top two rows compare a model against
labels a model drafted, and how much of the 0.905 that shared origin buys is
not estimable from this design -- the model identity of the drafting agent is
recorded nowhere here, so whether it was the same family as either rater is not
recoverable. Every row using that rater carries a note pointing at the entry.

The bottom three rows are the first measurement that does not have that
problem. A1 relabelled the same 70 emails blind from the same handout, and
scores 0.732 against the shipped labels and 0.675 against each model -- lower
than the models score against each other and lower than either scores against
the shipped labels. One blind human is not yet the check section 8.3 asks for,
and A1 is the annotator with the documented contamination; A2 and A3 are what
turns these three rows into a panel.

The intervals overlap almost exactly among the first two rows, so they cannot
be told apart at this n; the band names carried in the table are Landis & Koch
(1977), a reading convention quoted with its source rather than a result. All
of it is descriptive: `e4_kappa.csv` has no p-value column, because agreement
with one annotator is an estimate and not a test.

Exploratory, from the confusion tables rather than from any test: four of the
five disagreements are the same email labelled the same wrong way by both
models -- case-012 and case-043 (`general` read as `technical`), case-020
(`general` read as `billing`), case-018 (`technical` read as `account`). Two of
those four are cases this study had already written about before the arm ran.
case-043 is the leaked case of PREREGISTRATION section 10.10.1, whose label
sits verbatim in `prompts/v1.yaml`'s fourth few-shot example: shown the
definitions without that example, both raters put it somewhere else.
case-018 is the email `docs/DEGRADATION_DESIGN.md` named in advance as the one
v1's "SSO/2FA setup" line pulls into `account` -- and both raters, given that
same line in the definitions, put it in `account`. Of the three `general`
emails both models moved, `general` is also the category E5 found weakest for
the classifier itself (13/17 in the baseline arm).

What this arm cannot say is in PREREGISTRATION section 8.3: kappa measures
agreement, not correctness, and two raters agreeing at 0.9 with one annotator
can also mean all three share a blind spot. That caveat is harder to apply here
than it looks, because the annotator in question is itself of model-drafted
origin. Three people are labelling these 70 emails blind to close that: A1's
sheet is in and is in the table above, A2's and A3's are due 2026-09-23, and
gold v2 is their per-case majority.

### How to read the three-rater rows, once they exist

`e4_kappa.csv` gains three rows when all three human sheets are in, all three
across the panel rather than between a pair:

* **Fleiss' kappa** -- mean within-case agreement corrected by the pooled
  category proportions. Cases one annotator skipped are dropped, and the count
  is in `n_excluded`.
* **Krippendorff's alpha (nominal)** -- the same question with missing values
  kept: a case one annotator skipped still contributes the pairs that do exist.
  It is in the table for that difference. Its `po` and `pe` columns are
  1 - observed disagreement and 1 - expected disagreement, so the `kappa`
  column still reads `(po - pe) / (1 - pe)`. No Landis & Koch band is quoted
  for it, because that convention was published for kappa.
* **Exact three-way agreement rate** -- the share of cases all three wrote the
  same label on, chance agreement not removed. Its `kappa` column is empty
  because the row is a raw rate, and the value is in `po`.

All three carry a percentile bootstrap over cases, the same interval as every
other row in the table, which reflects the sampling of these 70 emails and not
annotator variance. Both new estimators reproduce a printed worked example in
`python -m experiments.stats` -- Randolph (2005) for Fleiss, Krippendorff
(2011) for alpha -- on the same terms as every other estimator here.

### Gold v2, and what would make it

`experiments/gold_v2.py` is section 9's rule set and nothing else: the per-case
majority of the three, the author ruling on a three-way split from a sheet
carrying the email and the human labels alone, the two-vote rule for a case one
sheet is short, and the two fallbacks fixed in advance. Where the entry is
silent the script refuses and names the cases, because a rule chosen after the
sheets are open is what registering the protocol was for. Run today, with only
A1 back, it declines twice over: before 2026-09-27 the regime is not chosen
yet, and at the deadline with one sheet it is fallback 2, under which gold
stays at v1.1 and A1's sheet is a reliability check used for nothing else.

When a sealed gold v2 exists, `analyze.py` recomputes the E0, E1 and E5 tables
on it and writes each beside its pre-registered version as `*_gold_v2.csv`,
plus `gold_v2_diff.csv` listing what moved, row by row and column by column.
The v1.1 files are not opened for writing, so "the published numbers did not
change" is checkable by hashing them. Every gold v2 row carries the family
`sensitivity_gold_v2`, takes no multiplicity correction, leaves `p_holm` and
`p_bh` empty and says to read `p_raw`: the pre-registered v1.1 results stay the
confirmatory result of section 4 and the recomputation is a sensitivity
analysis. E2 is untouched -- it carries no `category_match`.

## What is implemented, and what is not

Implemented and self-checked against published worked examples
(`python -m experiments.stats`, 44/44, standard library only): Wilson
interval, Clopper-Pearson interval, exact binomial test, exact McNemar,
Fisher exact for an unpaired 2x2, case-level cluster bootstrap, Holm,
Benjamini-Hochberg, Cohen's kappa, Newcombe method 10 paired risk-difference
interval, conditional power by case resampling, Fleiss' kappa, Krippendorff's
alpha (nominal).

Fisher exact arrived with E5 on 2026-09-16, checked against Fisher's own
lady-tasting-tea table (3, 1, 1, 3): two-sided 34/70 and one-sided 17/70, both
recomputed in the self-check from `math.comb` along a path that never calls the
function being checked, so a rewrite of it cannot agree with itself and pass.
E5 needs an exact test rather than chi-square because two of its strata are
n = 7 and n = 8.

One estimator in this package is not standard library: the E5 logistic model
with case-clustered standard errors, which statsmodels fits. When statsmodels
is absent, `e5_logit.csv` is written without coefficients and says so in the
row, the same way a missing matplotlib costs the figures and not the tables.

`--cross-check` adds seven comparisons against scipy and, since 2026-09-19,
three against statsmodels' `fleiss_kappa` -- the only second implementation of
that coefficient on this machine, and a cross-check rather than the validation,
which is the printed Randolph tables that run with no third party installed.
The analysis venv `~/.venvs/lrg-exp` has both -- Python 3.14.4 with scipy
1.18.1, numpy 2.5.3, statsmodels 0.15.0 and matplotlib 3.11.2, all cp314 wheels
-- and the flag reports 54/54 there, exit 0. Tail of the 2026-09-16 run, whose
scipy rows are unchanged:

```
[PASS] scipy binomtest agrees  -> 0.34375000
[PASS] scipy Clopper-Pearson agrees  -> 0.443905,0.974789
[PASS] scipy fisher_exact agrees on (3, 1, 1, 3)  -> 0.485714286
[PASS] scipy fisher_exact agrees on (7, 0, 57, 6)  -> 1.000000000
[PASS] scipy fisher_exact agrees on (4, 4, 60, 2)  -> 0.001036258
[PASS] scipy fisher_exact agrees on (10, 9, 54, 0)  -> 0.000000952
[PASS] scipy Wilson agrees  -> 0.490162,0.943318
```

and the three rows added on 2026-09-19, after which the run ends `54/54 checks
passed`:

```
[PASS] statsmodels fleiss_kappa agrees on Randolph table 1  -> 0.333333333
[PASS] statsmodels fleiss_kappa agrees on Randolph table 2  -> -0.200000000
[PASS] statsmodels fleiss_kappa agrees on a 4-category table  -> 0.176470588
```

The repo's own `.venv` has neither scipy nor statsmodels, so the same flag
there reports 44/46 with both cross-check rows marked FAIL. That is what a
wrong interpreter looks like, not a property of this machine: the flag refuses
to skip a comparison it could not make, so an absent cross-check cannot be read
as a passing one.
`checks/experiments_acceptance.sh` prints the interpreter it picked and
`has_scipy=yes/no` on its first line for the same reason.

**Newcombe method 10, validation closed 2026-09-15.** This estimator shipped on
2026-09-14 with weaker validation than the rest, because the paper's printed
worked example was not reachable from this machine. It now reproduces that
example: Newcombe (1998) Table III, e=20, f=12, g=2, h=16 (n=50), printed
95% interval 0.0562 to 0.3292, computed 0.056156 to 0.329207. The same table's
uncorrected method 8 row, 0.0618 to 0.3242, is reproduced by
`continuity=False` as 0.061805 to 0.324162, which pins the continuity
correction as well as the formula. Both are asserted in
`python -m experiments.stats` at the 4 dp the paper prints, alongside a check
that transposing f and g fails those endpoints, so a sign error in the
e/f/g/h to a/b/c/d translation cannot pass silently. The earlier structural
invariants and the Monte-Carlo coverage study (`--coverage`: 0.9520 to 0.9941
against a nominal 0.95 across ten scenario/size cells at 20,000 replicates)
still run. Both entries, 2026-09-14 and 2026-09-15, are in
`docs/PREREGISTRATION.md` section 9.

Deliberately raising `NotImplementedError` instead of shipping an unvalidated
approximation:

| Item | Needed for | Due |
|------|-----------|-----|
| Firth penalised likelihood | E5's separated baseline-arm model | open |
| Rank-flip check: does the system ranking change under another annotator's labels | E4 | 2026-09-24 |

Firth is the remedy for the separation described under *What E5 found*.
statsmodels 0.15.0 does not have it and `firthlogist` is not installed, so the
row in `e5_logit.csv` says not run and carries no approximate coefficients. It
is listed with no due date because nothing in the study depends on it: the
model that is reported is fitted on data where no level is constant.

Closed on 2026-09-19: Fleiss' kappa and Krippendorff's alpha, both against a
printed worked example, for the three-annotator panel of PREREGISTRATION
section 9. Closed earlier: E2's run (all 560 cells, in two batches --
`runner_pairwise.py --resume-failed-from`), and E4's annotator ingestion and
kappa matrix (`import_annotations.py`, `table_e4_kappa`,
`table_e4_confusions`), which report the model raters and take each human's
labels as soon as that annotator's file exists. Closed on 2026-09-16 with E5: the
logistic model with case-clustered standard errors. Closed on 2026-09-15:
"Newcombe method 10 checked against the paper's printed example", against
Newcombe (1998) Table III as described above.

`e5_strata.csv`, `e5_logit.csv`, `e2_consistency.csv`, `e2_position_pref.csv`,
`e4_kappa.csv` and the two `e4_confusion_*.csv` carry rows since 2026-09-16;
`figures/e5_forest.png` and `figures/e2_consistency.png` are both drawn. The E2
figure was skipped while one of its four cells had data, which is the rule
described above doing its job rather than an error.
`paired_mcnemar.csv`, `e1_main.csv` and `e1_power.csv` carry rows since
2026-09-15; before a v2* run existed they were written with headers and no
data, and `figures/e1_forest.png` / `figures/e1_power.png` were skipped rather
than drawn empty. `paired_mcnemar.csv` is the per-(case, repeat) sensitivity
view; the confirmatory test, which collapses repeats to one outcome per case,
is `e1_main.csv`, where the four `confirmatory_holm` rows are the
pre-registered family and the two `sensitivity_drop_leaked_case` rows are
section 10.10.1 and carry no adjusted p-value.

## Honesty rules for anything built on this data

- No employer data, no client data, no production traffic. The dataset is
  fictional (NebulaDesk) and hand-written.
- No run is discarded for looking bad. If a run is discarded for a technical
  reason, the raw file and the reason stay in the repository.
- Status words do not get upgraded on the way to a document: "measured" means
  a number in `results/`, "simulated" means it came out of a model, "planned"
  means it has not run.
- **LLM calls cannot be seeded.** The reproducibility claim is exactly: *raw
  data published, analysis fully reproducible from that raw data.* Re-running
  the API does not reproduce the numbers, and nothing written from this study
  may say that it does.
- Every figure footnote carries n, the interval method and the analysis seed.
