# Cost calibration and API behaviour, measured 2026-09-14

Everything below comes from two artifacts in this repository, not from an
estimate or a documentation page:

- `experiments/results/raw/smoke/smoke_2026-09-14.jsonl` -- 20 calls
  (10 cases x classifier + judge), 20/20 successful
- `experiments/results/probes/probes_sdk0.117.0_*.json` and
  `probes_sdk1.5.0_*.json` -- 12 calls probing parameter and version behaviour

One provenance quirk in the smoke run's metadata:
`smoke_2026-09-14.meta.json` records `dataset` as an absolute path under this
machine's home directory, and it is the only meta file in the repository that
does. The smoke run predates
`runner.py`'s `_repo_relative()` helper -- it predates the file being
committed at all, which is why its `git_commit` points at a commit with no
`experiments/runner.py` in it -- and every run after it writes repo-relative
paths. The meta file is left as written: raw data and its metadata are
evidence and are not back-filled (`docs/PREREGISTRATION.md` section 9,
2026-09-14 cost entry).

Prices used throughout: Haiku 4.5 $1.00 / $5.00 and Sonnet 5 $2.00 / $10.00
per 1M input/output tokens. `experiments/runner.py` refuses to start if the
effective price table differs. The measurements below were taken on a
checkout whose `evalkit/cost.py` still carried the pre-D-010 Sonnet price of
$3/$15, so they were produced with the `PRICE_CLAUDE_SONNET_5_*` overrides
exported; on a checkout that already has D-010 the overrides are redundant
and the guard passes either way.

## 1. Measured per-call cost

| Tier | Model requested | `response.model` | n | input tokens (mean / median / range) | output tokens (mean / median / range) | latency (median / max) | $ per call |
|------|-----------------|------------------|---|--------------------------------------|---------------------------------------|------------------------|------------|
| classifier | `claude-haiku-4-5` | `claude-haiku-4-5-20251001` | 10 | 1689.4 / 1685.5 / 1666-1716 | 84.8 / 88.5 / 70-99 | 1470 ms / 1637 ms | **$0.002113** |
| judge | `claude-sonnet-5` | `claude-sonnet-5` | 10 | 1100.6 / 1099.5 / 1052-1152 | 166.0 / 163.5 / 89-286 | 2506 ms / 3764 ms | **$0.003861** |

One evaluated case = one classifier call + one judge call = **$0.005975**.

The classifier's ~1,690 input tokens are almost entirely the v1 system prompt
plus four few-shot examples; the per-email text is a small fraction. The
degraded E1 versions cut few-shot examples, so their input cost is lower than
the baseline, which makes the projection below an upper bound.

Wall clock: 20 calls at concurrency 4 finished in 13.5 s.

## 2. Projected cost of the full design

| Experiment | Units | Unit price | Projected |
|------------|-------|-----------|-----------|
| E0 main arm (70 cases x 5 repeats, classifier + judge) | 350 cases | $0.005975 | $2.09 |
| E0 judge-isolation arm (70 x 4, judge only) | 280 calls | $0.003861 | $1.08 |
| E1 (4 degraded versions x 70 x 3, classifier + judge) | 840 cases | $0.005975 | $5.02 |
| E2 pairwise judge, 560 calls, half Sonnet half Haiku | 280 + 280 | see note | $1.75 |
| E4 model-as-annotator (Haiku 70 + Sonnet 70) | 140 calls | $0.002113 / $0.004228 | $0.44 |
| E5 stratified re-analysis | 0 calls | - | $0.00 |
| **Total (3,360 calls)** | | | **$10.38** |

**E2 re-measured, 2026-09-15.** That row was the one assumption in the table:
the pairwise prompt did not exist, so it was priced at the measured judge input
plus ~150 tokens, Sonnet $0.00416 and Haiku $0.00208 per call. The prompt now
exists and a 16-call calibration run (`e2_smoke`) prices it:

| Judge model | n | input tokens / call | output tokens / call | measured $/call | projected $ for its 280 calls |
|-------------|---|--------------------|----------------------|-----------------|-------------------------------|
| `claude-sonnet-5` | 8 | 1276.5 | 225.2 | $0.004805 | $1.35 |
| `claude-haiku-4-5` | 8 | 1226.8 | 342.1 | $0.002937 | $0.82 |

**$2.17 for the 560 calls, against $1.75 projected: 24% over.** Input was
close; output was not. The assumption implicitly priced a short verdict, and
the free-text `reasoning` field runs 225 tokens on Sonnet and 342 on Haiku --
Haiku writes half again as much prose as Sonnet for the same question, which
is why the cheaper model is only 39% cheaper here instead of 50%. The prompt
was not shortened to recover the difference: it is the instrument, and
trimming the judge's reasoning to hit a cost estimate would change what E2
measures.

Against the $20 workspace spend limit:

- Projected full design: **$10.38 (52% of the cap)**
- Spent so far (this session): **$0.0723**
- Remaining headroom after the full design: **~$9.5**
- With a 30% contingency for re-runs and failed batches: $13.5, 68% of the cap

**That arithmetic was wrong about which cap binds, and section 3 records how
it was found out**: the account stopped serving this key on 2026-09-15 with
$8.58 spent inside this package. The $20 figure is a number from the study
plan, not a number this package can read; the API console is the only source
of truth for it, and whatever it says, it is not the constraint the E2 run
met.

Lever if the budget tightens: E1's primary metric is `category_match`, which
needs no judge call. Running E1 with `--skip-judge` costs $1.78 instead of
$5.02 and loses only the secondary metric.

**Projection vs. outcome, E0 (2026-09-14).** Main arm projected $2.09, actual
$2.080951 (-0.4%). Judge-isolation arm projected $1.08, actual $1.078802
(-0.1%). The per-call figures in section 1 hold at 35x the sample they were
measured on, so the remaining projections are treated as good to about a
percent -- except E2, whose 24% overrun came from output tokens and is
measured above rather than projected.

## 3. Spend to date

| Item | Calls | Cost |
|------|-------|------|
| smoke run (`smoke_2026-09-14.jsonl`) | 20 | $0.059746 |
| probes, SDK 0.117.0 | 6 | $0.006922 |
| probes, SDK 1.5.0 | 6 | $0.005672 |
| first probe script iteration, artifact deleted | 4 (2 successful) | $0.004692 |
| E0 main arm (`e0_noise_20260914T115913Z.jsonl`) | 700 | $2.080951 |
| E0 judge-isolation arm (`e0_judge_iso_20260914T120557Z.jsonl`) | 280 (279 successful) | $1.084140 |
| E1 H1 (`e1_v2a_20260915T153333Z.jsonl`) | 420 | $1.226163 |
| E1 H2 (`e1_v2b_20260915T153711Z.jsonl`) | 420 | $1.185492 |
| E1 H3 (`e1_v2c_20260915T154039Z.jsonl`) | 420 | $1.189659 |
| E1 H4 (`e1_v2d_20260915T154407Z.jsonl`) | 420 | $1.237953 |
| E2 calibration (`e2_smoke_20260915T172942Z.jsonl`) | 16 | $0.061943 |
| E2 main run, stopped by the spend limit (`e2_pairwise_20260915T173047Z.jsonl`) | 560 (94 successful) | $0.443440 |
| probes, SDK 0.117.0, 2026-09-16 with the limit still in force | 6 (0 successful) | $0.000000 |
| probes, SDK 0.117.0, 2026-09-16 after the limit was raised | 6 (4 successful) | $0.006932 |
| E2 re-run of the refused cells (`e2_pairwise_20260916T145210Z.jsonl`) | 466 | $1.553728 |
| E4 model annotation (`e4_annot_20260916T145907Z.jsonl`) | 140 | $0.312869 |
| **Total** | 3,890 | **$10.460302** |

E1 cost $4.839267 for 1,680 calls, all four arms successful, which is
$0.002880 per call against E0 main arm's $0.002973. The four runs were
launched with `--max-cost-usd 6.00` each and the dearest used 20.6% of that
cap; `aborted_on_cost_cap` is false in all four meta files. Nothing in the E1 rows
separates recomputed cost from recorded cost, because no E1 call failed: the
$0.005338 gap below is still the single E0 call and nothing else.

The fourth row has no artifact in the repository: it was the first version of
`experiments/probes.py`, whose output file was deleted when the script was
rewritten to record the SDK version. The money was spent, so it is listed.

**One call was billed and recorded at $0; the accounting is now fixed, and the
E0 raw file is not.** The judge-isolation arm has one `ok: false` row
(`case-003`, repeat 3): the request reached the API, returned
`stop_reason: tool_use` and 1,144 input / 305 output tokens, and then failed
Pydantic validation because the returned tool input had no `score` field. At
the pinned Sonnet 5 price that call cost $0.005338. `runner.cost_for()`
returned $0 for it, because it returned $0 for every failed row.

What changed on 2026-09-14, before E1:

- `runner.cost_for()` now charges any call that reported token usage, whether
  or not the call succeeded. A request the API rejected outright still costs
  $0, because a 400 carries no usage.
- `analyze.py` recomputes every published cost from each row's `usage` object
  at the prices pinned in `experiments/__init__.py`, instead of summing the
  `cost_usd` field the runner wrote. So **the $0.005338 is now included**, in
  `MANIFEST.json`'s `raw_cost_usd`, without anything in the raw file changing.
- The E0 raw JSONL is **not** back-filled. It stays exactly as the runner
  wrote it, `cost_usd: 0.0` and all -- a raw file is evidence of what happened,
  not a working copy.
- `cost_ledger.json` is **not** back-filled either. It is append-only and each
  entry records what that run knew at the time, so it still reads $8.058766, the sum of its six runner entries.

Where each number now lives:

| Number | Value | Meaning |
|--------|-------|---------|
| `MANIFEST.totals.raw_cost_usd` | $10.436084 | recomputed from `usage` at pinned prices -- the published figure |
| `MANIFEST.totals.raw_cost_usd_recorded_by_runner` | $10.430746 | what the runner wrote into the raw files at run time |
| `MANIFEST.totals.raw_cost_usd_unrecorded_at_run_time` | $0.005338 | the difference, i.e. this one call, unchanged by the E2 and E4 runs |
| `cost_ledger.json` `cumulative_usd` | $10.430746 | append-only, runner runs only, never edited afterwards |
| `MANIFEST.totals.total_cost_usd` | $10.455610 | raw (recomputed) + the four surviving probe files |

`checks/experiments_acceptance.sh` C4 reconciles the first against the fourth
and needs them within 1%; the gap is 0.05%, and it is a number with a named
cause rather than a silent agreement. It shrinks as the study spends more,
because it is one fixed call's worth of money against a growing total.

**2026-09-15: the API account refused the rest of E2.** The main pairwise run
executed 94 of 560 calls and then took `400 invalid_request_error` on every
remaining one: "You have reached your specified API usage limits. You will
regain access on 2026-10-01 at 00:00 UTC." Three things follow for the
accounting, and one does not.

- The 466 refused calls cost nothing. A 400 carries no `usage`, so
  `cost_for()` charges $0 and the recomputed and recorded totals agree on this
  file at $0.443440 -- the same rule that charges a call which burns tokens and
  then fails validation leaves this one alone.
- The run is in the ledger and in the runs table like any other. It was not
  deleted and not re-run: E2 cannot be re-run until access returns.
- `e2_smoke`, three minutes earlier, went 16/16 across both judge models, so
  the cutoff is datable to within a few minutes and is not a property of the
  prompt, the models or the runner.

What does not follow is a number for how much of the limit this package used.
The message names no figure, `$8.586773` here plus $0.415487 for the
production baseline run on `main` is **$9.002260 of whatever the limit is**,
and any spend made with the same key outside this repository is invisible from
inside it. The plan's $20 is a planning figure; the console is the only place
the real one can be read.

**2026-09-16: the limit was raised, and the rest of E2 plus E4 ran.** Two probe
runs date the change from inside the repository. At 14:33:54 UTC six probe
calls came back 400 with the same "You have reached your specified API usage
limits" body as the E2 failures, costing nothing
(`probes_sdk0.117.0_20260916T143354Z.json`, `total_cost_usd` 0.0). At 14:39:23
UTC the same script got 200s from Haiku and the ordinary `temperature is
deprecated` 400 from Sonnet -- section 4.1 behaviour, not a limit
(`probes_sdk0.117.0_20260916T143923Z.json`, $0.006932). Both files are
committed; the refused one is kept because a probe that proves the account was
still locked is evidence, not a failed artifact.

What the two runs then cost, at concurrency 4:

| Run | Calls | Model split | $ per call | Total |
|-----|-------|-------------|-----------|-------|
| E2 re-run | 466 | 280 Haiku / 186 Sonnet | $0.002584 / $0.004463 | $1.553728 |
| E4 model annotation | 140 | 70 Haiku / 70 Sonnet | $0.001425 / $0.003044 | $0.312869 |

E2's pairwise calls are the dearest per call in the study apart from the E0
judge: the prompt carries an email, a reference summary and two candidate
summaries, and the judge writes a `reasoning` field (Haiku averaged 286 output
tokens, Sonnet 202). E4's are the cheapest, 33 output tokens on average,
because the tool returns one enum field and nothing else. Neither run aborted
on its cost cap ($3.00 and $1.00), neither used the cache, and no call failed,
so recomputed and recorded cost agree on both files.

Not in the table above and not in this package's ledger: the production
baseline bootstrap run on `main` (70 cases, 140 calls,
`eval_reports/baseline.json`, commit fe1cea8) cost $0.415487 through
`evalkit.cost`. It is listed here only so the total this repository can
account for is findable in one place: **$10.875789**. What share of the
account's limit that is remains unanswerable from inside the repository; the
limit was raised on 2026-09-16 by an account decision, and its new value is
not visible here either.

## 4. Measured API behaviour

### 4.1 Sonnet 5 rejects `temperature`; Haiku 4.5 accepts it

Sent two ways -- as an SDK keyword argument, and inside the request body via
`extra_body`, which bypasses the SDK's signature entirely:

| Model | transport | Result |
|-------|-----------|--------|
| Haiku 4.5 | SDK keyword | 200 OK |
| Haiku 4.5 | `extra_body` | 200 OK |
| Sonnet 5 | SDK keyword | **400** `invalid_request_error` |
| Sonnet 5 | `extra_body` | **400** `invalid_request_error` |

Verbatim error message, both transports:

```
Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error',
'message': '`temperature` is deprecated for this model.'}}
```

The rejection is server-side and model-specific, not an SDK limitation: the
same request succeeds on Haiku through both transports. Consequence for the
study: judge randomness can be measured (repeat the call) but not switched
off, and `--classifier-temperature` is usable on the classifier only.

Separately, the SDK matters for *how* the parameter is sent: `anthropic`
0.117.0 exposes `temperature`, `top_p` and `top_k` on `messages.create`;
1.5.0 has removed all three (`TypeError: Messages.create() got an unexpected
keyword argument 'temperature'`). `experiments/client.py` detects this and
falls back to `extra_body`, recording which transport was used on every row.

### 4.2 `response.model` resolves the alias for Haiku but not for Sonnet

| Requested | Returned |
|-----------|----------|
| `claude-haiku-4-5` | `claude-haiku-4-5-20251001` |
| `claude-sonnet-5` | `claude-sonnet-5` |

This weakens one of the study's own premises. Recording `response.model`
instead of the request string does pin the classifier snapshot, and it does
not pin the judge snapshot -- the API echoes the alias back. The write-up must
say that judge-side version drift is disclosed, not excluded. (Observed
2026-09-14 on both SDK versions; if Sonnet starts returning a dated id later,
re-measure before repeating this claim.)

### 4.3 Judge responses carry no thinking tokens

Sonnet 5 usage includes `output_tokens_details: {"thinking_tokens": 0}` and
the response contains a single `tool_use` block, no `thinking` block, on all
10 judge calls plus the probes. Haiku returns `output_tokens_details: null`.
So the judge's output-token count is billable answer tokens only, and the
$0.003861 figure needs no thinking-token correction.

Prompt caching was inactive on every call
(`cache_creation_input_tokens = 0`, `cache_read_input_tokens = 0`), which is
the expected state -- no `cache_control` block is set anywhere. Enabling it
for the ~1,690-token shared classifier prefix is a possible cost reduction for
E1 and is deliberately not used, so that the token counts in this study stay
comparable across runs.

## 5. Reproducing these numbers

```bash
export PRICE_CLAUDE_SONNET_5_INPUT=2.00 PRICE_CLAUDE_SONNET_5_OUTPUT=10.00
python -m experiments.runner --exp-id smoke --cases 10 --run-id <fresh-id>
python -m experiments.probes
python -m experiments.analyze --seed 20260920
```

`experiments/results/tables/tokens_latency_by_tier.csv` carries the per-tier
table above; `MANIFEST.json` carries the per-file sha256, line counts and the
cost recomputed from usage.
