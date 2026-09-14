# Cost calibration and API behaviour, measured 2026-09-14

Everything below comes from two artifacts in this repository, not from an
estimate or a documentation page:

- `experiments/results/raw/smoke/smoke_2026-09-14.jsonl` -- 20 calls
  (10 cases x classifier + judge), 20/20 successful
- `experiments/results/probes/probes_sdk0.117.0_*.json` and
  `probes_sdk1.5.0_*.json` -- 12 calls probing parameter and version behaviour

Prices used throughout: Haiku 4.5 $1.00 / $5.00 and Sonnet 5 $2.00 / $10.00
per 1M input/output tokens. `experiments/runner.py` refuses to start if the
effective price table differs. The measurements below were taken on a
checkout whose `evalkit/cost.py` still carried the pre-D-005 Sonnet price of
$3/$15, so they were produced with the `PRICE_CLAUDE_SONNET_5_*` overrides
exported; on a checkout that already has D-005 the overrides are redundant
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

E2 is the one line with an assumption rather than a measurement: the pairwise
judge prompt does not exist yet, so it is priced at the measured judge input
(1,101 tokens) plus ~150 tokens for the second summary and the tie
instruction, at each model's rate. Sonnet $0.00416/call, Haiku $0.00208/call.
Re-measure after the prompt is written.

Against the $20 workspace spend limit:

- Projected full design: **$10.38 (52% of the cap)**
- Spent so far (this session): **$0.0723**
- Remaining headroom after the full design: **~$9.5**
- With a 30% contingency for re-runs and failed batches: $13.5, 68% of the cap

Lever if the budget tightens: E1's primary metric is `category_match`, which
needs no judge call. Running E1 with `--skip-judge` costs $1.78 instead of
$5.02 and loses only the secondary metric.

**Projection vs. outcome, E0 (2026-09-14).** Main arm projected $2.09, actual
$2.080951 (-0.4%). Judge-isolation arm projected $1.08, actual $1.078802
(-0.1%). The per-call figures in section 1 hold at 35x the sample they were
measured on, so the remaining projections are treated as good to about a
percent -- except E2, which is still priced off a prompt that does not exist.

## 3. Spend to date

| Item | Calls | Cost |
|------|-------|------|
| smoke run (`smoke_2026-09-14.jsonl`) | 20 | $0.059746 |
| probes, SDK 0.117.0 | 6 | $0.006922 |
| probes, SDK 1.5.0 | 6 | $0.005672 |
| first probe script iteration, artifact deleted | 4 (2 successful) | $0.004692 |
| E0 main arm (`e0_noise_20260914T115913Z.jsonl`) | 700 | $2.080951 |
| E0 judge-isolation arm (`e0_judge_iso_20260914T120557Z.jsonl`) | 280 (279 successful) | $1.078802 |
| **Total** | 1,016 | **$3.236785** |

The fourth row has no artifact in the repository: it was the first version of
`experiments/probes.py`, whose output file was deleted when the script was
rewritten to record the SDK version. The money was spent, so it is listed.
`experiments/results/cost_ledger.json` therefore shows $3.219499 (runner runs
only) while `MANIFEST.json` totals $3.232093 (runner + surviving probe files);
the $0.004692 gap is that deleted probe file.

**One call is billed but recorded at $0.** `runner.cost_for()` returns 0 for
any row with `ok: false`, and the judge-isolation arm has one such row
(`case-003`, repeat 3): the request reached the API, returned
`stop_reason: tool_use` and 1,144 input / 305 output tokens, and then failed
Pydantic validation because the returned tool input had no `score` field. At
the pinned Sonnet 5 price that call cost $0.005338, so both the ledger and the
MANIFEST understate real spend by that much. Recorded here rather than
back-filled into the raw row, which stays exactly as the runner wrote it;
charging failed-but-billed calls is a runner change to make before E1, where
1,680 calls give the same bug more room.

Not in the table above and not in this package's ledger: the production
baseline bootstrap run on `main` (70 cases, 140 calls,
`eval_reports/baseline.json`, commit fe1cea8) cost $0.415487 through
`evalkit.cost`. It is listed here only so the workspace-level total is
findable in one place: **$3.652272 against the $20 spend limit (18%).**

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
