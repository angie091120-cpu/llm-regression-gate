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
  analyze.py       raw JSONL -> CSV tables + PNG figures + MANIFEST.json (no API)
  stats.py         estimators, standard library only, self-checked
  probes.py        four API-behaviour probes quoted in COST_CALIBRATION.md
  data/            second annotator's labels (E4), added 2026-09-23
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
| `PRICE_CLAUDE_SONNET_5_INPUT=2.00` | export before a run on any checkout that predates D-005 | this branch was cut from a `main` whose `evalkit/cost.py` still had $3/$15; the runner aborts unless the effective price is $2/$10, whichever way it gets there |
| `PRICE_CLAUDE_SONNET_5_OUTPUT=10.00` | same | same |
| `LLM_CLASSIFIER_MODEL` / `LLM_JUDGE_MODEL` | optional | same contract as production `llm.py` |
| `EXP_PYTHON` | optional | interpreter used by `checks/experiments_acceptance.sh` |

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

# analysis (free, re-runnable)
python -m experiments.analyze --seed 20260920

# acceptance
bash checks/experiments_acceptance.sh
```

Useful flags: `--cases 10` or `--cases case-001,case-064` or `--cases @ids.txt`;
`--concurrency` (default 4); `--skip-judge`; `--classifier-temperature`;
`--max-cost-usd` (hard stop, default $1.00); `--temperature-transport`.

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
3. **Cost is recomputed from `usage`,** by `analyze.py`, from the raw files --
   not carried over from whatever the runner printed.

## Runs completed

| Date | exp_id | Design | Calls (ok/planned) | Cost | Raw file |
|------|--------|--------|--------------------|------|----------|
| 2026-09-14 | `smoke` | 10 cases x 1, cost calibration only | 20/20 | $0.059746 | `results/raw/smoke/smoke_2026-09-14.jsonl` |
| 2026-09-14 | `e0_noise` | E0 main arm: v1 x 70 cases x 5 repeats, classifier + judge | 700/700 | $2.080951 | `results/raw/e0_noise/e0_noise_20260914T115913Z.jsonl` |
| 2026-09-14 | `e0_judge_iso` | E0 judge-isolation arm: repeat 0 classifier output frozen, judge re-scores 4x | 279/280 | $1.078802 | `results/raw/e0_judge_iso/e0_judge_iso_20260914T120557Z.jsonl` |

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
`results/figures/noise_floor_vs_gate_thresholds.png`. Fleiss kappa and
Krippendorff alpha are still `NotImplementedError` items, so the two columns
reserved for them in the judge tables are empty and carry a note saying why --
no approximation is written there.

Still to run: E1, E2, E4, E5.

## What is implemented, and what is not

Implemented and self-checked against published worked examples and scipy
(`python -m experiments.stats --cross-check`, 19/19): Wilson interval,
Clopper-Pearson interval, exact binomial test, exact McNemar, case-level
cluster bootstrap, Holm, Benjamini-Hochberg, Cohen's kappa.

Deliberately raising `NotImplementedError` instead of shipping an unvalidated
approximation:

| Item | Needed for | Due |
|------|-----------|-----|
| Newcombe method 10 CI for a paired risk difference | E1 | 2026-09-21 |
| Fleiss kappa | E0 | 2026-09-19 |
| Charging failed-but-billed calls in `cost_for()` | cost accounting | before E1 |
| Krippendorff alpha | E0 / E4 | 2026-09-24 |
| Logistic model with case-clustered standard errors | E5 | 2026-09-22 |
| Power curve simulation (n = 30/50/70) | E1 figure | 2026-09-21 |
| Forest plot figure | E1 figure | 2026-09-21 |
| Pairwise judge prompt and position-bias analysis | E2 | 2026-09-23 |
| Second-annotator ingestion, kappa matrix, rank-flip check | E4 | 2026-09-24 |

`paired_mcnemar.csv` is generated with zero data rows until a v2* run exists;
its `ci_note` column states why the interval column is empty.

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
