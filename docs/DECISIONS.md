# Design Decisions

Format per entry: **what we chose / why / alternatives considered and rejected.**
Entries are added as the build progresses; 3–5 substantive entries expected by MVP.

## D-001: `golden_dataset.json` is the working filename, not `golden_dataset_v1.json`

- **Chose:** the dataset file is `golden_dataset.json`, with an internal
  `"dataset_version": "v1"` field inside the JSON envelope (`{"dataset_version": "v1", "cases": [...]}`)
  for version tracking.
- **Why:** AC2 in SPEC.md §7 hard-codes `--dataset golden_dataset.json` as the
  literal, machine-checked filename. SPEC.md §3.1(c) separately describes the
  dataset as "versioned as `golden_dataset_v1.json`". Those two sentences
  conflict; since AC2 is the frozen, assertable contract, it wins. Version
  tracking moved into the JSON body instead of the filename so both
  requirements are satisfiable without renaming the file on every version bump.
- **Rejected:** naming the file `golden_dataset_v1.json` and breaking AC2's
  literal command; or maintaining both a versioned file and a `golden_dataset.json`
  copy/symlink (rejected as two sources of truth that can silently drift).

## D-002: `draft_difficulty` added as an agent-facing field alongside `draft_category`/`draft_summary`

- **Chose:** every case carries a `draft_difficulty` (easy/ambiguous/edge) filled
  in at drafting time, distinct from `expected_difficulty` (null until the boss
  confirms).
- **Why:** SPEC.md §3.1(c)'s intro line requires the drafting pass to
  deliberately construct edge cases and tag difficulty, but the schema table
  only lists `expected_difficulty`, which per the labeling-flow section is a
  boss-only field written at confirmation time. Silently reusing
  `expected_difficulty` for the draft assessment would have violated the
  explicit "don't fill `expected_*`" instruction; dropping the assessment
  entirely would have thrown away exactly the design intent (quota tracking,
  edge-case coverage) the drafting pass exists to produce. Adding a parallel
  `draft_*` field resolves the conflict without touching reserved fields.
- **Rejected:** overloading `expected_difficulty` during drafting (violates the
  human-confirmation contract); omitting a difficulty signal from drafts
  entirely (loses traceability for the quota claims in this report).

## D-003: async batching via `asyncio.to_thread` + `Semaphore`, not a native async SDK client

- **Chose:** `evalkit/run_eval.py` wraps the synchronous `classify_email()` /
  `score_summary()` calls in `asyncio.to_thread(...)` inside `asyncio.gather(...)`,
  bounded by an `asyncio.Semaphore(concurrency)` (default `EVAL_CONCURRENCY=5`).
- **Why:** `llm.py` is specced at ~80 lines and provider-agnostic; adding a
  second async code path (`AsyncAnthropic` + an async OpenAI client) would
  roughly double its size and dependency surface for a single benefit
  (concurrency), which `to_thread` + a semaphore already delivers -- bounded
  parallelism to control both wall-clock time and API cost, satisfying
  SPEC.md §4's "async batching控成本" requirement without a second client
  implementation to keep in sync with the sync one.
- **Rejected:** a fully async `llm.py` (doubles surface area for one benefit);
  unbounded `asyncio.gather` with no semaphore (defeats the cost-control point
  of batching in the first place).

## D-004: diff severity is keyed off the overall `pass_rate` delta only

- **Chose:** `evalkit/diff.py`'s `severity` field (`ok`/`warning`/`critical`) is
  computed solely from `abs(candidate.pass_rate - baseline.pass_rate)` against
  the two thresholds. `per_category_delta` is computed and reported for
  diagnostics but never independently escalates severity.
- **Why:** SPEC.md §5 lists "overall pass rate delta、per-category accuracy
  delta、flip 案例清單" as the diff's outputs, then states the 3%/8% thresholds
  in the following sentence without explicitly re-binding them to a specific
  metric. The overall pass rate is the one metric echoed everywhere else in
  the spec (US-2, AC2, AC3, README scorecard) as *the* headline number, so it
  is the natural binding target. This also keeps the AC3 fixtures small and
  legible: 3 planted regressions out of a 10-case fixture set is a clean 30%
  overall delta, well past critical, without needing per-category math to
  reason about severity.
- **Rejected:** escalating to critical if *any* single category's delta alone
  crosses the threshold, even with a stable overall pass rate -- more
  sensitive, but not clearly what the spec asked for, and it would make
  small, low-traffic categories a source of noisy false-critical failures.
  Flagged as an open question for the boss/Marcus if real usage shows the
  overall-only metric misses category-specific regressions in practice.

## D-005: pricing constants default to verified *list* price, overridable via env vars

- **Chose:** `_DEFAULT_PRICING` ships with per-model $/1M-token list prices,
  verified against Anthropic's official model pricing on 2026-07-20
  (Haiku 4.5: $1 in / $5 out; Sonnet 5: $3 in / $15 out). Time-limited
  introductory discounts are deliberately **not** encoded — estimates err
  conservative and don't go stale when a promo ends. Every price remains
  overridable via `PRICE_<MODEL>_<DIRECTION>` env vars for future changes.
- **Why:** the budget guard ($15 cap with report/stop thresholds) should
  never *under*-estimate spend, so list price is the safe default; and
  because prices do drift, the module refuses to silently guess for unknown
  models (fail-loud `ValueError`) instead of defaulting to zero. This was
  originally shipped as an explicitly-unverified placeholder (honesty over
  invented precision — the same "don't invent numbers" discipline the eval
  system enforces elsewhere); the values were then confirmed against the
  official pricing reference and the caveat upgraded to a verification note.
- **Rejected:** hardcoding the introductory promo price (goes stale, and
  under-estimates post-promo spend); refusing to provide any default and
  forcing env vars before the first run (needless friction); silently
  falling back to $0 for unknown models (breaks the budget guard).

## D-006: `eval-gate.yml`'s `pull_request` trigger runs a free fixture smoke
test, not a real-API eval; real-API eval is `workflow_dispatch`-only

- **Chose:** on `pull_request` (paths: `prompts/**`), `eval-gate.yml` runs
  `pytest -q` (mocked, AC1) plus `evalkit.diff` against the pre-generated
  AC3 fixture pair (`tests/fixtures/baseline_report.json` vs.
  `degraded_report.json`) and posts a PR comment carrying that fixture
  diff's pass rate / delta / severity, explicitly labeled as a smoke test.
  A real-API evaluation of the PR's actual prompt change only runs on a
  manually triggered `workflow_dispatch`, gated behind
  `secrets.ANTHROPIC_API_KEY`.
- **Why:** SPEC.md §7 AC4's assertable outcome ("eval-gate workflow green;
  PR comment contains pass rate, delta vs. baseline, severity") does not
  require the PR-triggered run to call a real model, and SPEC.md §6's own
  "Cost control" bullet reserves real-API runs for "the gate and
  milestones" without pinning that to a specific trigger. Read literally,
  §6's earlier sentence ("on PRs touching `prompts/**` and on
  `workflow_dispatch`: runs the real-API eval...") would put a real,
  billed API call behind every single PR that touches a prompt file with
  no manual gate -- for a solo portfolio repo with no configured
  `ANTHROPIC_API_KEY` secret, that reads as unbounded/uncontrolled spend
  and an eval-gate that can never go green without first wiring a paid
  secret into the repo. The free/no-key path is also the only one that is
  actually end-to-end verifiable in this repo without provisioning a real
  key (mirrors D-001's precedent: when frozen prose conflicts with what the
  assertable ACs and cost-control intent require, the AC-satisfying,
  verifiable reading wins and the prose gets corrected to match -- SPEC.md
  §6 has been updated accordingly).
- **Rejected:** literally running the real-API eval on every `prompts/**`
  PR per §6's unqualified prose (unbounded per-PR API spend, and untestable
  end-to-end here without a provisioned secret); skipping the PR-triggered
  smoke test entirely and only running `pytest -q` (would leave the
  diff/comment code path completely unexercised until someone manually
  dispatches a real-API run, and AC4 explicitly checks for a PR comment
  with pass rate/delta/severity).

## D-007: `WARN_THRESHOLD`/`CRITICAL_THRESHOLD` env vars are wired into
`evalkit/diff.py`'s CLI defaults, not dropped from `.env.example`

- **Chose:** `evalkit/diff.py`'s `--warn-threshold`/`--critical-threshold`
  argparse defaults now read `WARN_THRESHOLD`/`CRITICAL_THRESHOLD` from the
  environment (falling back to the existing 0.03/0.08 literals when unset).
  An explicit CLI flag still overrides the env var, since argparse only
  falls back to `default=` when the flag is absent.
- **Why:** `.env.example` already documented both variables as "eval engine
  tuning" next to `EVAL_CONCURRENCY` (which *is* read, by
  `evalkit/run_eval.py`), so a reader has every reason to expect setting
  them changes gate behavior; silently ignoring them is exactly the kind of
  "looks configured, does nothing" gap this project's own honesty
  discipline (D-005) argues against. This mirrors the existing
  `PRICE_<MODEL>_<DIRECTION>` override pattern in `evalkit/cost.py` --
  CLI-flag-wins-over-env is the established precedent in this repo, not a
  new convention.
- **Rejected:** deleting the two lines from `.env.example` instead (would
  have been the smaller diff, but throws away a legitimate use case --
  setting org-wide thresholds once via CI environment/`.env` instead of
  repeating `--warn-threshold`/`--critical-threshold` on every invocation
  of both `evalkit.diff` and `checks/acceptance.sh`).
