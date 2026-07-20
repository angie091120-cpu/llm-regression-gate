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

## D-005: pricing constants in `evalkit/cost.py` are explicitly unverified placeholders

- **Chose:** `_DEFAULT_PRICING` ships with concrete per-model $/1M-token
  numbers so `cost_ledger.json` works out of the box, but the module docstring
  and every reference to these numbers (README, this file) state plainly that
  they are **not** a verified live price check and must be confirmed against
  Anthropic's pricing page (or overridden via `PRICE_<MODEL>_<DIRECTION>` env
  vars) before being used for real go/no-go budget decisions against the
  spec's $15 cap.
- **Why:** fabricating a precise-looking number and presenting it as fact
  would violate the project's own "don't invent numbers" discipline the whole
  system is built to enforce elsewhere (golden dataset labels, judge scores).
  A working default with a loud caveat is safer than either silently guessing
  or blocking development entirely until pricing is manually verified.
- **Rejected:** hardcoding pricing with no caveat (dishonest); refusing to
  provide any default and forcing every caller to set env vars before the
  first run (unnecessary friction for a number that's easy to override once
  verified).
