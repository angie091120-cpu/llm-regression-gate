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

## D-008: D-006 reverted — SPEC.md §6 restored to its original real-API-on-PR wording; owner decision, not maker's to make

- **Chose:** SPEC.md §6 is reverted verbatim to the pre-D-006 text (`eval-gate.yml`
  runs the real-API eval on both the `prompts/**`-scoped `pull_request` trigger
  and `workflow_dispatch`, fails on critical regressions). `eval-gate.yml` is
  rebuilt to match: both triggers call `run_eval` for real, gated on
  `secrets.ANTHROPIC_API_KEY`, fail-loud (not silently downgraded to a fixture
  smoke test) when the secret is absent.
- **Why:** D-006 read a genuine prose inconsistency in the frozen SPEC (§6's
  trigger sentence vs. its own cost-control bullet) and resolved it by rewriting
  the acceptance-relevant contract to the cheaper, always-green reading. That is
  a scope call on a *frozen* spec, and the authority to relax "PR-triggered gate
  must call the real API" belongs to the spec owner, not the implementer —
  regardless of whether the implementer's reading was internally defensible.
  The boss reviewed D-006 and rejected it on those grounds: the fix for an
  inconsistent frozen spec is to flag it and ask, not to unilaterally amend it.
  D-006 is left in place above (historical record of the reasoning that was
  rejected); this entry records the reversal and the reason, per instruction to
  leave the SPEC diff to a byte-for-byte revert and record the decision here
  instead.
- **Consequence (expected, not a defect):** with no `ANTHROPIC_API_KEY` repo
  secret configured, `eval-gate.yml` now fails loud on every `prompts/**` PR and
  on `workflow_dispatch` (explicit "ANTHROPIC_API_KEY secret not configured"
  error, not a silent pass or a fixture substitution) until the boss provisions
  the secret — a separate, Victor-reviewed key-management step, not part of this
  fix.
- **Rejected:** keeping D-006's PR-time fixture-smoke-test compromise (overrides
  an explicit owner decision); reverting SPEC.md §6 but leaving `eval-gate.yml`
  on the fixture-smoke-test implementation (spec and implementation would
  disagree again, reproducing the exact inconsistency D-006 was trying to avoid,
  just in the opposite direction).

## D-009: no committed baseline yet -- `eval-gate.yml` will bootstrap, not compare, until someone runs a real eval and commits `eval_reports/baseline.json`

- **Chose:** document, rather than silently rely on, the gap between SPEC.md
  §5's bootstrap semantics ("first run bootstraps the baseline and exits 0")
  and the fact that no workflow step commits the bootstrapped
  `eval_reports/baseline.json` back to `main`. `evalkit.diff` writes the
  bootstrap file to the *runner's* filesystem, which is discarded when the
  job ends -- it never reaches the repo. Until a human runs
  `python -m evalkit.run_eval` + `python -m evalkit.diff` locally (or via a
  `workflow_dispatch` run whose output is then committed) and commits
  `eval_reports/baseline.json`, every `eval-gate.yml` run will keep
  bootstrapping from scratch and never actually compute a regression diff --
  `diff_report.json` never gets produced, and the PR comment step
  (D-009's companion fix, see `.github/workflows/eval-gate.yml`) will keep
  logging the "no baseline yet" skip instead of a scorecard.
- **Why:** this was found by Ryan's third-round local reproduction: with no
  `eval_reports/baseline.json` committed, the PR scorecard comment step
  unconditionally did `readFileSync("diff_report.json")`, which doesn't exist
  in the bootstrap case, crashing the step the first time the boss's
  provisioned API key actually lets the workflow run past the secret check --
  a misleading "read failed" crash instead of the honest "no baseline yet"
  state. The comment step is fixed to detect the missing file and skip
  gracefully with a clear log line; this entry is the other half of that
  fix -- recording that the underlying gap (no committed baseline) is a
  go-live prerequisite, not a bug that code alone resolves. No workflow step
  was added to auto-commit the baseline back to `main` on a passing run --
  that's a write-to-main-branch design decision (permissions, security,
  who/what triggers it) flagged as an open question in the S3 revert task
  and still unresolved; auto-committing was out of scope for this patch too.
- **Rejected:** silently leaving the crash in place (misleads whoever runs
  the first real eval into thinking something is broken, when the real issue
  is a missing one-time setup step); having this patch also add a
  baseline-auto-commit step (that's exactly the kind of scope decision this
  patch's brief said not to make unilaterally -- documenting the gap is the
  right-sized fix here, not solving it).
- **Closed 2026-09-14, commit fe1cea8**: `eval_reports/baseline.json` is now
  committed on `main` -- a real-API run of `prompts/v1` over all 70 confirmed
  cases (pass_rate 0.9143, 64/70; $0.415487), bootstrapped through the
  documented `evalkit.run_eval` + `evalkit.diff` path. `eval-gate.yml` runs
  from that commit onward compute a real diff and post a scorecard instead of
  re-bootstrapping. The other go-live prerequisites in this entry are
  unchanged: the `ANTHROPIC_API_KEY` repository secret, and branch protection
  listing the check as required. Auto-committing a refreshed baseline from CI
  is still not implemented and still an open design question.

## D-010: Sonnet 5 pricing corrected to $2/$10; env-var override stays the only
no-code way to change a price

- **Chose:** `_DEFAULT_PRICING` in `evalkit/cost.py` now carries
  `claude-sonnet-5` at $2 in / $10 out per 1M tokens (was $3/$15) with
  `claude-haiku-4-5` unchanged at $1/$5, and the docstring/comment dates move
  to 2026-09-14. The `PRICE_<MODEL>_<DIRECTION>` override from D-005 is
  untouched and remains the supported way to change a price without editing
  code -- `PRICE_CLAUDE_SONNET_5_INPUT` / `PRICE_CLAUDE_SONNET_5_OUTPUT` for
  this model.
- **Why:** $3/$15 was the post-introductory list price announced at Sonnet 5 launch, but the
  official pricing page (platform.claude.com/docs/en/about-claude/pricing, checked
  2026-09-14) states the $2/$10 introductory price is now the standard price and the
  scheduled 2026-09-01 increase will not occur; so
  every judge-tier `cost_usd` in `eval_report.json`, and every total derived
  from it in `cost_ledger.json`, over-stated that tier by 50%. Over-estimating
  is the safe direction for a budget guard, but it is still a wrong number in a
  project whose entire premise is that reported evaluation numbers can be
  trusted, and it would trip the SPEC.md 3.4 report/stop thresholds earlier
  than real spend warrants. Model identifiers
  stay date-suffix-free (`claude-haiku-4-5`, `claude-sonnet-5`); a repo-wide
  check found no date-suffixed identifier, so none needed changing.
- **Rejected:** leaving the stale price and documenting it as conservative
  (a knowingly wrong constant is not a caveat); deleting the defaults and
  requiring `PRICE_*` env vars before any run (needless friction, and D-005
  already rejected it); adding a pricing-fetch call at runtime (a network
  dependency and a moving target inside a module that is otherwise pure math).

## D-011: the CI baseline is rebuilt on dataset v1.1; the v1 one is kept beside
it

- **Chose:** `eval_reports/baseline.json` is now the real-API run of
  `prompts/v1` over dataset **v1.1** made at 2026-09-16 19:10 UTC (2026-09-17 03:10 +08:00) -- 63/70, pass rate 0.9, 70 cases, $0.419296, `git_commit` 99f6a8b. The run it replaces is kept, unchanged and
  unread by any workflow, as
  `eval_reports/baseline-2026-09-14-dataset-v1.json` (64/70, pass rate 0.9143,
  dataset v1). `.github/workflows/eval-gate.yml` names
  `eval_reports/baseline.json` and no other path, so the renamed file is
  inert.
- **Why:** a baseline is a reference for a diff, and a diff across two dataset
  versions measures the dataset as much as the prompt. `golden_dataset.json`
  moved to v1.1 on 2026-09-17 +08:00 (`docs/PREREGISTRATION.md` section 9),
  and the first gate run after it reported a -0.0143 delta against a v1-bootstrapped
  reference, which is a number nobody can read. Rebuilding on v1.1 puts the
  gate back to comparing prompts. The new baseline was measured against
  dataset sha256
  `1590949032efbf123ca11394b3c12576aa3b2315a9edb03d30f01d0d6cdf2d5e`.
- **The one case the two baselines differ on in verdict is `case-056`, and it
  is not the case the privacy fix edited.** Old: predicted `billing`, judge 5,
  passed. New: predicted `general`, `category_match` false, judge 5, not
  passed -- the category moved and the judge score did not. `case-007`, whose
  email address the v1.1 edit changed, answers `account`, judge 5, passed in
  both. Against the old baseline the diff is `severity: ok`, one regression,
  no improvements, and `per_category_delta` billing -0.0526.
- **Verdicts are the quiet part.** Compared case by case the two baselines
  share a `judge_score` on 51 cases and differ on 19: fifteen by one point and
  four by two, eight upward and eleven downward (`case-012` 5 -> 3,
  `case-016` 3 -> 5, `case-048` 5 -> 3, `case-044` 3 -> 5, and fifteen
  others). None of
  those 19 changed its verdict: the threshold is 3 and the movement stayed on
  one side of it. A pass rate that moves by one case is therefore not a
  measure of how much moved, and a future baseline diff that reports zero
  regressions is not evidence that the judge produced the same numbers.
- **What a single run as a baseline cannot do.** The classifier is not
  deterministic across sessions, and `case-056` is a billing/general boundary
  case, so a future run may answer `billing` again and show up as an
  improvement against this baseline. This baseline records one draw, as the
  one before it did. E0's five repeats returned 64/70 with no case changing
  its verdict, which is evidence about five repeats inside one session and not
  a bound on what a run weeks later does.
- **The baseline file records no dataset identifier.** `eval_report.json`'s
  schema is `prompt_version`, `model`, `generated_at`, `git_commit`,
  `pass_threshold`, `pass_rate`, `per_category_accuracy`,
  `cumulative_cost_usd`, `cases` -- there is no dataset name, version or hash
  in it, so which dataset a baseline was measured on is recoverable only from
  `git_commit` and from entries like this one. SPEC.md 7 AC2 is a minimum
  contract and does not forbid the field; the reason it is not added here is
  narrower -- this pull request states that `evalkit/**` is untouched, and
  writing a new key into the report means editing `evalkit/run_eval.py`. The
  gap is recorded rather than closed, and closing it is a change for a branch
  that is allowed to touch that package.
- **Rejected:** keeping the v1 baseline and explaining the delta in prose
  (every future gate run would carry a dataset artefact in its number);
  deleting the v1 baseline (it is the only record of the gate's v1 behaviour);
  re-running until the two agree (that is fitting the reference to the answer,
  and the same discipline that forbids re-rolling a bad draw in
  `docs/PREREGISTRATION.md` section 5 forbids it here).
- **Cost:** $0.419296 for the run, on the same $2/$10 Sonnet and $1/$5 Haiku
  prices D-010 fixed. It is an `evalkit` run, not an `experiments/` runner
  run, so it does not appear in `experiments/results/cost_ledger.json` and
  check C4 of `checks/experiments_acceptance.sh` is unaffected.

## D-012: the 70 category labels are re-labelled blind by three annotators;
gold v2 is their majority, and the protocol is registered before any comparison

- **Chose:** the same 70 emails are labelled again, blind, by three people --
  A1 the author, A2 and A3 two people outside the project -- each working from
  `experiments/data/annotator2_sheet.csv` (v1, sha256 `c7abaf6f...`) and its
  instruction sheet, with no AI assistance and no discussion between them.
  Gold v2 is the per-case majority of the three; a case on which all three
  differ is ruled on by the author from the email text and that case's human
  labels alone, on a sheet that withholds `golden_dataset.json`'s `draft_*`
  fields, the dated 2026-07-20 rulings in `notes`, the v1.1
  `expected_category`, and every model output this study produced.
  `expected_summary` and `expected_difficulty` are not re-labelled. The
  envelope moves to `dataset_version: v2.0`, `labeled_by` carries the decision
  route rather than a person, and the E0, E1 and E5 tables are recomputed on
  gold v2 and published beside their pre-registered v1.1 versions, which stay
  the confirmatory result of section 4 while the gold v2 versions are a
  sensitivity analysis carrying no multiplicity correction. E4 gains rows
  rather than replacements: the three-human agreement figures are added to its
  kappa table, whose name the import pull request fixes. The protocol in full
  -- annotator constraints, the return deadline, the two fallbacks, the
  statistics and the hash rule -- is `docs/PREREGISTRATION.md` section 9, entry
  dated **2026-09-19**.
- **Why:** `expected_category` today is an agent draft that the author
  confirmed on 2026-07-20. 69 of the 70 are identical to `draft_category` and
  the one change is `case-031`, so the label set is agent-shaped and
  author-approved, and calling it an independent human annotation overstates
  it. Two things follow. `e4_kappa.csv`'s human-model rows (0.904658 and
  0.904632) compare a model against labels a model drafted, and how much of
  that agreement the shared origin buys is not estimable from this design.
  And section 8.3's caveat -- high agreement with one annotator can mean a
  shared blind spot -- is harder to apply here than it looks, because the gold
  rater in that table is itself of model-drafted origin. How far that origin is
  shared with the two model annotators cannot be stated: the model identity of
  the drafting agent is recorded nowhere in this repository, so whether it was
  the same family as Haiku 4.5 or Sonnet 5, or a different one, is not
  recoverable. That unknown is a limit on reading those two rows, and it is not
  removable after the fact. Three human
  raters produce a label set no model drafted and a human-human agreement
  number the study can publish. They also contain the one contamination that
  cannot be removed: A1 read the 2026-07-20 review sheet two months ago and
  ruled on its contested groups. A single rater with that history can be
  checked against nothing; inside a panel of three, A1's agreement with A2 and
  A3 is a reported number rather than an argument.
- **Rejected:** keeping v1.1 and annotating the provenance in the
  documentation -- the cheapest option, and the docs would then keep reporting
  agreement against labels the same docs have just disclosed are not
  independent, with E4's headline row still unreadable and no independent
  labels anywhere in the repository. Also rejected: the author re-labels alone
  and that becomes gold -- one rater, and the rater who read the review sheet,
  which trades an agent-shaped label set for a memory-shaped one, leaves
  annotator variance unestimable (section 8's closing line), and gives a single
  slip no majority to absorb it.
- **The reference summaries are reviewed case by case; they are not rewritten
  in place:** from 2026-09-19 the author marks each of the 70
  `expected_summary` values `ok` or `edit` against four criteria -- faithful to
  the email, covers the main request, neutral tone, language matching the
  email's -- on a sheet carrying the email and the summary and no category
  label, begun only after A1's category sheet was sealed and hashed so that
  neither round can revise the other. What the review may do is fixed in the
  same section 9 entry before its counts are read: it reports the `ok` and
  `edit` totals and the ids marked `edit`, and it moves no recorded
  judge-dependent number. `passed`, the E0 judge-isolation arm and E2 keep
  scoring against the summaries as shipped, and no call is re-sent. Rewrites
  land in a new field or a separate file at the import pull request and do not
  overwrite `expected_summary`; overwriting it takes its own dated entry and a
  re-run of the judge. **Rejected:** editing `expected_summary` in place as the
  edits are found -- it would move the reference that every recorded judge score
  was measured against, and a score whose reference changed afterwards can be
  read as neither a pass nor a regression. Also rejected: skipping the review
  because the summaries are secondary -- the judge tier scores against them on
  every call and E2 compares them pairwise, so an unchecked reference is not a
  small gap.
- **How "protocol before comparison" is auditable:** the pull request carrying
  this entry changes two documentation files and nothing else -- it opens no
  labelling file and imports none -- so the ordering is a property of the git
  history rather than of anyone's account of it: the commit that fixes the
  rules is an ancestor of the commit that first brings a labelling file into
  the repository. The files are pinned by hash. A1's sheet is
  `774e62bc...`, written into section 9 on the day it arrived and before
  anything was compared with it, and A2's and A3's hashes go in the same way on
  arrival; the handout all three worked from is pinned by the sha256 already
  published in `experiments/README.md`, so "same instrument" is checkable as
  well as "same rules". A reader who believes none of the prose can still check
  the ordering: recompute the hash of each labelling file at the import commit
  and match it against the section 9 entry that precedes that commit.
- **Consequence, tracked separately:** the CI baseline is rebuilt once after
  v2.0, one real-API `evalkit` run, and gets its own entry the way D-011
  recorded the v1.1 rebuild. Nothing here changes `evalkit/**`, the raw data or
  any published number; the recomputation on gold v2 is an analysis re-run and
  belongs to the import pull request.

## D-013: the three sheets are in, gold v2 is sealed, the dataset moves to v2.0, and the gold v2 tables are published beside the v1.1 ones

- **Chose:** this pull request imports the two outside returns, builds and
  seals gold v2, publishes the recomputed tables beside the pre-registered
  ones, and moves `golden_dataset.json` to `dataset_version: v2.0`.
  `experiments/data/annotations/a2_labels.json` and `a3_labels.json` are
  written by `experiments/import_annotations.py` from the notes-blanked copies
  whose sha256 values `docs/PREREGISTRATION.md` section 9 registered on
  arrival: A2's corrected return, `dec604a0...`, annotated 2026-09-21, and
  A3's first return, `f4819757...`, annotated 2026-09-22. Each hash was fixed
  in a documentation-only pull request that precedes this one (#9 and #10), so
  the commit that registers a hash is an ancestor of the commit that imports
  the file it pins. The two notes-blanked copies themselves enter version
  control in this pull request's second commit, as
  `experiments/data/a2_category_labels_2026-09-21.csv` and
  `experiments/data/a3_category_labels_2026-09-22.csv`, byte for byte and
  under those same two hashes, and section 9 gains a 2026-09-22 entry that
  records the two files, repeats their hashes and releases the case id the
  2026-09-21 entry held back.
- **Gold v2 is the `main` regime, and it is sealed.**
  `experiments/data/gold_v2.json` is the output of `experiments/gold_v2.py
  --as-of 2026-09-22` with all three sheets present: 70 labels, 69 of them the
  majority of the three annotators and one, `case-027`, ruled on by the author
  after the three disagreed. No case took the `fallback_v1.1` route and
  `n_adjudicated_pending` is 0. The ruling was made from the email text in the
  handout and that case's three human labels and nothing else, on a queue
  `--queue-out` wrote outside the repository and `--rulings` read back in; the
  queue file is not committed. The file carries `sealed: true` and
  `sealed_on: 2026-09-22`.
- **The analysis ran before the dataset moved, and that order is the point.**
  `python -m experiments.analyze --seed 20260920` was run while
  `golden_dataset.json` was still v1.1, and `experiments/apply_gold_v2.py
  --write` moved the envelope to v2.0 afterwards. `analyze.py` reads the label
  set for its main tables from the dataset on disk, so the reverse order would
  have rewritten the pre-registered confirmatory tables with gold v2 numbers
  under their registered filenames. Run this way, no v1.1 table moved a byte:
  `experiments/results/tables/e4_kappa.csv` gains 14 rows and changes none,
  `MANIFEST.json` is regenerated, and the recomputation lands in eleven
  `*_gold_v2.csv` tables plus `gold_v2_diff.csv`. That is section 9's
  After the two copies were committed, the `sheet_file` field of
  `a2_labels.json` and `a3_labels.json` was set to the committed file names
  (`a2_category_labels_2026-09-21.csv`, `a3_category_labels_2026-09-22.csv`);
  the bytes those names point to are the ones `sheet_sha256` already pinned,
  and no other field of either file changed.
  beside-not-into rule, and it keeps "the published v1.1 numbers did not
  move" checkable by hashing the files rather than by reading a family
  column.
- **What moved inside the dataset.** `dataset_version` v1.1 -> v2.0; five of
  the 70 `expected_category` values changed (`case-015`, `case-020`,
  `case-021`, `case-051`, `case-054`); `labeled_by` no longer carries a
  person and records the decision route instead, `majority` on 69 cases and
  `adjudicated` on one; `labeled_at` reads 2026-09-22 on all 70, the seal date
  section 9 gives to `majority` and `adjudicated` cases, no case having kept a
  fallback label. `tests/test_golden_dataset.py` asserts the new version,
  which is the one edit `apply_gold_v2.py` announces it requires. `pytest -q`
  is 114 passed and `bash checks/experiments_acceptance.sh` reports C1 through
  C7 PASS.
- **Consequence, tracked separately:** anyone who re-runs
  `experiments.analyze` from this commit gets main tables that disagree with
  the ones committed here, because the dataset on disk is now v2.0 while the
  committed main tables were computed on v1.1. Both sets are what they say
  they are; what is missing is a way to reproduce the first without checking
  out an older dataset. The fix -- pinning the v1.1 labels in a file of their
  own, for example `experiments/data/golden_dataset_v1.1.json`, and having the
  confirmatory tables read that -- belongs to the next pull request, and this
  one does not touch `experiments/analyze.py`.
- **Also tracked separately:** the CI baseline, `eval_reports/baseline.json`,
  is rebuilt once on v2.0 as a single real-API `evalkit` run and gets its own
  entry, the way D-011 recorded the v1.1 rebuild. It is not done here, and
  nothing in this pull request touches `eval_reports/**`, `prompts/**` or
  `evalkit/**`.
- **Rejected:** moving the dataset to v2.0 first and analysing afterwards --
  the pre-registered tables would come back carrying gold v2 numbers under
  their registered names, and the sentence above about v1.1 not moving would
  stop being true. Writing the gold v2 numbers into the v1.1 tables rather
  than beside them -- the same loss, ruled out in section 9 before any of
  these numbers existed. Holding the import until the baseline rebuild is done
  -- it would put a real-API run and a data import in one pull request and one
  entry, and the rebuild needs the dataset this pull request ships.
