"""E2 runner: pairwise judge calls, one JSONL line per call.

Separate from `runner.py` because the unit is different. `runner.py` measures
one case at a time (classifier call, then judge call); E2 measures one *pair*
of already-produced summaries at a time and makes no classifier call at all.
Everything the two share -- the .env loader, the price guard, the raw-row
contract, the cost cap, the ledger -- is imported from `runner.py` rather than
copied, so a change to the raw-data contract cannot apply to one runner and
not the other.

What it builds, fixed by docs/PREREGISTRATION.md section 5 and the E2 row of
the sprint plan:

  Easy layer   the same email summarised by the baseline prompt (E0 repeat 0)
               and by v2b, the zero-shot degradation (E1 repeat 0)
  Hard layer   the same email summarised twice by the same baseline prompt
               (E0 repeat 0 and repeat 1) -- two draws from one distribution

  70 cases x 2 layers x 2 orders x 2 judge models = 560 calls

"Order" is the whole point: every pair is judged twice, once with the baseline
summary in position A and once with it in position B. A judge with no position
preference returns the same verdict both times. `position_a_source` and
`position_b_source` travel on every row, so the analysis reads which *source*
won without having to know which order it was in.

    export PRICE_CLAUDE_SONNET_5_INPUT=2.00 PRICE_CLAUDE_SONNET_5_OUTPUT=10.00

    python -m experiments.runner_pairwise --exp-id e2_pairwise \
        --baseline-from experiments/results/raw/e0_noise/<run_id>.jsonl \
        --contrast-from experiments/results/raw/e1_v2b/<run_id>.jsonl \
        --dry-run

    python -m experiments.runner_pairwise --exp-id e2_pairwise \
        --baseline-from experiments/results/raw/e0_noise/<run_id>.jsonl \
        --contrast-from experiments/results/raw/e1_v2b/<run_id>.jsonl \
        --max-cost-usd 3.00

--resume-failed-from <jsonl> sends only the design cells that an earlier run of
this experiment recorded as failures. The 2026-09-15 run wrote all 560 rows and
the API answered 94 of them before the account hit its spend limit, so the
re-run is 466 calls and not 560: re-asking a question that already has an answer
spends money twice and leaves two successful rows in one cell, which the
analysis refuses to resolve. The new rows go to a new raw file -- the old one is
evidence and is never appended to or rewritten -- and analyze.py reads both,
keeping the answered row of each cell.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evalkit.cost import record_usage  # noqa: E402
from evalkit.dataset import GoldenCase, load_confirmed_cases  # noqa: E402
from experiments.client import CallRecord, call_structured, sdk_version  # noqa: E402
from experiments.runner import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_CONCURRENCY,
    DEFAULT_DATASET,
    DEFAULT_LEDGER,
    DEFAULT_RAW_DIR,
    RunState,
    _repo_relative,
    assert_prices_current,
    build_row,
    git_commit,
    load_dotenv,
    now_iso,
    select_cases,
)

PROMPT_PATH = REPO_ROOT / "experiments" / "prompts" / "judge_pairwise_v1.yaml"
DEFAULT_JUDGE_MODELS = ("claude-sonnet-5", "claude-haiku-4-5")
LAYERS = ("easy", "hard")
ORDERS = ("left_first", "right_first")
TIER = "judge_pairwise"


class PairwiseVerdict(BaseModel):
    """Structured output for the pairwise judge -- same mechanism as the
    production judge (tool-use with a Pydantic schema, `tool_choice` pinned to
    this tool), different fields. Nothing is parsed out of free text."""

    choice: Literal["A", "B", "tie"] = Field(description="Which candidate is closer to the reference summary")
    reasoning: str


class PairwisePrompt:
    def __init__(self, path: Path) -> None:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        missing = [k for k in ("version", "system_prompt", "user_template", "tool_name") if k not in data]
        if missing:
            raise SystemExit(f"{path} is missing required keys: {missing}")
        if data["tool_name"] != PairwiseVerdict.__name__:
            raise SystemExit(
                f"{path} names tool {data['tool_name']!r} but the schema this runner sends is "
                f"{PairwiseVerdict.__name__!r}; the two must agree"
            )
        self.version: str = data["version"]
        self.system_prompt: str = data["system_prompt"]
        self.user_template: str = data["user_template"]

    def render(self, *, input_text: str, expected_summary: str, summary_a: str, summary_b: str) -> str:
        """Substitution by replace, not str.format: an email that contains a
        brace -- a JSON snippet pasted into a support ticket, say -- would make
        format() raise or swallow it, and the email text is data."""
        out = self.user_template
        for key, value in (
            ("input_text", input_text),
            ("expected_summary", expected_summary),
            ("summary_a", summary_a),
            ("summary_b", summary_b),
        ):
            placeholder = "{" + key + "}"
            if placeholder not in out:
                raise SystemExit(f"{PROMPT_PATH} user_template has no {placeholder} placeholder")
            out = out.replace(placeholder, value)
        return out


def load_summaries(path: Path, repeat_idx: int) -> dict[str, dict]:
    """Successful classifier summaries from one repeat of a previous run.

    A failed classifier call contributes no summary, so the pair that would
    have used it is dropped and counted -- never replaced by an empty string,
    which the judge would happily compare.
    """
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("tier") != "classifier" or not row.get("ok") or row.get("repeat_idx") != repeat_idx:
            continue
        summary = (row.get("parsed") or {}).get("summary")
        if not summary:
            continue
        out[row["case_id"]] = {
            "summary": summary,
            "category": (row.get("parsed") or {}).get("category"),
            "source_run_id": row.get("run_id"),
            "source_call_id": row.get("call_id"),
            "source_exp_id": row.get("exp_id"),
            "source_repeat_idx": row.get("repeat_idx"),
            "source_prompt_version": row.get("prompt_version"),
        }
    if not out:
        raise SystemExit(f"{path} has no successful classifier rows at repeat_idx={repeat_idx}")
    return out


def build_pairs(cases: list[GoldenCase], sources: dict[str, dict[str, dict]]) -> tuple[list[dict], list[str]]:
    """One entry per (case, layer). `left` is always the baseline summary, so
    "did the verdict survive the swap" is a question about the same two texts
    in both orders.
    """
    pairs: list[dict] = []
    skipped: list[str] = []
    for layer in LAYERS:
        left_key, right_key = ("baseline_r0", "contrast_r0") if layer == "easy" else ("baseline_r0", "baseline_r1")
        for case in cases:
            left = sources[left_key].get(case.id)
            right = sources[right_key].get(case.id)
            if left is None or right is None:
                skipped.append(f"{layer}:{case.id}")
                continue
            pairs.append({
                "pair_id": f"{layer}:{case.id}",
                "layer": layer,
                "case": case,
                "left_source": left_key,
                "right_source": right_key,
                "left": left,
                "right": right,
                "summaries_identical": left["summary"] == right["summary"],
            })
    return pairs, skipped


RESUME_KEY_FIELDS = ("request_model", "layer", "order", "pair_id")


def cell_key(request_model: str, layer: str, order: str, pair_id: str) -> tuple[str, str, str, str]:
    """The unit a re-run resumes on: one judge model asked about one pair in
    one order. analyze.py groups the raw rows by the same four fields."""
    return (str(request_model), str(layer), str(order), str(pair_id))


def load_resume_targets(path: Path, *, exp_id: str, prompt_version: str) -> tuple[dict[tuple, dict], dict]:
    """Cells of an earlier run that failed and still have no answer.

    Cells that succeeded are subtracted, so a resumed batch cannot produce a
    second successful row for a cell that already has one -- the case
    analyze.py stops on. Two properties of the earlier file are checked rather
    than assumed, because getting either wrong merges two different
    experiments without saying so: the same exp_id (the analysis joins on it)
    and the same judge prompt version (the prompt is the instrument).
    """
    ok_cells: dict[tuple, dict] = {}
    failed_cells: dict[tuple, dict] = {}
    n_rows = 0
    exp_ids: set[str] = set()
    versions: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("tier") != TIER:
            continue
        n_rows += 1
        exp_ids.add(str(row.get("exp_id")))
        versions.add(str(row.get("prompt_version")))
        key = cell_key(row.get("request_model"), row.get("layer"), row.get("order"), row.get("pair_id"))
        (ok_cells if row.get("ok") else failed_cells)[key] = row
    if not n_rows:
        raise SystemExit(f"--resume-failed-from {path} has no {TIER} rows")
    if exp_ids != {exp_id}:
        raise SystemExit(
            f"--resume-failed-from {path} carries exp_id {sorted(exp_ids)} and this run is {exp_id!r}; "
            "a resumed batch has to land in the same experiment or the analysis will not join the two"
        )
    if versions != {prompt_version}:
        raise SystemExit(
            f"--resume-failed-from {path} used judge prompt {sorted(versions)} and this run would use "
            f"{prompt_version!r}; the prompt is the instrument here, so resuming across a prompt change "
            "is a different experiment rather than a continuation"
        )
    targets = {k: v for k, v in failed_cells.items() if k not in ok_cells}
    stats = {
        "resume_rows_read": n_rows,
        "resume_cells_already_answered": len(ok_cells),
        "resume_cells_failed": len(failed_cells),
        "resume_cells_failed_but_answered_elsewhere": len(failed_cells) - len(targets),
        "resume_cells_to_run": len(targets),
    }
    return targets, stats


def check_resume_inputs(targets: dict[tuple, dict], pairs: list[dict], judge_models: list[str]) -> None:
    """Every resumed cell has to be reproducible from the summaries this run
    loaded, and has to be the *same* two summaries the failed call would have
    compared.

    The pair texts come from --baseline-from / --contrast-from. Pointing either
    at a different file would re-run the design against different texts while
    the raw rows still claimed to be one experiment, so the source call ids
    recorded on the old row are compared with the ones about to be sent, and a
    mismatch stops the run.
    """
    index: dict[tuple, tuple[dict, str]] = {}
    for model in judge_models:
        for order in ORDERS:
            for pair in pairs:
                index[cell_key(model, pair["layer"], order, pair["pair_id"])] = (pair, order)
    missing = sorted(k for k in targets if k not in index)
    if missing:
        raise SystemExit(
            f"{len(missing)} cell(s) to resume cannot be built from the pairs this run loaded, e.g. "
            f"{missing[:5]}; check --judge-models, --cases and the two --*-from files"
        )
    mismatched: list[str] = []
    for key, row in sorted(targets.items()):
        pair, order = index[key]
        a_key, b_key = ("left", "right") if order == "left_first" else ("right", "left")
        want = row.get("source_call_ids") or {}
        got = {"position_a": pair[a_key]["source_call_id"], "position_b": pair[b_key]["source_call_id"]}
        if want and want != got:
            mismatched.append(f"{key}: recorded {want}, would send {got}")
    if mismatched:
        raise SystemExit(
            f"{len(mismatched)} cell(s) would be judged against different summaries than the failed call "
            f"used, e.g. {mismatched[:3]}; the summaries are inputs to the design, so this is a stop"
        )


async def judge_pair(
    pair: dict, order: str, model: str, prompt: PairwisePrompt,
    state: RunState, semaphore: asyncio.Semaphore, args: argparse.Namespace, cache_dir: Path | None,
) -> None:
    async with semaphore:
        if state.aborted:
            return
        if order == "left_first":
            a_key, b_key = "left", "right"
        else:
            a_key, b_key = "right", "left"
        case: GoldenCase = pair["case"]
        user_content = prompt.render(
            input_text=case.input_text,
            expected_summary=case.expected_summary,
            summary_a=pair[a_key]["summary"],
            summary_b=pair[b_key]["summary"],
        )
        try:
            rec = await asyncio.to_thread(
                call_structured,
                TIER,
                prompt.system_prompt,
                [{"role": "user", "content": user_content}],
                PairwiseVerdict,
                args.judge_temperature,
                args.max_tokens,
                cache_dir,
                args.temperature_transport,
                model,
            )
        except Exception as exc:  # noqa: BLE001 -- one bad call must not kill the batch
            rec = CallRecord(tier=TIER, request_model=model, ok=False, error_type=type(exc).__name__,
                             error_message=str(exc)[:2000], timestamp_utc=now_iso())
        await state.write(build_row(
            rec,
            run_id=args.run_id,
            exp_id=args.exp_id,
            repeat_idx=0,
            case_id=case.id,
            prompt_version=prompt.version,
            extra={
                "pair_id": pair["pair_id"],
                "layer": pair["layer"],
                "order": order,
                "position_a_source": pair[a_key]["source_label"],
                "position_b_source": pair[b_key]["source_label"],
                "left_source": pair["left"]["source_label"],
                "right_source": pair["right"]["source_label"],
                "summaries_identical": pair["summaries_identical"],
                "judge_model_requested": model,
                "source_run_ids": {
                    "position_a": pair[a_key]["source_run_id"],
                    "position_b": pair[b_key]["source_run_id"],
                },
                "source_call_ids": {
                    "position_a": pair[a_key]["source_call_id"],
                    "position_b": pair[b_key]["source_call_id"],
                },
            },
        ))


async def orchestrate(pairs: list[dict], prompt: PairwisePrompt, args: argparse.Namespace,
                      state: RunState, cache_dir: Path | None,
                      only_cells: set[tuple] | None = None) -> None:
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = []
    for model in args.judge_models:
        for layer in LAYERS:
            for order in ORDERS:
                for pair in pairs:
                    if pair["layer"] != layer:
                        continue
                    if only_cells is not None and cell_key(model, layer, order, pair["pair_id"]) not in only_cells:
                        continue
                    tasks.append(judge_pair(pair, order, model, prompt, state, semaphore, args, cache_dir))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for res in results:
        if isinstance(res, BaseException):
            print(f"!! task raised {type(res).__name__}: {res}", file=sys.stderr)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp-id", default="e2_pairwise")
    p.add_argument("--run-id", default=None)
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--baseline-from", required=True, help="JSONL of the E0 main arm (supplies both baseline repeats)")
    p.add_argument("--contrast-from", required=True, help="JSONL of the E1 v2b arm (supplies the easy-layer contrast)")
    p.add_argument("--baseline-repeats", default="0,1", help="two repeat indices of --baseline-from: first is the left summary, second is the hard-layer contrast")
    p.add_argument("--contrast-repeat", type=int, default=0)
    p.add_argument("--resume-failed-from", default=None,
                   help="JSONL of an earlier run of this experiment: send only the cells it recorded as failures")
    p.add_argument("--judge-models", default=",".join(DEFAULT_JUDGE_MODELS))
    p.add_argument("--cases", default=None, help="N | id,id,id | @file-of-ids (default: all confirmed cases)")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--judge-temperature", type=float, default=None)
    p.add_argument("--temperature-transport", choices=["auto", "param", "extra_body"], default="auto")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--cache", action="store_true")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    p.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    p.add_argument("--max-cost-usd", type=float, default=1.0)
    p.add_argument("--dry-run", action="store_true", help="print the plan, one rendered prompt and the call count; call nothing")
    args = p.parse_args(argv)
    args.judge_models = [m.strip() for m in args.judge_models.split(",") if m.strip()]
    if not args.judge_models:
        raise SystemExit("--judge-models is empty")
    repeats = [int(r) for r in args.baseline_repeats.split(",")]
    if len(repeats) != 2 or repeats[0] == repeats[1]:
        raise SystemExit(f"--baseline-repeats needs two distinct indices, got {args.baseline_repeats!r}")
    args.baseline_repeats = repeats
    if args.cache:
        # Two orders of the same pair are different requests, so the cache would
        # not collapse them -- but a re-run of the same order would be served
        # from disk and silently look perfectly consistent.
        raise SystemExit(
            "refusing to run: --cache would serve a repeated identical request from disk, and "
            "E2 measures exactly how stable a repeated judgement is. Drop --cache."
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    loaded = load_dotenv(REPO_ROOT / ".env")
    if loaded:
        print(f"loaded from .env: {', '.join(sorted(loaded))}")
    prices = assert_prices_current()
    prompt = PairwisePrompt(PROMPT_PATH)

    cases = select_cases(load_confirmed_cases(args.dataset), args.cases)
    base_r0, base_r1 = args.baseline_repeats
    sources = {
        "baseline_r0": load_summaries(Path(args.baseline_from), base_r0),
        "baseline_r1": load_summaries(Path(args.baseline_from), base_r1),
        "contrast_r0": load_summaries(Path(args.contrast_from), args.contrast_repeat),
    }
    for key, group in sources.items():
        for entry in group.values():
            entry["source_label"] = f"{entry['source_prompt_version']}_repeat{entry['source_repeat_idx']}"
    pairs, skipped = build_pairs(cases, sources)
    n_planned = len(pairs) * len(ORDERS) * len(args.judge_models)

    resume_targets: dict[tuple, dict] | None = None
    resume_stats: dict = {}
    if args.resume_failed_from:
        resume_targets, resume_stats = load_resume_targets(
            Path(args.resume_failed_from), exp_id=args.exp_id, prompt_version=prompt.version
        )
        check_resume_inputs(resume_targets, pairs, args.judge_models)
        n_planned = len(resume_targets)
        if n_planned == 0:
            raise SystemExit(
                f"--resume-failed-from {args.resume_failed_from} leaves nothing to run: every failed cell "
                "in that file already has a successful row"
            )

    args.run_id = args.run_id or f"{args.exp_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = Path(args.raw_dir) / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.run_id}.jsonl"
    meta_path = out_dir / f"{args.run_id}.meta.json"
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite existing raw file {out_path} (pass a fresh --run-id)")

    identical = sorted(p["pair_id"] for p in pairs if p["summaries_identical"])
    plan = {
        "run_id": args.run_id,
        "exp_id": args.exp_id,
        "judge_prompt_version": prompt.version,
        "judge_prompt_path": _repo_relative(PROMPT_PATH),
        "judge_models_requested": args.judge_models,
        "cases": len(cases),
        "layers": list(LAYERS),
        "orders": list(ORDERS),
        "pairs": len(pairs),
        "pairs_by_layer": {layer: sum(1 for p in pairs if p["layer"] == layer) for layer in LAYERS},
        "pairs_with_identical_summaries": identical,
        "pairs_skipped_missing_summary": skipped,
        "planned_calls": n_planned,
        "resume_failed_from": _repo_relative(Path(args.resume_failed_from)) if args.resume_failed_from else None,
        **resume_stats,
        "baseline_from": _repo_relative(Path(args.baseline_from)),
        "baseline_repeats": args.baseline_repeats,
        "contrast_from": _repo_relative(Path(args.contrast_from)),
        "contrast_repeat": args.contrast_repeat,
        "judge_temperature": args.judge_temperature,
        "temperature_transport": args.temperature_transport,
        "concurrency": args.concurrency,
        "cache_enabled": False,
        "max_cost_usd": args.max_cost_usd,
        "prices_per_mtok": prices,
        "git_commit": git_commit(),
        "dataset": _repo_relative(Path(args.dataset)),
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if args.dry_run:
        sample = pairs[0]
        print("\n--- rendered user message, first pair, left_first ---")
        print(prompt.render(
            input_text=sample["case"].input_text,
            expected_summary=sample["case"].expected_summary,
            summary_a=sample["left"]["summary"],
            summary_b=sample["right"]["summary"],
        ))
        print("--- end ---")
        print(f"dry run: would write {n_planned} rows to {out_path}")
        return 0

    state = RunState(out_path, args.max_cost_usd)
    started = now_iso()
    try:
        asyncio.run(orchestrate(
            pairs, prompt, args, state, None,
            set(resume_targets) if resume_targets is not None else None,
        ))
    finally:
        state.close()
    finished = now_iso()

    ok_rows = sum(1 for r in state.rows if r["ok"])
    err_rows = [r for r in state.rows if not r["ok"]]
    cumulative = record_usage(state.rows, ledger_path=args.ledger)
    choices: dict[str, int] = {}
    for row in state.rows:
        if row["ok"]:
            key = str((row.get("parsed") or {}).get("choice"))
            choices[key] = choices.get(key, 0) + 1
    meta = {
        **plan,
        "started_utc": started,
        "finished_utc": finished,
        "executed_calls": len(state.rows),
        "ok_calls": ok_rows,
        "failed_calls": len(err_rows),
        "failure_types": sorted({str(r.get("error_type")) for r in err_rows}),
        "cache_hits": sum(1 for r in state.rows if r.get("cache_hit")),
        "run_cost_usd": round(state.total_cost, 8),
        "ledger_cumulative_usd": cumulative,
        "aborted_on_cost_cap": state.aborted,
        "response_models_seen": sorted({str(r.get("response_model")) for r in state.rows if r.get("response_model")}),
        "choice_counts": dict(sorted(choices.items())),
        "anthropic_sdk_version": sdk_version(),
        "sdk_max_retries": 2,
        "raw_path": str(out_path.relative_to(REPO_ROOT)),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"\nwrote {len(state.rows)}/{n_planned} rows -> {out_path}"
        f"\n  ok={ok_rows} failed={len(err_rows)}"
        f"\n  choices {meta['choice_counts']}"
        f"\n  run cost ${state.total_cost:.6f}  ledger cumulative ${cumulative:.6f}"
        f"\n  response models: {', '.join(meta['response_models_seen']) or '(none)'}"
        f"\n  meta -> {meta_path}"
    )
    if state.aborted:
        return 2
    return 0 if len(state.rows) == n_planned and not err_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
