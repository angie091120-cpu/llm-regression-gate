"""Experiment runner: evaluate golden cases and write one JSONL line per API call.

The production runner (`evalkit/run_eval.py`) writes one aggregated
`eval_report.json` per run. That is the right artifact for a CI gate and the
wrong one for a study: it drops the per-call token counts, the dated model
snapshot, the failures, and it cannot express "the same case, measured five
times". This runner keeps the per-case logic identical (same prompt config,
same classifier schema, same judge prompt -- all imported from the modules
under test, none copied) and changes only what gets recorded.

Output layout:

    experiments/results/raw/<exp_id>/<run_id>.jsonl        one line per call
    experiments/results/raw/<exp_id>/<run_id>.meta.json    run-level provenance
    experiments/results/cost_ledger.json                   appended per run

Examples:

    # 10-case smoke, classifier + judge, no repeats
    python -m experiments.runner --exp-id smoke --cases 10

    # E0 noise floor: 70 cases x 5 repeats
    python -m experiments.runner --exp-id e0_noise --repeats 5

    # E0 judge-isolation arm: freeze run-1's classifier outputs, re-judge 4x
    python -m experiments.runner --exp-id e0_judge_iso --repeats 4 \\
        --judge-only-from experiments/results/raw/e0_noise/<run_id>.jsonl

    # E1 degradation arm
    python -m experiments.runner --exp-id e1_v2a --prompt-version v2a --repeats 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classifier import ClassificationResult, PromptConfig, load_prompt_config  # noqa: E402
from classifier import _few_shot_messages as build_few_shot_messages  # noqa: E402
from evalkit.cost import _price, record_usage  # noqa: E402
from evalkit.dataset import GoldenCase, load_confirmed_cases  # noqa: E402
from evalkit.judge import JUDGE_SYSTEM_PROMPT, JudgeVerdict  # noqa: E402
from experiments import PINNED_PRICES, price_key  # noqa: E402
from experiments.client import CallRecord, call_structured, sdk_version  # noqa: E402

DEFAULT_DATASET = REPO_ROOT / "golden_dataset.json"
DEFAULT_RAW_DIR = REPO_ROOT / "experiments" / "results" / "raw"
DEFAULT_CACHE_DIR = REPO_ROOT / "experiments" / "results" / "cache"
DEFAULT_LEDGER = REPO_ROOT / "experiments" / "results" / "cost_ledger.json"
DEFAULT_CONCURRENCY = 4
SCHEMA_VERSION = "1.0"

# List price per 1M tokens that every cost figure in this study assumes; the
# table itself lives in experiments/__init__.py, because analyze.py recomputes
# cost against the same pinned numbers. evalkit/cost.py on `main` still carries
# the pre-2026-09 Sonnet price, so the run aborts unless the PRICE_* overrides
# from docs/DECISIONS.md D-005 are exported. Changing a price means changing
# that table *and* re-running, not silently reinterpreting old numbers.
EXPECTED_PRICES: dict[str, tuple[float, float]] = dict(PINNED_PRICES)


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------
def load_dotenv(path: Path) -> list[str]:
    """Minimal .env loader (no third-party dependency). Returns the variable
    names it set -- names only, never values, so a log line can show that a
    key was loaded without showing the key."""
    loaded: list[str] = []
    if not path.exists():
        return loaded
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def assert_prices_current() -> dict[str, dict[str, float]]:
    """Fail loudly when the effective price table is not the one this study
    assumes. A stale price does not crash anything downstream -- it silently
    produces a wrong cost column -- so it is checked before the first call."""
    effective: dict[str, dict[str, float]] = {}
    problems: list[str] = []
    for model, (want_in, want_out) in EXPECTED_PRICES.items():
        got_in = _price(model, "input")
        got_out = _price(model, "output")
        effective[model] = {"input": got_in, "output": got_out}
        if abs(got_in - want_in) > 1e-9 or abs(got_out - want_out) > 1e-9:
            env_in = f"PRICE_{model.upper().replace('-', '_')}_INPUT"
            env_out = f"PRICE_{model.upper().replace('-', '_')}_OUTPUT"
            problems.append(
                f"  {model}: effective ${got_in}/${got_out} per 1M, expected ${want_in}/${want_out}\n"
                f"    fix: export {env_in}={want_in:.2f} {env_out}={want_out:.2f}"
            )
    if problems:
        raise SystemExit(
            "refusing to run: the effective token price is not the one this study assumes\n"
            + "\n".join(problems)
            + "\n(see docs/DECISIONS.md D-005 and experiments/README.md)"
        )
    return effective


def cost_for(record: CallRecord) -> float:
    """Dollars for one call, from the usage the API reported.

    A failed call is charged whenever the API still reported token usage. The
    original version returned $0 for every `ok: false` row, which is right for
    a request the API rejected (a 400 carries no usage) and wrong for a
    request that returned 200, burned tokens and then failed validation on our
    side -- and the second kind is exactly what a large batch produces. The
    E0 judge-isolation arm hit this once, for $0.005338; at E1's call volume
    the same bug has far more room (experiments/COST_CALIBRATION.md section 3).

    Cache hits are $0 by definition. A record with no usage at all is $0,
    which also keeps the price lookup away from the placeholder request model
    written when a call never reached the API.
    """
    if record.cache_hit:
        return 0.0
    if not record.input_tokens and not record.output_tokens:
        return 0.0
    key = price_key(record.response_model, record.request_model)
    return (record.input_tokens * _price(key, "input") + record.output_tokens * _price(key, "output")) / 1_000_000


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5, cwd=REPO_ROOT
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _repo_relative(path: Path) -> str:
    """Committed provenance records repo-relative paths, so a run's metadata
    does not carry someone's home directory into a public repository."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------
# case selection
# --------------------------------------------------------------------------
def select_cases(cases: list[GoldenCase], spec: str | None) -> list[GoldenCase]:
    """--cases accepts: N (first N, dataset order), a comma-separated id list,
    or @path to a newline-separated id file."""
    if spec is None:
        return cases
    spec = spec.strip()
    if spec.isdigit():
        n = int(spec)
        if n <= 0 or n > len(cases):
            raise SystemExit(f"--cases {n} out of range (dataset has {len(cases)} confirmed cases)")
        return cases[:n]
    if spec.startswith("@"):
        ids = [ln.strip() for ln in Path(spec[1:]).read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        ids = [s.strip() for s in spec.split(",") if s.strip()]
    by_id = {c.id: c for c in cases}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"--cases lists ids that are not confirmed cases in the dataset: {missing}")
    return [by_id[i] for i in ids]


def load_frozen_summaries(path: Path, repeat_idx: int) -> dict[str, dict]:
    """Judge-isolation arm: read a previous run's classifier rows and freeze
    their outputs, so a repeated judge call varies only in the judge."""
    frozen: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("tier") != "classifier" or not row.get("ok"):
            continue
        if row.get("repeat_idx") != repeat_idx:
            continue
        parsed = row.get("parsed") or {}
        frozen[row["case_id"]] = {
            "summary": parsed.get("summary"),
            "category": parsed.get("category"),
            "source_call_id": row.get("call_id"),
            "source_run_id": row.get("run_id"),
        }
    if not frozen:
        raise SystemExit(f"--judge-only-from {path} has no successful classifier rows at repeat_idx={repeat_idx}")
    return frozen


# --------------------------------------------------------------------------
# run state
# --------------------------------------------------------------------------
class RunState:
    def __init__(self, out_path: Path, max_cost_usd: float) -> None:
        self.out_path = out_path
        self.max_cost_usd = max_cost_usd
        self.lock = asyncio.Lock()
        self.rows: list[dict] = []
        self.total_cost = 0.0
        self.aborted = False
        self.handle = out_path.open("a", encoding="utf-8")

    async def write(self, row: dict) -> None:
        async with self.lock:
            self.rows.append(row)
            self.total_cost += row.get("cost_usd", 0.0)
            self.handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
            self.handle.flush()
            if self.total_cost > self.max_cost_usd and not self.aborted:
                self.aborted = True
                print(
                    f"!! cost cap hit: ${self.total_cost:.4f} > --max-cost-usd ${self.max_cost_usd:.4f}; "
                    "stopping after in-flight calls",
                    file=sys.stderr,
                )

    def close(self) -> None:
        self.handle.close()


def build_row(
    record: CallRecord,
    *,
    run_id: str,
    exp_id: str,
    repeat_idx: int,
    case_id: str,
    prompt_version: str,
    extra: dict | None = None,
) -> dict:
    """The raw-data contract. Key order is fixed for readability of the JSONL;
    analyze.py reads by name, never by position."""
    rec = asdict(record)
    row = {
        "schema_version": SCHEMA_VERSION,
        "call_id": uuid.uuid4().hex,
        "run_id": run_id,
        "exp_id": exp_id,
        "repeat_idx": repeat_idx,
        "case_id": case_id,
        "prompt_version": prompt_version,
        "tier": rec["tier"],
        "request_model": rec["request_model"],
        "response_model": rec["response_model"],
        "temperature": rec["temperature"],
        "temperature_transport": rec["temperature_transport"],
        "max_tokens": rec["max_tokens"],
        "ok": rec["ok"],
        "input_tokens": rec["input_tokens"],
        "output_tokens": rec["output_tokens"],
        "usage": rec["usage"],
        "latency_ms": round(rec["latency_ms"], 2) if rec["latency_ms"] is not None else None,
        "cost_usd": round(cost_for(record), 8),
        "price_key": price_key(rec["response_model"], rec["request_model"]),
        "timestamp_utc": rec["timestamp_utc"],
        "cache_hit": rec["cache_hit"],
        "request_hash": rec["request_hash"],
        "stop_reason": rec["stop_reason"],
        "has_thinking_block": rec["has_thinking_block"],
        "parsed": rec["parsed"],
        "error_type": rec["error_type"],
        "error_message": rec["error_message"],
        "status_code": rec["status_code"],
        "request_id": rec["request_id"],
        "raw_response_json": rec["raw_response_json"],
    }
    if extra:
        row.update(extra)
    return row


# --------------------------------------------------------------------------
# per-case work
# --------------------------------------------------------------------------
async def run_case(
    case: GoldenCase,
    repeat_idx: int,
    prompt_config: PromptConfig,
    state: RunState,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    cache_dir: Path | None,
) -> None:
    """One case, one repeat: classifier call then judge call. Any failure is
    recorded as a row and swallowed -- the batch keeps going."""
    async with semaphore:
        if state.aborted:
            return
        try:
            messages = build_few_shot_messages(prompt_config.few_shot_examples)
            messages.append({"role": "user", "content": case.input_text})
            clf = await asyncio.to_thread(
                call_structured,
                "classifier",
                prompt_config.system_prompt,
                messages,
                ClassificationResult,
                args.classifier_temperature,
                args.max_tokens,
                cache_dir,
                args.temperature_transport,
            )
        except Exception as exc:  # noqa: BLE001 -- non-API failure (bad prompt config etc.)
            clf = CallRecord(tier="classifier", request_model="?", ok=False, error_type=type(exc).__name__, error_message=str(exc)[:2000], timestamp_utc=now_iso())
        await state.write(
            build_row(clf, run_id=args.run_id, exp_id=args.exp_id, repeat_idx=repeat_idx, case_id=case.id, prompt_version=prompt_config.version)
        )
        if not clf.ok or args.skip_judge or state.aborted:
            return

        summary = (clf.parsed or {}).get("summary", "")
        await judge_call(
            case=case,
            repeat_idx=repeat_idx,
            candidate_summary=summary,
            state=state,
            args=args,
            prompt_version=prompt_config.version,
            cache_dir=cache_dir,
            extra={"judged_summary_source": "same_run_classifier"},
        )


async def judge_call(
    *,
    case: GoldenCase,
    repeat_idx: int,
    candidate_summary: str,
    state: RunState,
    args: argparse.Namespace,
    prompt_version: str,
    cache_dir: Path | None,
    extra: dict,
) -> None:
    user_content = (
        f"Original email:\n{case.input_text}\n\n"
        f"Reference summary (human-approved):\n{case.expected_summary}\n\n"
        f"Candidate summary (system under test):\n{candidate_summary}\n\n"
        "Score the candidate 1-5 by calling the JudgeVerdict tool."
    )
    try:
        rec = await asyncio.to_thread(
            call_structured,
            "judge",
            JUDGE_SYSTEM_PROMPT,
            [{"role": "user", "content": user_content}],
            JudgeVerdict,
            args.judge_temperature,
            args.max_tokens,
            cache_dir,
            args.temperature_transport,
        )
    except Exception as exc:  # noqa: BLE001
        rec = CallRecord(tier="judge", request_model="?", ok=False, error_type=type(exc).__name__, error_message=str(exc)[:2000], timestamp_utc=now_iso())
    await state.write(
        build_row(rec, run_id=args.run_id, exp_id=args.exp_id, repeat_idx=repeat_idx, case_id=case.id, prompt_version=prompt_version, extra=extra)
    )


async def run_judge_only(
    case: GoldenCase,
    repeat_idx: int,
    frozen: dict,
    state: RunState,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    prompt_version: str,
    cache_dir: Path | None,
) -> None:
    async with semaphore:
        if state.aborted:
            return
        await judge_call(
            case=case,
            repeat_idx=repeat_idx,
            candidate_summary=frozen["summary"] or "",
            state=state,
            args=args,
            prompt_version=prompt_version,
            cache_dir=cache_dir,
            extra={
                "judged_summary_source": "frozen_classifier_output",
                "source_run_id": frozen["source_run_id"],
                "source_call_id": frozen["source_call_id"],
                "frozen_category": frozen["category"],
            },
        )


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
async def orchestrate(cases: list[GoldenCase], prompt_config: PromptConfig, args: argparse.Namespace, state: RunState, cache_dir: Path | None) -> None:
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = []
    if args.judge_only_from:
        frozen = load_frozen_summaries(Path(args.judge_only_from), args.source_repeat_idx)
        for repeat_idx in range(args.repeats):
            for case in cases:
                if case.id not in frozen:
                    continue
                tasks.append(run_judge_only(case, repeat_idx, frozen[case.id], state, semaphore, args, prompt_config.version, cache_dir))
    else:
        for repeat_idx in range(args.repeats):
            for case in cases:
                tasks.append(run_case(case, repeat_idx, prompt_config, state, semaphore, args, cache_dir))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for res in results:
        if isinstance(res, BaseException):
            print(f"!! task raised {type(res).__name__}: {res}", file=sys.stderr)


def planned_calls(n_cases: int, args: argparse.Namespace) -> int:
    per_case = 1 if (args.judge_only_from or args.skip_judge) else 2
    return n_cases * args.repeats * per_case


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp-id", required=True, help="experiment id; also the raw output subdirectory")
    p.add_argument("--run-id", default=None, help="default: <exp_id>_<UTC timestamp>")
    p.add_argument("--prompt-version", default="v1")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--cases", default=None, help="N | id,id,id | @file-of-ids (default: all confirmed cases)")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--classifier-temperature", type=float, default=None, help="sent only on the classifier (Haiku) call")
    p.add_argument("--judge-temperature", type=float, default=None, help="sent only on the judge call; Sonnet 5 returns 400 '`temperature` is deprecated for this model' (measured, see experiments/COST_CALIBRATION.md)")
    p.add_argument("--temperature-transport", choices=["auto", "param", "extra_body"], default="auto", help="how a temperature reaches the wire; auto picks param on anthropic<1.0 and extra_body on >=1.0")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--skip-judge", action="store_true", help="classifier only (primary metric category_match needs no judge)")
    p.add_argument("--judge-only-from", default=None, help="JSONL from a previous run; freeze its classifier outputs and re-judge them")
    p.add_argument("--source-repeat-idx", type=int, default=0, help="which repeat of --judge-only-from to freeze")
    p.add_argument("--cache", action="store_true", help="content-hash cache; OFF by default, see the warning in experiments/README.md")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    p.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    p.add_argument("--max-cost-usd", type=float, default=1.0, help="hard stop for this run")
    p.add_argument("--dry-run", action="store_true", help="print the plan and the projected call count, call nothing")
    args = p.parse_args(argv)

    if args.cache and args.repeats > 1:
        raise SystemExit(
            "refusing to run: --cache with --repeats > 1.\n"
            "Repeated identical requests are the measurement in E0/E1 -- serving them "
            "from cache would report a noise floor of exactly zero. Drop --cache."
        )
    if args.judge_only_from and args.skip_judge:
        raise SystemExit("--judge-only-from and --skip-judge are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    loaded = load_dotenv(REPO_ROOT / ".env")
    if loaded:
        print(f"loaded from .env: {', '.join(sorted(loaded))}")
    prices = assert_prices_current()

    prompt_config = load_prompt_config(args.prompt_version)
    all_cases = load_confirmed_cases(args.dataset)
    cases = select_cases(all_cases, args.cases)
    args.run_id = args.run_id or f"{args.exp_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    n_planned = planned_calls(len(cases), args)

    out_dir = Path(args.raw_dir) / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.run_id}.jsonl"
    meta_path = out_dir / f"{args.run_id}.meta.json"
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite existing raw file {out_path} (pass a fresh --run-id)")

    from experiments.client import model_for_tier

    plan = {
        "run_id": args.run_id,
        "exp_id": args.exp_id,
        "prompt_version": prompt_config.version,
        "cases": len(cases),
        "case_ids": [c.id for c in cases],
        "repeats": args.repeats,
        "planned_calls": n_planned,
        "classifier_model_requested": model_for_tier("classifier"),
        "judge_model_requested": model_for_tier("judge"),
        "classifier_temperature": args.classifier_temperature,
        "judge_temperature": args.judge_temperature,
        "temperature_transport": args.temperature_transport,
        "concurrency": args.concurrency,
        "cache_enabled": bool(args.cache),
        "judge_only_from": args.judge_only_from,
        "skip_judge": bool(args.skip_judge),
        "max_cost_usd": args.max_cost_usd,
        "prices_per_mtok": prices,
        "git_commit": git_commit(),
        "dataset": _repo_relative(Path(args.dataset)),
    }
    print(json.dumps({k: v for k, v in plan.items() if k != "case_ids"}, indent=2, ensure_ascii=False))
    if args.dry_run:
        print(f"dry run: would write {n_planned} rows to {out_path}")
        return 0

    cache_dir = Path(args.cache_dir) if args.cache else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    state = RunState(out_path, args.max_cost_usd)
    started = now_iso()
    try:
        asyncio.run(orchestrate(cases, prompt_config, args, state, cache_dir))
    finally:
        state.close()
    finished = now_iso()

    ok_rows = sum(1 for r in state.rows if r["ok"])
    err_rows = [r for r in state.rows if not r["ok"]]
    cumulative = record_usage(state.rows, ledger_path=args.ledger)
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
        "anthropic_sdk_version": sdk_version(),
        "sdk_max_retries": 2,
        "raw_path": str(out_path.relative_to(REPO_ROOT)),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"\nwrote {len(state.rows)}/{n_planned} rows -> {out_path}"
        f"\n  ok={ok_rows} failed={len(err_rows)} cache_hits={meta['cache_hits']}"
        f"\n  run cost ${state.total_cost:.6f}  ledger cumulative ${cumulative:.6f}"
        f"\n  response models: {', '.join(meta['response_models_seen']) or '(none)'}"
        f"\n  meta -> {meta_path}"
    )
    if state.aborted:
        return 2
    return 0 if len(state.rows) == n_planned and not err_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
