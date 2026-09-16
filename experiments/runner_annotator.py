"""E4 model-annotation arm: each model labels every case once, from the
category definitions alone.

Separate from `runner.py` and `runner_pairwise.py` because the unit is
different again. `runner.py` measures the system under test (classifier call,
then judge call); this measures a *rater*: one call per (model, case), no
judge, no summary, no few-shot examples. Everything the three share -- the
.env loader, the price guard, the raw-row contract, the cost cap, the ledger --
is imported from `runner.py` rather than copied.

What E4 compares (docs/PREREGISTRATION.md sections 2 and 6): the gold labels in
`golden_dataset.json`, decided by one human in July 2026, against a second
human annotator (`experiments/data/annotator2_labels.json`, handled by
`import_annotator2.py`) and against these model annotations. The three raters
are given the same thing -- the email text, the four category definitions and
the tie-break rule -- so that the kappa between any two of them is about the
labelling task and not about who saw more.

    export PRICE_CLAUDE_SONNET_5_INPUT=2.00 PRICE_CLAUDE_SONNET_5_OUTPUT=10.00

    python -m experiments.runner_annotator --exp-id e4_annot --dry-run
    python -m experiments.runner_annotator --exp-id e4_annot --max-cost-usd 1.00

70 cases x 2 models = 140 calls.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap
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
from experiments.client import CallRecord, call_structured  # noqa: E402
from experiments.runner import (  # noqa: E402
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

PROMPT_PATH = REPO_ROOT / "experiments" / "prompts" / "annotator_model_v1.yaml"
SOURCE_PROMPT_PATH = REPO_ROOT / "prompts" / "v1.yaml"
DEFAULT_MODELS = ("claude-haiku-4-5", "claude-sonnet-5")
TIER = "annotator"
CATEGORIES = ("billing", "technical", "account", "general")


class AnnotationLabel(BaseModel):
    """One category and nothing else.

    No reasoning field, on purpose: asking for a written justification changes
    the task -- it is a different rater from the one that just picks a label,
    and the human annotators were asked for a label, not an argument.
    """

    category: Literal["billing", "technical", "account", "general"] = Field(
        description="the single category this email belongs to"
    )


def quoted_block(path: Path, start: int, end: int) -> str:
    """Lines `start`..`end` of a file, 1-based and inclusive, with the common
    leading indentation removed."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if end > len(lines) or start < 1 or start > end:
        raise SystemExit(f"{path} has no lines {start}-{end} (file has {len(lines)})")
    return textwrap.dedent("\n".join(lines[start - 1:end]))


class AnnotatorPrompt:
    """The E4 prompt, with its quotes checked against the file they came from.

    The point of the check: the four category definitions are the measurement
    instrument, shared between the classifier being studied, the human
    annotators' handout and this prompt. An edit to `prompts/v1.yaml` that
    leaves this file behind would produce a kappa between raters who were given
    different definitions, and nothing in the output would show it. So the line
    ranges are recorded in the YAML, re-read here, and a mismatch stops the run.
    """

    REQUIRED_KEYS = ("version", "system_prompt", "user_template", "tool_name", "quoted_from")

    def __init__(self, path: Path) -> None:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        missing = [k for k in self.REQUIRED_KEYS if k not in data]
        if missing:
            raise SystemExit(f"{path} is missing required keys: {missing}")
        if data["tool_name"] != AnnotationLabel.__name__:
            raise SystemExit(
                f"{path} names tool {data['tool_name']!r} but the schema this runner sends is "
                f"{AnnotationLabel.__name__!r}; the two must agree"
            )
        self.version: str = data["version"]
        self.system_prompt: str = data["system_prompt"]
        self.user_template: str = data["user_template"]
        self.quoted_from: dict = data["quoted_from"]
        self.quote_status = self._check_quotes()

    def _check_quotes(self) -> dict:
        source = REPO_ROOT / str(self.quoted_from["path"])
        if source.resolve() != SOURCE_PROMPT_PATH.resolve():
            raise SystemExit(
                f"{PROMPT_PATH} quotes {source}, but E4's definitions have to come from "
                f"{_repo_relative(SOURCE_PROMPT_PATH)}"
            )
        status: dict[str, str] = {}
        problems: list[str] = []
        for key in ("category_definitions_lines", "tie_break_lines"):
            start, end = (int(v) for v in self.quoted_from[key])
            block = quoted_block(source, start, end)
            if block in self.system_prompt:
                status[key] = f"{_repo_relative(source)}:{start}-{end} quoted verbatim"
            else:
                problems.append(
                    f"{key}: {_repo_relative(source)} lines {start}-{end} are not in this prompt verbatim. "
                    f"First line expected: {block.splitlines()[0]!r}"
                )
        if problems:
            raise SystemExit(
                "refusing to run: the E4 prompt no longer quotes the definitions it claims to quote.\n  "
                + "\n  ".join(problems)
                + "\n(edit experiments/prompts/annotator_model_v1.yaml, or the line ranges under "
                "quoted_from, so the two agree again)"
            )
        return status

    def render(self, *, input_text: str) -> str:
        """Substitution by replace, not str.format: an email containing a brace
        is data, not a format field."""
        placeholder = "{input_text}"
        if placeholder not in self.user_template:
            raise SystemExit(f"{PROMPT_PATH} user_template has no {placeholder} placeholder")
        return self.user_template.replace(placeholder, input_text)


async def annotate(
    case: GoldenCase, model: str, prompt: AnnotatorPrompt, state: RunState,
    semaphore: asyncio.Semaphore, args: argparse.Namespace,
) -> None:
    async with semaphore:
        if state.aborted:
            return
        try:
            rec = await asyncio.to_thread(
                call_structured,
                TIER,
                prompt.system_prompt,
                [{"role": "user", "content": prompt.render(input_text=case.input_text)}],
                AnnotationLabel,
                args.temperature,
                args.max_tokens,
                None,
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
                "annotator": f"model:{model}",
                "annotator_model_requested": model,
                "annotation_target": "category",
            },
        ))


async def orchestrate(cases: list[GoldenCase], prompt: AnnotatorPrompt,
                      args: argparse.Namespace, state: RunState) -> None:
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [annotate(case, model, prompt, state, semaphore, args)
             for model in args.models for case in cases]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for res in results:
        if isinstance(res, BaseException):
            print(f"!! task raised {type(res).__name__}: {res}", file=sys.stderr)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp-id", default="e4_annot")
    p.add_argument("--run-id", default=None)
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--cases", default=None, help="N | id,id,id | @file-of-ids (default: all confirmed cases)")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--temperature", type=float, default=None,
                   help="left unset by default: Sonnet 5 rejects the parameter (COST_CALIBRATION.md 4.1)")
    p.add_argument("--temperature-transport", choices=["auto", "param", "extra_body"], default="auto")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    p.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    p.add_argument("--max-cost-usd", type=float, default=1.0)
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and one rendered prompt; call nothing")
    args = p.parse_args(argv)
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not args.models:
        raise SystemExit("--models is empty")
    if len(set(args.models)) != len(args.models):
        raise SystemExit(f"--models repeats a model: {args.models}; one annotation per model per case")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    loaded = load_dotenv(REPO_ROOT / ".env")
    if loaded:
        print(f"loaded from .env: {', '.join(sorted(loaded))}")
    prices = assert_prices_current()
    prompt = AnnotatorPrompt(PROMPT_PATH)

    cases = select_cases(load_confirmed_cases(args.dataset), args.cases)
    n_planned = len(cases) * len(args.models)

    args.run_id = args.run_id or f"{args.exp_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = Path(args.raw_dir) / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.run_id}.jsonl"
    meta_path = out_dir / f"{args.run_id}.meta.json"
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite existing raw file {out_path} (pass a fresh --run-id)")

    plan = {
        "run_id": args.run_id,
        "exp_id": args.exp_id,
        "prompt_version": prompt.version,
        "prompt_path": _repo_relative(PROMPT_PATH),
        "prompt_quotes_verified": prompt.quote_status,
        "few_shot_examples": 0,
        "models_requested": args.models,
        "cases": len(cases),
        "categories": list(CATEGORIES),
        "planned_calls": n_planned,
        "temperature": args.temperature,
        "temperature_transport": args.temperature_transport,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "cache_enabled": False,
        "max_cost_usd": args.max_cost_usd,
        "prices_per_mtok": prices,
        "git_commit": git_commit(),
        "dataset": _repo_relative(Path(args.dataset)),
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("\n--- system prompt ---")
        print(prompt.system_prompt)
        print("--- rendered user message, first case ---")
        print(prompt.render(input_text=cases[0].input_text))
        print("--- end ---")
        print(f"dry run: would write {n_planned} rows to {out_path}")
        return 0

    state = RunState(out_path, args.max_cost_usd)
    started = now_iso()
    try:
        asyncio.run(orchestrate(cases, prompt, args, state))
    finally:
        state.close()
    finished = now_iso()

    ok_rows = sum(1 for r in state.rows if r["ok"])
    err_rows = [r for r in state.rows if not r["ok"]]
    cumulative = record_usage(state.rows, ledger_path=args.ledger)
    labels: dict[str, dict[str, int]] = {}
    for row in state.rows:
        if row["ok"]:
            per_model = labels.setdefault(row["request_model"], {})
            key = str((row.get("parsed") or {}).get("category"))
            per_model[key] = per_model.get(key, 0) + 1
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
        "label_counts_by_model": {m: dict(sorted(v.items())) for m, v in sorted(labels.items())},
        "sdk_max_retries": 2,
        "raw_path": str(out_path.relative_to(REPO_ROOT)),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"\nwrote {len(state.rows)}/{n_planned} rows -> {out_path}"
        f"\n  ok={ok_rows} failed={len(err_rows)}"
        f"\n  labels {meta['label_counts_by_model']}"
        f"\n  run cost ${state.total_cost:.6f}  ledger cumulative ${cumulative:.6f}"
        f"\n  response models: {', '.join(meta['response_models_seen']) or '(none)'}"
        f"\n  meta -> {meta_path}"
    )
    if state.aborted:
        return 2
    return 0 if len(state.rows) == n_planned and not err_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
