"""Evaluation engine (SPEC.md §4): run classify_email() against every
confirmed golden-dataset case and score it on four dimensions --
category_match, judge_score, latency_ms, tokens.

`from classifier import classify_email, load_prompt_config` below is the one
project-specific line in this package. To vendor evalkit/ into a different
project (e.g. #6 hybrid-rag-engine), swap that import for that project's
function-under-test and its config loader -- dataset.py, judge.py, cost.py,
diff.py and report.py have no dependency on the classifier and need no
changes.

Usage: python -m evalkit.run_eval --prompt-version v1 --dataset golden_dataset.json --out eval_report.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from classifier import ClassificationResult, PromptConfig, classify_email, load_prompt_config
from evalkit.cost import record_usage, token_cost_usd
from evalkit.dataset import GoldenCase, load_confirmed_cases
from evalkit.judge import score_summary

DEFAULT_PASS_THRESHOLD = 3  # judge_score >= this AND category_match -> pass (SPEC.md §4)
DEFAULT_CONCURRENCY = int(os.environ.get("EVAL_CONCURRENCY", "5"))


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip()
    except Exception:
        return None


async def _evaluate_case(case: GoldenCase, prompt_config: PromptConfig, semaphore: asyncio.Semaphore, pass_threshold: int) -> dict:
    async with semaphore:
        result: ClassificationResult = await asyncio.to_thread(classify_email, case.input_text, prompt_config)
        clf_resp = result.llm_response
        assert clf_resp is not None, "classify_email must attach its LLMResponse (see classifier.py)"
        judge_score, judge_resp = await asyncio.to_thread(
            score_summary, case.input_text, case.expected_summary, result.summary
        )

    category_match = result.category == case.expected_category
    passed = bool(category_match and judge_score >= pass_threshold)
    cost_usd = token_cost_usd(clf_resp.model, clf_resp.input_tokens, clf_resp.output_tokens) + token_cost_usd(
        judge_resp.model, judge_resp.input_tokens, judge_resp.output_tokens
    )
    return {
        "id": case.id,
        "language": case.language,
        "expected_category": case.expected_category,
        "predicted_category": result.category,
        "expected_summary": case.expected_summary,
        "predicted_summary": result.summary,
        "category_match": category_match,
        "judge_score": judge_score,
        "passed": passed,
        "latency_ms": round(clf_resp.latency_ms, 2),
        "tokens": {
            "input": clf_resp.input_tokens + judge_resp.input_tokens,
            "output": clf_resp.output_tokens + judge_resp.output_tokens,
        },
        "cost_usd": round(cost_usd, 6),
    }


async def _run_all(cases: list[GoldenCase], prompt_config: PromptConfig, concurrency: int, pass_threshold: int) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [_evaluate_case(c, prompt_config, semaphore, pass_threshold) for c in cases]
    return await asyncio.gather(*tasks)


def _per_category_accuracy(results: list[dict]) -> dict[str, float]:
    by_cat: dict[str, list[bool]] = {}
    for r in results:
        by_cat.setdefault(r["expected_category"], []).append(r["passed"])
    return {cat: round(sum(v) / len(v), 4) for cat, v in by_cat.items()}


def run_eval(
    prompt_version: str,
    dataset_path: str | Path,
    pass_threshold: int = DEFAULT_PASS_THRESHOLD,
    concurrency: int = DEFAULT_CONCURRENCY,
    limit: int | None = None,
    cost_ledger_path: str | Path | None = None,
) -> dict:
    """Run the eval suite and return the eval_report.json contents as a dict.
    Raises ValueError if the dataset has zero confirmed cases -- this
    function never silently evaluates against nothing."""
    prompt_config = load_prompt_config(prompt_version)
    cases = load_confirmed_cases(dataset_path)
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise ValueError(
            f"no label_status='confirmed' cases found in {dataset_path} "
            "(after an optional --limit). Golden dataset cases need "
            "expected_category/expected_summary filled in and label_status "
            "flipped to 'confirmed' by a human reviewer before they can be "
            "evaluated -- see SPEC.md §3.1(c)."
        )

    results = asyncio.run(_run_all(cases, prompt_config, concurrency, pass_threshold))
    pass_rate = round(sum(r["passed"] for r in results) / len(results), 4)
    cumulative_usd = record_usage(results, ledger_path=cost_ledger_path)

    return {
        "prompt_version": prompt_config.version,
        "model": prompt_config.model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "pass_threshold": pass_threshold,
        "pass_rate": pass_rate,
        "per_category_accuracy": _per_category_accuracy(results),
        "cumulative_cost_usd": cumulative_usd,
        "cases": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the eval suite against confirmed golden dataset cases.")
    parser.add_argument("--prompt-version", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="eval_report.json")
    parser.add_argument("--pass-threshold", type=int, default=DEFAULT_PASS_THRESHOLD)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N confirmed cases (cheap local iteration; see SPEC.md §3.4)")
    parser.add_argument("--cost-ledger", default=None, help="override cost_ledger.json path (default: evalkit.cost.LEDGER_PATH)")
    args = parser.parse_args(argv)

    report = run_eval(
        args.prompt_version,
        args.dataset,
        pass_threshold=args.pass_threshold,
        concurrency=args.concurrency,
        limit=args.limit,
        cost_ledger_path=args.cost_ledger,
    )
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"pass_rate={report['pass_rate']:.3f} cases={len(report['cases'])} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
