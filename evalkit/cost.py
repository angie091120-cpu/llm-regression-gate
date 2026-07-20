"""Token-cost accounting (SPEC.md §3.4 budget tracking).

Pricing defaults were verified against Anthropic's official model pricing
on 2026-07-20 (list price per 1M tokens: Haiku 4.5 $1/$5, Sonnet 5 $3/$15).
We deliberately use list price rather than any time-limited introductory
discount, so cost estimates err conservative and don't go stale when a
promo ends. Prices do change -- re-verify before relying on
cost_ledger.json for hard budget decisions, or override per model with the
PRICE_<MODEL>_INPUT / PRICE_<MODEL>_OUTPUT environment variables (model
name upper-cased, hyphens -> underscores, e.g.
PRICE_CLAUDE_HAIKU_4_5_INPUT). See docs/DECISIONS.md D-005.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

LEDGER_PATH = Path(os.environ.get("EVALKIT_COST_LEDGER", "cost_ledger.json"))

# USD per 1,000,000 tokens, list price (verified 2026-07-20 -- see docstring).
_DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
}


def _price(model: str, direction: str) -> float:
    env_key = f"PRICE_{model.upper().replace('-', '_')}_{direction.upper()}"
    if env_key in os.environ:
        return float(os.environ[env_key])
    if model in _DEFAULT_PRICING:
        return _DEFAULT_PRICING[model][direction]
    raise ValueError(f"no pricing configured for model={model!r}; set {env_key}")


def token_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * _price(model, "input") + output_tokens * _price(model, "output")) / 1_000_000


def record_usage(case_results: list[dict], ledger_path: str | Path | None = None) -> float:
    """Append this run's total cost to the cost ledger and return the new
    cumulative spend, so callers can enforce the $7.5 (report) / $12 (stop)
    thresholds from SPEC.md §3.4. `case_results` entries are expected to
    carry a `cost_usd` key (see evalkit/run_eval.py)."""
    path = Path(ledger_path) if ledger_path is not None else LEDGER_PATH
    ledger = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": [], "cumulative_usd": 0.0}
    run_total = round(sum(r.get("cost_usd", 0.0) for r in case_results), 6)
    ledger["runs"].append(
        {"timestamp": datetime.now(timezone.utc).isoformat(), "cases": len(case_results), "cost_usd": run_total}
    )
    ledger["cumulative_usd"] = round(ledger.get("cumulative_usd", 0.0) + run_total, 6)
    path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return ledger["cumulative_usd"]
