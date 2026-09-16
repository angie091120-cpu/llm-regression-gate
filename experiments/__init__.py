"""Pilot experiments for the LLM-as-judge reliability study.

Additive package: nothing under evalkit/, prompts/v1.yaml, golden_dataset.json
or tests/ is modified by anything in here. See experiments/README.md.
"""

# USD per 1M input/output tokens, list price on 2026-09-14. Every cost figure
# in this study is computed from this table and nothing else:
# `runner.py` refuses to start when the effective `evalkit.cost` prices differ
# from it, and `analyze.py` recomputes cost from raw `usage` against it rather
# than through the environment, so a stray PRICE_* override in a shell cannot
# change a published number after the fact (docs/PREREGISTRATION.md section 7).
PINNED_PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
}


def price_key(response_model: str | None, request_model: str) -> str:
    """Map a dated snapshot id (claude-haiku-4-5-20251001) back to the price
    table key (claude-haiku-4-5). Falls back to the request string when the
    call failed and there is no response model."""
    name = response_model or request_model
    parts = name.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 8:
        return parts[0]
    return name


def cost_from_usage(key: str, input_tokens: int, output_tokens: int) -> float:
    """Cost of one call at the pinned prices. Raises on an unknown model
    rather than returning 0 -- a silent zero is how a cost column goes wrong
    without anyone noticing."""
    if not input_tokens and not output_tokens:
        return 0.0
    if key not in PINNED_PRICES:
        raise KeyError(f"no pinned price for model {key!r}; refusing to cost it at $0")
    price_in, price_out = PINNED_PRICES[key]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
