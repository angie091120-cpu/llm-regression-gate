"""evalkit/cost.py: pricing lookups and ledger accumulation. Pure math + file
I/O, no network involved."""
from __future__ import annotations

import json

import pytest

from evalkit import cost


def test_token_cost_usd_default_pricing():
    # 1,000,000 input + 1,000,000 output tokens at the default haiku rates
    usd = cost.token_cost_usd("claude-haiku-4-5", 1_000_000, 1_000_000)
    assert usd == pytest.approx(1.00 + 5.00)


def test_token_cost_usd_env_override(monkeypatch):
    monkeypatch.setenv("PRICE_CLAUDE_HAIKU_4_5_INPUT", "2.5")
    usd = cost.token_cost_usd("claude-haiku-4-5", 1_000_000, 0)
    assert usd == pytest.approx(2.5)


def test_token_cost_usd_unknown_model_raises():
    with pytest.raises(ValueError):
        cost.token_cost_usd("some-unknown-model", 100, 100)


def test_record_usage_accumulates(tmp_path):
    ledger_path = tmp_path / "cost_ledger.json"
    case_results_1 = [{"cost_usd": 0.01}, {"cost_usd": 0.02}]
    total_1 = cost.record_usage(case_results_1, ledger_path=ledger_path)
    assert total_1 == pytest.approx(0.03)

    case_results_2 = [{"cost_usd": 0.05}]
    total_2 = cost.record_usage(case_results_2, ledger_path=ledger_path)
    assert total_2 == pytest.approx(0.08)

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert len(ledger["runs"]) == 2
    assert ledger["cumulative_usd"] == pytest.approx(0.08)
