"""Cheap empirical probes of API behaviour that the study has to state as fact.

Four claims appear in the write-up and none of them should rest on a doc page:

  P1  Whether a sampling parameter (`temperature`) reaches Haiku 4.5 and
      Sonnet 5 at all, and which layer refuses it -- the SDK or the model.
      Sent two ways: as an SDK keyword ("param") and inside the request body
      ("extra_body"), which bypasses the SDK signature entirely.
  P2  What `response.model` returns for each alias.
  P3  Whether a judge response carries thinking tokens (usage fields and
      content block types).
  P4  The installed SDK version, recorded with every probe, because P1's
      answer depends on it.

Six short calls per interpreter, about a cent. Results land in
experiments/results/probes/probes_<sdk>_<ts>.json and are quoted in
experiments/COST_CALIBRATION.md.

    python -m experiments.probes
"""
from __future__ import annotations

import inspect
import json
from dataclasses import asdict
from datetime import datetime, timezone

from experiments.client import call_structured
from experiments.runner import REPO_ROOT, assert_prices_current, cost_for, load_dotenv

PROBE_DIR = REPO_ROOT / "experiments" / "results" / "probes"

TINY_EMAIL = "Hi, my invoice for March charged me twice. Please refund the duplicate."
TINY_JUDGE_USER = (
    f"Original email:\n{TINY_EMAIL}\n\n"
    "Reference summary (human-approved):\nCustomer reports a duplicate charge on the March invoice and asks for a refund.\n\n"
    "Candidate summary (system under test):\nCustomer was billed twice in March and wants the extra charge returned.\n\n"
    "Score the candidate 1-5 by calling the JudgeVerdict tool."
)

PROBES = [
    ("haiku_baseline_no_temperature", "classifier", None, "param"),
    ("sonnet_baseline_no_temperature", "judge", None, "param"),
    ("haiku_temperature0_sdk_param", "classifier", 0.0, "param"),
    ("sonnet_temperature0_sdk_param", "judge", 0.0, "param"),
    ("haiku_temperature0_extra_body", "classifier", 0.0, "extra_body"),
    ("sonnet_temperature0_extra_body", "judge", 0.0, "extra_body"),
]


def _sdk_info() -> dict:
    import anthropic

    try:
        params = list(inspect.signature(anthropic.Anthropic().messages.create).parameters)
    except Exception as exc:  # noqa: BLE001
        params = [f"<introspection failed: {exc}>"]
    return {
        "anthropic_version": getattr(anthropic, "__version__", "unknown"),
        "messages_create_accepts_temperature": "temperature" in params,
        "messages_create_params": params,
    }


def _probe(name: str, tier: str, temperature: float | None, transport: str) -> dict:
    from classifier import ClassificationResult
    from evalkit.judge import JUDGE_SYSTEM_PROMPT, JudgeVerdict

    if tier == "classifier":
        system = "You are an email triage assistant. Classify into billing, technical, account or general and write a one-sentence summary."
        messages = [{"role": "user", "content": TINY_EMAIL}]
        schema = ClassificationResult
    else:
        system = JUDGE_SYSTEM_PROMPT
        messages = [{"role": "user", "content": TINY_JUDGE_USER}]
        schema = JudgeVerdict

    rec = call_structured(
        tier, system, messages, schema, temperature=temperature, max_tokens=512, temperature_transport=transport
    )
    out = asdict(rec)
    out["probe"] = name
    out["cost_usd"] = round(cost_for(rec), 8)
    out["content_block_types"] = [b.get("type") for b in (rec.raw_response_json or {}).get("content", []) if isinstance(b, dict)]
    return out


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    assert_prices_current()
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    sdk = _sdk_info()
    print(f"anthropic SDK {sdk['anthropic_version']} (messages.create accepts temperature: {sdk['messages_create_accepts_temperature']})\n")

    results = []
    total = 0.0
    for name, tier, temp, transport in PROBES:
        res = _probe(name, tier, temp, transport)
        res["anthropic_version"] = sdk["anthropic_version"]
        total += res["cost_usd"]
        results.append(res)
        verdict = "ok" if res["ok"] else f"{res['error_type']} status={res['status_code']}"
        print(
            f"{name:34s} temperature={temp} via {transport:10s} -> {verdict}\n"
            f"    response.model = {res['response_model']}\n"
            f"    usage          = {json.dumps(res['usage'], ensure_ascii=False)}\n"
            f"    blocks         = {res['content_block_types']} thinking_block={res['has_thinking_block']}"
        )
        if not res["ok"]:
            print(f"    error          = {(res['error_message'] or '')[:400]}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = PROBE_DIR / f"probes_sdk{sdk['anthropic_version']}_{ts}.json"
    out_path.write_text(
        json.dumps({"generated_utc": ts, "sdk": sdk, "total_cost_usd": round(total, 8), "probes": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\ntotal probe cost ${total:.6f} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
