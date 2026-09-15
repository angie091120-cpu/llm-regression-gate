"""Instrumented Anthropic call layer for the pilot experiments.

Why this exists instead of calling `llm.complete()`: the production client
returns `LLMResponse.model` set to the *request* string (`claude-haiku-4-5`),
which is the alias, not the dated snapshot the alias resolved to. One of the
things this study has to be able to state is which snapshot produced each
number, so every call here records `response.model` as returned by the API,
alongside the request string, the full `usage` object and the raw response.

It also adds three things the production path does not have and should not
grow for a portfolio repo: an optional content-hash cache, a per-call
temperature, and structured capture of API errors instead of an exception
that aborts a 1,680-call batch.

Production code is imported, not copied, wherever the two must agree
(`llm._model_for` for tier -> model resolution, `evalkit.cost` for pricing),
so the experiment cannot silently drift from the system it measures.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Type, TypeVar

from pydantic import BaseModel

import llm as production_llm  # noqa: F401  (tier -> model table lives there)

T = TypeVar("T", bound=BaseModel)

DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_RETRIES = 2  # SDK-level retries on 429/5xx; transparent, recorded in meta

_KEY_PATTERN = re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}")
_client: Any = None


def redact(text: str) -> str:
    """Strip anything that looks like an API key out of a string before it is
    written to disk. Error strings from an HTTP client are the one place a
    credential can leak into a committed artifact."""
    return _KEY_PATTERN.sub("sk-ant-<redacted>", text)


def _assert_anthropic() -> None:
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    if provider != "anthropic":
        raise RuntimeError(
            f"experiments/ runs against Anthropic only, got LLM_PROVIDER={provider!r}. "
            "The provider-agnostic path is production code (llm.py); the study "
            "measures one provider on purpose."
        )


def model_for_tier(tier: str) -> str:
    """Resolve tier -> model through the production table (llm.py), so
    LLM_CLASSIFIER_MODEL / LLM_JUDGE_MODEL behave identically here."""
    _assert_anthropic()
    return production_llm._model_for(tier)


def sdk_accepts_temperature() -> bool:
    """anthropic >= 1.0 dropped temperature/top_p/top_k from messages.create.
    Checked by introspection (free, no API call) so the runner can pick a
    transport instead of failing on whichever SDK happens to be installed."""
    import inspect

    import anthropic

    try:
        return "temperature" in inspect.signature(anthropic.resources.messages.Messages.create).parameters
    except Exception:  # noqa: BLE001
        return False


def resolve_temperature_transport(requested: str) -> str:
    if requested != "auto":
        return requested
    return "param" if sdk_accepts_temperature() else "extra_body"


def _get_client() -> Any:
    global _client
    if _client is None:
        import anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set (expected in .env, loaded by runner.py)")
        _client = anthropic.Anthropic(max_retries=DEFAULT_MAX_RETRIES, timeout=DEFAULT_TIMEOUT_S)
    return _client


@dataclass
class CallRecord:
    """One API call (or one cache hit standing in for one). Serialised as one
    JSONL line; field names are the raw-data contract consumed by analyze.py."""

    tier: str
    request_model: str
    response_model: str | None = None
    ok: bool = False
    temperature: float | None = None
    temperature_transport: str = "param"
    max_tokens: int = DEFAULT_MAX_TOKENS
    input_tokens: int = 0
    output_tokens: int = 0
    usage: dict = field(default_factory=dict)
    latency_ms: float | None = None
    stop_reason: str | None = None
    parsed: dict | None = None
    raw_response_json: dict | None = None
    error_type: str | None = None
    error_message: str | None = None
    status_code: int | None = None
    request_id: str | None = None
    cache_hit: bool = False
    request_hash: str = ""
    timestamp_utc: str = ""
    has_thinking_block: bool = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def request_hash(model: str, system: str, messages: list[dict], schema: Type[T], temperature: float | None, max_tokens: int) -> str:
    payload = {
        "model": model,
        "system": system,
        "messages": messages,
        "schema": schema.model_json_schema(),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _dump_response(resp: Any) -> dict:
    for attempt in (lambda: resp.model_dump(mode="json"), lambda: json.loads(resp.to_json())):
        try:
            return attempt()
        except Exception:  # noqa: BLE001 -- fall through to the next serialiser
            continue
    return {"_serialisation_failed": True, "repr": redact(repr(resp))[:4000]}


def call_structured(
    tier: str,
    system: str,
    messages: list[dict],
    schema: Type[T],
    temperature: float | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    cache_dir: Path | None = None,
    temperature_transport: str = "param",
    model: str | None = None,
) -> CallRecord:
    """Run one structured (tool-use) completion and return a CallRecord.

    Never raises for an API-side failure: a 429/500/timeout comes back as
    `ok=False` with the error captured, so a long batch records the failure
    and keeps going. Programming errors (bad tier, missing key) still raise.

    `temperature_transport` decides how a temperature reaches the wire:
    "param" passes it as an SDK keyword (absent from anthropic>=1.0),
    "extra_body" puts it in the request body regardless of SDK version. The
    two answer different questions -- whether the SDK accepts the parameter,
    and whether the model does.

    `model` overrides the tier table for the one case where a tier is not a
    fixed model: E2 sends the same pairwise prompt to two judge models, and the
    model is then part of the design rather than a configuration default. The
    provider guard still applies, and `response_model` is still read off the
    response, so an override cannot silently become "whatever the alias points
    at today" in the raw data.
    """
    if model is None:
        model = model_for_tier(tier)
    else:
        _assert_anthropic()
    temperature_transport = resolve_temperature_transport(temperature_transport)
    rhash = request_hash(model, system, messages, schema, temperature, max_tokens)
    record = CallRecord(
        tier=tier,
        request_model=model,
        temperature=temperature,
        temperature_transport=temperature_transport if temperature is not None else "none",
        max_tokens=max_tokens,
        request_hash=rhash,
        timestamp_utc=_now_iso(),
    )

    if cache_dir is not None:
        cached = _cache_read(cache_dir, rhash)
        if cached is not None:
            cached.update({"cache_hit": True, "timestamp_utc": record.timestamp_utc, "latency_ms": None})
            return CallRecord(**{k: v for k, v in cached.items() if k in CallRecord.__dataclass_fields__})

    tool_name = schema.__name__
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
        "tools": [
            {
                "name": tool_name,
                "description": f"Return output matching the {tool_name} schema. Always call this tool.",
                "input_schema": schema.model_json_schema(),
            }
        ],
        "tool_choice": {"type": "tool", "name": tool_name},
    }
    if temperature is not None:
        if temperature_transport == "extra_body":
            kwargs["extra_body"] = {"temperature": temperature}
        elif temperature_transport == "param":
            kwargs["temperature"] = temperature
        else:
            raise ValueError(f"unknown temperature_transport={temperature_transport!r}")

    client = _get_client()
    start = time.perf_counter()
    try:
        resp = client.messages.create(**kwargs)
    except Exception as exc:  # noqa: BLE001 -- deliberate: one bad call must not kill the batch
        record.latency_ms = (time.perf_counter() - start) * 1000
        record.ok = False
        record.error_type = type(exc).__name__
        record.error_message = redact(str(exc))[:2000]
        record.status_code = getattr(exc, "status_code", None)
        record.request_id = getattr(exc, "request_id", None)
        body = getattr(exc, "body", None)
        if body is not None:
            try:
                record.raw_response_json = {"error_body": json.loads(redact(json.dumps(body, ensure_ascii=False)))}
            except Exception:  # noqa: BLE001
                record.raw_response_json = {"error_body_repr": redact(repr(body))[:2000]}
        return record

    record.latency_ms = (time.perf_counter() - start) * 1000
    record.response_model = getattr(resp, "model", None)
    record.stop_reason = getattr(resp, "stop_reason", None)
    record.raw_response_json = _dump_response(resp)
    usage = getattr(resp, "usage", None)
    if usage is not None:
        try:
            record.usage = usage.model_dump(mode="json")
        except Exception:  # noqa: BLE001
            record.usage = {"input_tokens": getattr(usage, "input_tokens", 0), "output_tokens": getattr(usage, "output_tokens", 0)}
        record.input_tokens = int(record.usage.get("input_tokens") or 0)
        record.output_tokens = int(record.usage.get("output_tokens") or 0)
    record.has_thinking_block = any(getattr(b, "type", "") in ("thinking", "redacted_thinking") for b in getattr(resp, "content", []))

    tool_use = next((b for b in resp.content if getattr(b, "type", "") == "tool_use"), None)
    if tool_use is None:
        record.ok = False
        record.error_type = "NoToolUseBlock"
        record.error_message = f"response contained no tool_use block (stop_reason={record.stop_reason})"
        return record
    try:
        parsed = schema.model_validate(tool_use.input)
    except Exception as exc:  # noqa: BLE001
        record.ok = False
        record.error_type = f"SchemaValidation:{type(exc).__name__}"
        record.error_message = redact(str(exc))[:2000]
        record.parsed = dict(tool_use.input) if isinstance(tool_use.input, dict) else None
        return record

    record.ok = True
    record.parsed = parsed.model_dump(mode="json")
    if cache_dir is not None:
        _cache_write(cache_dir, rhash, record)
    return record


def _cache_path(cache_dir: Path, rhash: str) -> Path:
    return cache_dir / rhash[:2] / f"{rhash}.json"


def _cache_read(cache_dir: Path, rhash: str) -> dict | None:
    path = _cache_path(cache_dir, rhash)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- a corrupt cache entry is a cache miss, never a crash
        return None


def _cache_write(cache_dir: Path, rhash: str, record: CallRecord) -> None:
    path = _cache_path(cache_dir, rhash)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True), encoding="utf-8")
