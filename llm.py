"""Provider-agnostic LLM client for structured completions.

Two knobs, both environment variables, no other configuration surface:

  LLM_PROVIDER            "anthropic" (default) or "openai"
  LLM_<TIER>_MODEL        e.g. LLM_CLASSIFIER_MODEL, LLM_JUDGE_MODEL
                           overrides the default model for that tier

"Tier" is a caller-chosen label (this project uses "classifier" and "judge")
that maps to a model name via the table below, so callers never hardcode a
model string. Structured output is enforced via native tool-use (Anthropic)
or function-calling (OpenAI) against a Pydantic schema -- never regex-parsed
out of free text.

API keys (ANTHROPIC_API_KEY / OPENAI_API_KEY) are read by the underlying SDK
clients directly from the environment; this module never touches them.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_DEFAULT_MODELS: dict[str, dict[str, str]] = {
    "anthropic": {"classifier": "claude-haiku-4-5", "judge": "claude-sonnet-5"},
    "openai": {"classifier": "gpt-4o-mini", "judge": "gpt-4o"},
}


@dataclass
class LLMResponse:
    parsed: BaseModel
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


def _provider() -> str:
    return os.environ.get("LLM_PROVIDER", "anthropic").lower()


def _model_for(tier: str) -> str:
    provider = _provider()
    if provider not in _DEFAULT_MODELS:
        raise ValueError(f"unsupported LLM_PROVIDER={provider!r} (expected 'anthropic' or 'openai')")
    defaults = _DEFAULT_MODELS[provider]
    if tier not in defaults:
        raise ValueError(f"unknown tier {tier!r} for provider {provider!r}")
    return os.environ.get(f"LLM_{tier.upper()}_MODEL", defaults[tier])


def complete(tier: str, system: str, messages: list[dict[str, str]], schema: Type[T]) -> LLMResponse:
    """Run one structured completion for `tier` and return a schema-validated result."""
    provider = _provider()
    model = _model_for(tier)  # also validates provider/tier, so the dispatch below can't fall through
    start = time.perf_counter()
    if provider == "anthropic":
        parsed, in_tok, out_tok = _complete_anthropic(model, system, messages, schema)
    else:
        parsed, in_tok, out_tok = _complete_openai(model, system, messages, schema)
    latency_ms = (time.perf_counter() - start) * 1000
    return LLMResponse(parsed=parsed, model=model, input_tokens=in_tok, output_tokens=out_tok, latency_ms=latency_ms)


def _complete_anthropic(model: str, system: str, messages: list[dict[str, str]], schema: Type[T]) -> tuple[T, int, int]:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    tool_name = schema.__name__
    tool = {
        "name": tool_name,
        "description": f"Return output matching the {tool_name} schema. Always call this tool.",
        "input_schema": schema.model_json_schema(),
    }
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=messages,
        tools=[tool],
        tool_choice={"type": "tool", "name": tool_name},
    )
    tool_use = next(b for b in resp.content if b.type == "tool_use")
    parsed = schema.model_validate(tool_use.input)
    return parsed, resp.usage.input_tokens, resp.usage.output_tokens


def _complete_openai(model: str, system: str, messages: list[dict[str, str]], schema: Type[T]) -> tuple[T, int, int]:
    from openai import OpenAI  # optional dependency; only imported when LLM_PROVIDER=openai

    client = OpenAI()  # reads OPENAI_API_KEY from env
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, *messages],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
        },
    )
    content: Any = resp.choices[0].message.content
    parsed = schema.model_validate_json(content)
    usage = resp.usage
    return parsed, usage.prompt_tokens, usage.completion_tokens
