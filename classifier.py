"""Email classification for NebulaDesk (fictional B2B SaaS) support tickets.

classify_email() is the feature under test (see SPEC.md §1). The prompt --
system prompt, few-shot examples, output schema description -- is never
hardcoded here; it is loaded from a versioned YAML file in prompts/. Testing
a new prompt means adding prompts/v2.yaml and passing --prompt-version v2 to
the eval CLI, with zero code changes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, PrivateAttr

from llm import LLMResponse, complete

PROMPTS_DIR = Path(__file__).parent / "prompts"

Category = Literal["billing", "technical", "account", "general"]


class ClassificationResult(BaseModel):
    """Output of classify_email(). Matches SPEC.md §1's four-category contract.

    `llm_response` carries the raw LLMResponse (model, tokens, latency) used to
    produce this result. It is a private attribute -- excluded from
    model_dump()/model_dump_json() -- so ClassificationResult's public shape
    stays exactly {category, summary} as specced, while evalkit (which needs
    the usage/latency numbers for the four-dimension eval report) can still
    read it via the `llm_response` property without a second, duplicate call.
    """

    category: Category
    summary: str
    _llm_response: LLMResponse | None = PrivateAttr(default=None)

    @property
    def llm_response(self) -> LLMResponse | None:
        return self._llm_response


class PromptConfig(BaseModel):
    version: str
    model: str
    created_at: str
    system_prompt: str
    few_shot_examples: list[dict]
    output_schema: dict


def load_prompt_config(version: str) -> PromptConfig:
    """Load prompts/<version>.yaml. Fails loudly on a missing file or a
    version field that doesn't match the filename -- a mismatch almost
    always means the wrong file got copy-pasted."""
    path = PROMPTS_DIR / f"{version}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no prompt config at {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None or "version" not in raw:
        raise ValueError(f"{path} is missing required field 'version'")
    if raw["version"] != version:
        raise ValueError(f"{path} declares version={raw['version']!r}, expected {version!r}")
    return PromptConfig(**raw)


def _few_shot_messages(examples: list[dict]) -> list[dict]:
    messages: list[dict] = []
    for ex in examples:
        messages.append({"role": "user", "content": ex["input"]})
        messages.append(
            {
                "role": "assistant",
                "content": f"category: {ex['output']['category']}\nsummary: {ex['output']['summary']}",
            }
        )
    return messages


def classify_email(email_text: str, prompt_config: PromptConfig) -> ClassificationResult:
    """Classify a support email into one of four categories plus a one-sentence
    summary. `prompt_config` is fully external/configurable (SPEC.md §1) --
    this function has no hardcoded prompt text."""
    if not email_text or not email_text.strip():
        raise ValueError("email_text must not be empty")
    messages = _few_shot_messages(prompt_config.few_shot_examples)
    messages.append({"role": "user", "content": email_text})
    resp = complete(
        tier="classifier",
        system=prompt_config.system_prompt,
        messages=messages,
        schema=ClassificationResult,
    )
    result = resp.parsed
    result._llm_response = resp
    return result
