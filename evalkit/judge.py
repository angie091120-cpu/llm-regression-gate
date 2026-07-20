"""LLM-as-judge scoring (SPEC.md §4).

Scores how well a candidate summary (produced by the system under test)
captures the same meaning as the human-approved reference summary, on a 1-5
scale. Uses the "judge" tier from llm.py, which defaults to a stronger model
(claude-sonnet-5) than the classifier under test (claude-haiku-4-5) --
judging should never be weaker than what it's judging.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from llm import LLMResponse, complete

JUDGE_SYSTEM_PROMPT = """You are grading the output of a customer-support email \
classifier for a fictional SaaS product, NebulaDesk. You will be shown the \
original email, a reference summary that a human reviewer approved as \
correct, and a candidate summary produced by the system under test.

Score 1-5 how well the candidate captures the same meaning and key facts as \
the reference summary:
  5 = equivalent meaning, no material omissions
  4 = equivalent meaning, minor omission/phrasing difference
  3 = mostly right but missing or fuzzing a fact that matters
  2 = captures the general topic but misses the actual ask
  1 = misses or contradicts the reference

Do not penalize differences in wording, length, or language choice alone --
judge meaning, not style."""


class JudgeVerdict(BaseModel):
    score: int = Field(ge=1, le=5)
    reasoning: str


def score_summary(input_text: str, expected_summary: str, actual_summary: str) -> tuple[int, LLMResponse]:
    """Return (score, raw LLMResponse) so callers can fold judge-call tokens/
    latency into their own cost accounting."""
    user_content = (
        f"Original email:\n{input_text}\n\n"
        f"Reference summary (human-approved):\n{expected_summary}\n\n"
        f"Candidate summary (system under test):\n{actual_summary}\n\n"
        "Score the candidate 1-5 by calling the JudgeVerdict tool."
    )
    resp = complete(
        tier="judge",
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
        schema=JudgeVerdict,
    )
    return resp.parsed.score, resp
