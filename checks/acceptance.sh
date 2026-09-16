#!/usr/bin/env bash
# Acceptance gate for llm-regression-gate — runs the machine-checkable ACs from SPEC.md §7.
# AC1/AC3 are free (mocked); AC2 needs ANTHROPIC_API_KEY (skipped with a warning when a
# fresh report is absent and no key is set); AC5 needs docker (fails loudly if missing).
#
# Interpreter: pass EXP_PYTHON to choose one, otherwise python3 from PATH, which is
# what this script has always used. The first line of output names the interpreter
# that actually ran, so a pass can be attributed to a known environment.
#
#   bash checks/acceptance.sh
#   EXP_PYTHON=/path/to/python bash checks/acceptance.sh
#
# AC5 runs pytest inside the image, on the image's own interpreter; EXP_PYTHON does
# not reach it, which is the point of building the image.
set -u
cd "$(dirname "$0")/.."
fail=0

if [ -n "${EXP_PYTHON:-}" ]; then PY="$EXP_PYTHON"; PY_SRC=EXP_PYTHON; else PY=python3; PY_SRC=default; fi
PY_PATH="$(command -v "$PY" 2>/dev/null || echo "$PY")"
PY_VER="$("$PY" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null)" || { echo "interpreter: $PY_PATH ($PY_SRC) -- unusable"; exit 1; }
echo "interpreter: $PY_PATH  python=$PY_VER  source=$PY_SRC"

step() { printf '\n== %s ==\n' "$1"; }

step "AC1: pytest, no API key, fully mocked"
if (unset ANTHROPIC_API_KEY; "$PY" -m pytest -q); then echo "AC1 PASS"; else echo "AC1 FAIL"; fail=1; fi

step "AC2: eval report shape"
if [ -f eval_report.json ]; then
  "$PY" - <<'EOF' || fail=1
import json, sys
r = json.load(open("eval_report.json"))
assert isinstance(r["pass_rate"], float) and 0.0 <= r["pass_rate"] <= 1.0, "pass_rate not a 0-1 float"
assert len(r["cases"]) >= 60, f"only {len(r['cases'])} cases (<60)"
for c in r["cases"]:
    for k in ("category_match", "judge_score", "latency_ms", "tokens"):
        assert k in c, f"case {c.get('id')} missing {k}"
print("AC2 PASS")
EOF
else
  echo "AC2 SKIP: eval_report.json absent — run 'python -m evalkit.run_eval' with ANTHROPIC_API_KEY first"; fail=1
fi

step "AC3: degraded-fixture diff exits non-zero and names planted regressions"
if "$PY" -m evalkit.diff --baseline tests/fixtures/baseline_report.json \
     --candidate tests/fixtures/degraded_report.json \
     --warn-threshold 0.03 --critical-threshold 0.08 --out diff_report.json; then
  echo "AC3 FAIL: diff exited 0 on a degraded candidate"; fail=1
else
  "$PY" - <<'EOF' || fail=1
import json
d = json.load(open("diff_report.json"))
planted = set(json.load(open("tests/fixtures/known_regressions.json")))
found = {r["id"] for r in d["regressions"]}
assert d["severity"] == "critical", f"severity={d['severity']}"
assert planted <= found, f"missing planted regressions: {planted - found}"
print("AC3 PASS")
EOF
fi

step "AC5: docker build + in-container pytest"
if command -v docker >/dev/null 2>&1; then
  if docker build -t llm-regression-gate:test . && docker run --rm llm-regression-gate:test pytest -q; then
    echo "AC5 PASS"
  else echo "AC5 FAIL"; fail=1; fi
else
  echo "AC5 FAIL: docker not available on this machine"; fail=1
fi

step "AC4 (manual): open a test PR touching prompts/** and verify eval-gate + scorecard comment"
if [ ! -f .github/workflows/eval-gate.yml ]; then
  echo "AC4 FAIL: eval-gate.yml missing"; fail=1
elif grep -q 'TODO(S2)' .github/workflows/eval-gate.yml; then
  echo "AC4 FAIL: eval-gate.yml is still a stub (TODO(S2) marker present)"; fail=1
else
  echo "AC4 PARTIAL: workflow file present and implemented (no stub marker) -- this script can't drive GitHub Actions itself"
  echo "  end-to-end is no longer hypothetical: every push to PR #4 runs eval-gate.yml on the pull_request event and leaves a scorecard comment on the PR. Read the newest comment for the current numbers rather than any figure quoted here; the README CI-gate section explains them"
  echo "  not yet shown on any PR: a critical regression turning the check red -- the only red eval-gate run to date (PR #2, 2026-08-24) was the missing-ANTHROPIC_API_KEY guard failing loud, not a regression. Branch protection still does not list this check as required"
fi

exit $fail
