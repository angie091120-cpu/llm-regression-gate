#!/usr/bin/env bash
# Acceptance gate for experiments/ -- the reproducibility claims the write-up makes.
#
#   C1  analyze.py --seed <seed> twice in a row produces byte-identical CSVs
#   C2  MANIFEST.json line counts match the raw JSONL files on disk
#   C3  the production test suite is still green and still has the same test count
#   C4  MANIFEST raw cost agrees with experiments/results/cost_ledger.json within 1%
#   C5  the standard-library estimators still reproduce their published worked examples
#   C6  no .env and no API key material is tracked by git
#   C7  experiments/stats.py rejects an unknown flag with exit 2
#
# Every check is machine-checkable; nothing here needs an API key or network.
#
#   bash checks/experiments_acceptance.sh
#   EXP_PYTHON=/path/to/python bash checks/experiments_acceptance.sh
#
# Interpreter: pass EXP_PYTHON, or let the script take the first of
# ~/.venvs/lrg-exp/bin/python (the analysis venv -- scipy, numpy, statsmodels,
# matplotlib), ./.venv/bin/python, ../llm-regression-gate/.venv/bin/python,
# python3. The first line of output names the interpreter that actually ran and
# whether scipy imports in it.
set -u
cd "$(dirname "$0")/.."
fail=0
SEED="${EXP_SEED:-20260920}"
# 46 until 2026-09-19, when the blind re-labelling import added 58: the two new
# agreement estimators against their printed examples and their undefined case
# (7), the two sheet importers and their refusals (10), every branch of the
# gold v2 decision rule on synthetic sheets (17), the gold v2 re-grading,
# family labelling and difference listing (11), the dataset write-out including
# the labeled_at rule (7), and E4's three-rater panel rows (6). C3 asserts the
# count as well as the result, so a test file that stops being collected fails
# here instead of passing quietly.
EXPECTED_TESTS="${EXPECTED_TESTS:-104}"

step() { printf '\n== %s ==\n' "$1"; }
ok()   { echo "$1 PASS"; }
bad()  { echo "$1 FAIL: $2"; fail=1; }

if [ -n "${EXP_PYTHON:-}" ]; then
  PY="$EXP_PYTHON"
elif [ -x "${HOME:-}/.venvs/lrg-exp/bin/python" ]; then
  PY="${HOME}/.venvs/lrg-exp/bin/python"
elif [ -x ".venv/bin/python" ]; then
  PY="$(pwd)/.venv/bin/python"
elif [ -x "../llm-regression-gate/.venv/bin/python" ]; then
  PY="$(cd .. && pwd)/llm-regression-gate/.venv/bin/python"
else
  PY="$(command -v python3)"
fi
PYVER=$("$PY" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null) \
  || { echo "interpreter: $PY -- unusable"; exit 1; }
if "$PY" -c 'import scipy' >/dev/null 2>&1; then HAS_SCIPY=yes; else HAS_SCIPY=no; fi
# Printed before any check result, on purpose. The 2026-09-15 README correction
# started as a 28/29 --cross-check read off the repo .venv and written up as a
# property of the machine; which interpreter ran is no longer left implicit.
echo "interpreter: $PY (python $PYVER, has_scipy=$HAS_SCIPY)"

TABLES="experiments/results/tables"
RAW="experiments/results/raw"

step "C1: analyze.py --seed $SEED is deterministic across two runs"
# Runs into two throwaway directories, so the check never rewrites the
# committed tables/figures/MANIFEST it is supposed to be checking.
if [ ! -d "$RAW" ] || [ -z "$(find "$RAW" -name '*.jsonl' -print -quit 2>/dev/null)" ]; then
  bad "C1" "no raw JSONL under $RAW -- run experiments/runner.py first"
else
  d1=$(mktemp -d); d2=$(mktemp -d)
  if "$PY" -m experiments.analyze --seed "$SEED" --out-dir "$d1" --no-figures >/dev/null 2>/tmp/exp_c1a.log \
     && "$PY" -m experiments.analyze --seed "$SEED" --out-dir "$d2" --no-figures >/dev/null 2>/tmp/exp_c1b.log; then
    h1=$(cd "$d1/tables" && ls *.csv | sort | xargs sha256sum | sed "s| .*/| |")
    h2=$(cd "$d2/tables" && ls *.csv | sort | xargs sha256sum | sed "s| .*/| |")
    n_csv=$(cd "$d1/tables" && ls *.csv | wc -l)
    if [ "$h1" = "$h2" ] && [ "$n_csv" -gt 0 ]; then
      echo "$n_csv CSV files, identical sha256 across two runs"
      ok "C1"
    else
      bad "C1" "CSV hashes differ between runs"
      diff <(echo "$h1") <(echo "$h2") || true
    fi
  else
    bad "C1" "analyze run failed: $(tail -n 3 /tmp/exp_c1a.log; tail -n 3 /tmp/exp_c1b.log)"
  fi
  rm -rf "$d1" "$d2"
fi

step "C2: MANIFEST line counts match the raw files"
if "$PY" - <<'EOF'
import json, sys
from pathlib import Path
manifest = Path("experiments/results/MANIFEST.json")
if not manifest.exists():
    print("C2 FAIL: MANIFEST.json absent"); sys.exit(1)
data = json.loads(manifest.read_text(encoding="utf-8"))
problems = []
for entry in data["raw_files"]:
    path = Path(entry["path"])
    actual = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if actual != entry["lines"]:
        problems.append(f"{path}: manifest {entry['lines']} != file {actual}")
total = sum(e["lines"] for e in data["raw_files"])
if total != data["totals"]["raw_lines"]:
    problems.append(f"totals.raw_lines {data['totals']['raw_lines']} != sum {total}")
if problems:
    print("C2 FAIL: " + "; ".join(problems)); sys.exit(1)
print(f"({len(data['raw_files'])} raw files, {total} lines)")
EOF
then ok "C2"; else bad "C2" "see message above"; fi

step "C3: production test suite still green with $EXPECTED_TESTS tests"
out=$(unset ANTHROPIC_API_KEY; "$PY" -m pytest -q 2>&1 | tail -n 3)
echo "$out"
if echo "$out" | grep -qE "^${EXPECTED_TESTS} passed"; then ok "C3"; else bad "C3" "expected '${EXPECTED_TESTS} passed'"; fi

step "C4: MANIFEST raw cost vs cost_ledger.json within 1%"
if "$PY" - <<'EOF'
import json, sys
from pathlib import Path
manifest = json.loads(Path("experiments/results/MANIFEST.json").read_text(encoding="utf-8"))
ledger_path = Path("experiments/results/cost_ledger.json")
if not ledger_path.exists():
    print("C4 FAIL: experiments/results/cost_ledger.json absent"); sys.exit(1)
ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
m = float(manifest["totals"]["raw_cost_usd"])
l = float(ledger["cumulative_usd"])
if l == 0 and m == 0:
    print("(both zero)"); sys.exit(0)
rel = abs(m - l) / max(abs(l), 1e-12)
print(f"manifest raw ${m:.6f}  ledger ${l:.6f}  relative diff {rel*100:.4f}%")
sys.exit(0 if rel < 0.01 else 1)
EOF
then ok "C4"; else bad "C4" "cost mismatch or missing ledger"; fi

step "C5: standard-library estimators reproduce their worked examples"
if "$PY" -m experiments.stats >/tmp/exp_c5.log 2>&1; then ok "C5"; else bad "C5" "$(tail -n 5 /tmp/exp_c5.log)"; fi

step "C6: no credential material tracked by git"
tracked=$(git ls-files | grep -E '(^|/)\.env$' || true)
if [ -n "$tracked" ]; then bad "C6" ".env is tracked: $tracked"; else
  hits=$(git grep -I -l -E 'sk-ant-[A-Za-z0-9_-]{8,}' -- . ':(exclude)checks/experiments_acceptance.sh' 2>/dev/null || true)
  if [ -n "$hits" ]; then bad "C6" "API key pattern in tracked files: $hits"; else ok "C6"; fi
fi

step "C7: experiments.stats rejects an unknown flag instead of running the default"
c7=0
for flag in --no-such-flag --self-check --corss-check; do
  rc=0
  "$PY" -m experiments.stats "$flag" >/tmp/exp_c7.log 2>&1 || rc=$?
  if [ "$rc" -eq 2 ]; then
    echo "  $flag -> exit 2"
  else
    echo "  $flag -> exit $rc (expected 2)"
    c7=1
  fi
done
if [ "$c7" -eq 0 ]; then ok "C7"; else bad "C7" "an unknown flag did not exit 2"; fi

printf '\n== result ==\n'
[ $fail -eq 0 ] && echo "ALL CHECKS PASSED" || echo "SOME CHECKS FAILED"
exit $fail
