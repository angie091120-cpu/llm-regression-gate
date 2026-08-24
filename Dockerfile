# AC5: `docker build -t llm-regression-gate:test .` then
# `docker run --rm llm-regression-gate:test pytest -q` must both exit 0.
# Default entrypoint runs the free, fully-mocked test suite (AC1) -- no
# ANTHROPIC_API_KEY needed. Real-API paths (evalkit.run_eval, evalkit.judge)
# need ANTHROPIC_API_KEY passed at `docker run` time (see README.md).
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY . .

CMD ["pytest", "-q"]
