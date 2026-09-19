# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first so code edits don't invalidate the layer.
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps -e .

# --- Runtime image: just the CLI -------------------------------------------
FROM base AS runtime
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser
ENTRYPOINT ["research-assistant"]
CMD ["--help"]

# --- Test image: adds dev dependencies and the test suite -------------------
FROM base AS test
RUN pip install -r requirements-dev.txt
COPY tests ./tests
CMD ["pytest", "-m", "not integration"]
