# Credit Risk Scorer — FastAPI + XGBoost bundle
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8765

WORKDIR /app

# System deps for scientific wheels (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# XGBoost 3.x pulls nvidia-nccl-cu12 via pip metadata; CPU inference does not need it.
RUN pip install --upgrade pip \
    && grep -v '^xgboost==' requirements.txt > /tmp/requirements-no-xgb.txt \
    && pip install -r /tmp/requirements-no-xgb.txt \
    && pip install --no-deps "xgboost==3.1.3"

COPY . .

RUN mkdir -p /app/data \
    && useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD sh -c 'curl -f "http://127.0.0.1:${UVICORN_PORT:-8765}/docs" >/dev/null 2>&1 || exit 1'

CMD ["sh", "-c", "exec uvicorn app:app --host ${UVICORN_HOST} --port ${UVICORN_PORT}"]
