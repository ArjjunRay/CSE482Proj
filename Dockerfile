FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip wheel --wheel-dir /wheels .
RUN python -m pip install --no-cache-dir /wheels/*
RUN nascar-performance demo-data --output-dir /build/data/demo
RUN nascar-performance train \
    --data-dir /build/data/demo \
    --model-out /artifacts/nascar_rank_model.joblib \
    --metrics-out /artifacts/metrics.json

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/nascar_rank_model.joblib \
    PORT=8000

RUN addgroup --system app && adduser --system --ingroup app app
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/* && rm -rf /wheels
COPY --from=builder --chown=app:app /artifacts ./models
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"
CMD ["sh", "-c", "uvicorn nascar_performance.api:app --host 0.0.0.0 --port ${PORT}"]

