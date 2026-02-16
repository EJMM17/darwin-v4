# ═══════════════════════════════════════════════════════════
# 🧬 Darwin Agent v4.0 — Dockerfile
# ═══════════════════════════════════════════════════════════

FROM python:3.12-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir redis[hiredis] asyncpg

# App code
COPY darwin_agent/ darwin_agent/
COPY config_example.yaml config_example.yaml

# Create data dirs
RUN mkdir -p /app/data /app/logs

# Non-root user
RUN useradd -m -s /bin/bash darwin && chown -R darwin:darwin /app
USER darwin

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "darwin_agent.main", "--config", "/app/config.yaml"]
