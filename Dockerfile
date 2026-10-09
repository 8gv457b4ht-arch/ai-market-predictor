# Python 3.13 = the interpreter the test suite was run with.
FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --home /app app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

COPY backend backend
COPY frontend frontend
COPY scripts scripts
RUN mkdir -p data models backups && chown -R app:app data models backups

# ---------------------------------------------------------------- test stage
# docker build --target test .   -> runs the full pytest suite inside the image
FROM base AS test
COPY backend/requirements-dev.txt backend/requirements-dev.txt
RUN pip install -r backend/requirements-dev.txt
COPY tests tests
COPY pytest.ini docker-compose.yml .env.example Dockerfile .gitignore ./
COPY deploy deploy
# byte-compile as root (the app user cannot write into /app), then run the suite unprivileged
RUN python -m compileall -q backend scripts tests
USER app
RUN python -m pytest

# --------------------------------------------------------------- runtime stage
FROM base AS runtime
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=8s --start-period=30s --retries=3 CMD ["python", "scripts/healthcheck.py", "api"]
# Proxy headers are NOT trusted by uvicorn (would make rate limiting spoofable when the port is
# exposed directly); behind the bundled Caddy set TRUST_PROXY_HEADERS=true instead.
CMD ["uvicorn", "backend.app.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", \
     "--no-proxy-headers", "--no-server-header"]
