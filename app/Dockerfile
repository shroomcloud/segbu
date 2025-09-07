FROM python:3.12.11-slim AS deps

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

RUN uv venv /opt/venv \
    && UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --frozen --no-dev

#####################

FROM python:3.12.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* 

WORKDIR /app
COPY --from=deps /opt/venv /opt/venv
COPY --from=deps /app/pyproject.toml ./ 
COPY ./src ./src

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["python", "src/main.py"]

