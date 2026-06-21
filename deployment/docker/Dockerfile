# ==========================================
# 🐳 MULTI-STAGE ENTERPRISE DOCKERFILE
# ==========================================

# Base Stage: Install dependencies
FROM python:3.10-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/
RUN pip install --no-cache-dir \
    -e packages/core_utils \
    -e packages/data_ingestor \
    -e packages/feature_engineer \
    -e packages/trading_environment


# Build Stage: Run linters & tests during container construction
FROM base AS test-runner
RUN pip install --no-cache-dir ruff pytest pytest-cov mypy
RUN ruff check . && pytest --cov=src tests/unit/

# API Stage: Light production image for the FastAPI gateway
FROM base AS api
EXPOSE 8000
CMD ["python", "apps/terminal_api/app.py"]

# Dashboard Stage: Image hosting the Cyber-Quant Terminal dashboard
FROM base AS dashboard
EXPOSE 8501
CMD ["streamlit", "run", "apps/dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
