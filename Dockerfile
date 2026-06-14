# ── Stage 1: build the React frontend ────────────────────────────────────────
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build
# Output: /app/frontend/dist/


# ── Stage 2: Python runtime ───────────────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

# System dependencies (pandas needs these for C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY backend/ ./backend/

# Dataset (static reference data — baked into image at build time)
COPY data/ ./data/

# Compiled React app (served by FastAPI as static files at runtime)
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# PDF output directory (mount a persistent volume here in production)
RUN mkdir -p backend/output

EXPOSE 8000

# --workers 1 is required: RunStore uses in-memory asyncio.Queue — multiple
# workers each get their own memory, so SSE streams and run state would not
# be shared across workers. See CLAUDE.md Known Limitations for the Redis
# pub/sub upgrade path when horizontal scaling is needed.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
