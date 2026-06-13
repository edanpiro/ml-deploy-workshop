# ─── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ─── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL org.opencontainers.image.title="iris-classifier-api"
LABEL org.opencontainers.image.description="Servicio de inferencia Iris – Taller MLOps UNI 2026"

# Copiar paquetes instalados del builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copiar código fuente y modelo
COPY app/ ./app/
COPY models/ ./models/

# Usuario sin privilegios (buena práctica de seguridad)
RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
