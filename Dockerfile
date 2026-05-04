# syntax=docker/dockerfile:1.7

# ============================================
# Stage 1 — builder: instala dependencias en un prefijo aislado
# ============================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

COPY pyproject.toml ./

# Solo dependencias de inference (FastAPI + Uvicorn[standard] + numpy + sklearn + joblib).
# --prefix=/install permite copiar el árbol completo al runtime sin reinstalar.
RUN pip install --prefix=/install ".[inference]"


# ============================================
# Stage 2 — runtime: imagen mínima para servir el modelo
# ============================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONOPTIMIZE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Copia las libs ya instaladas desde el builder
COPY --from=builder /install /usr/local

# Solo se necesita el código de servir y el modelo
COPY app/ app/
COPY models/ models/

# Usuario no-root para reducir superficie de ataque
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()" || exit 1

# Uvicorn con uvloop + httptools para mejor latencia / throughput.
# --workers se ajusta vía variable de entorno (default 1, suficiente para 1 vCPU).
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--no-access-log"]
