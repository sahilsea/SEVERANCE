FROM python:3.12-slim

WORKDIR /app

# tesseract-ocr: on-device OCR for scanned PDFs (ingest/ocr.py).
# build-essential: fallback for any dependency without a prebuilt wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root: the code sandbox's per-process limit (RLIMIT_NPROC) is not
# enforced for root, so running as root would silently weaken it.
RUN useradd --create-home --uid 1000 severance \
    && mkdir -p /app/data \
    && chown -R severance:severance /app
USER severance

# Ollama stays on the Mac itself (Docker on macOS has no GPU access, so a
# model inside the container would be CPU-only and much slower). The
# container reaches it via host.docker.internal, which the network guard
# only accepts because it is explicitly allowlisted here -- it is never
# DNS-resolved to decide. See trust/network_monitor.py.
ENV SEVERANCE_DB_PATH=/app/data/severance.db \
    AGENT_BACKEND=ollama \
    OLLAMA_API_BASE=http://host.docker.internal:11434 \
    SEVERANCE_LOCAL_HOST_ALLOWLIST=host.docker.internal \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
