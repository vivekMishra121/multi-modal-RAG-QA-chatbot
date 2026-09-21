FROM python:3.9-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Install system deps for document processing (Tesseract, Ghostscript/Cairo).
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements/base.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY . .
RUN pip install .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD ["python", "-c", "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"]

CMD ["uvicorn", "rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]