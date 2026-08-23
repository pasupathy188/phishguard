FROM python:3.11-slim

WORKDIR /app

# Install deps first so Docker layer caching works (deps rarely change, code does)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Non-root user: never run a production API as root in a container
RUN useradd --create-home --shell /bin/bash appuser
COPY app/ ./app/
COPY scripts/ ./scripts/
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
