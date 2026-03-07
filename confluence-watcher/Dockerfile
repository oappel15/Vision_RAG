FROM python:3.11-slim

RUN pip install --no-cache-dir requests playwright qdrant-client \
    && playwright install chromium --with-deps

WORKDIR /app
COPY watcher.py .

CMD ["python", "watcher.py"]
