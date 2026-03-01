FROM ghcr.io/open-webui/pipelines:latest

# System deps for pdf2image
RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Our extra Python deps
COPY requirements.txt /app/pipelines/requirements.txt
RUN pip install --no-cache-dir -r /app/pipelines/requirements.txt

# Bake in the pipeline
COPY colpali-pipeline.py /app/pipelines/colpali-pipeline.py