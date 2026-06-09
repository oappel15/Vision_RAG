# FROM ghcr.io/open-webui/open-webui:main

# # Poppler for PDF → image conversion
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     poppler-utils \
#     && rm -rf /var/lib/apt/lists/*

# # ColQwen visual pipeline dependencies
# RUN pip install --no-cache-dir --upgrade \
#     pdf2image \
#     qdrant-client \
#     colpali-engine \
#     Pillow

# # Pure CPU PyTorch
# RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# # ← THIS IS THE VISION PIPELINE (the important part you were missing)
# RUN mkdir -p /app/backend/pipelines && \
#     wget -O /app/backend/pipelines/colpali-pipeline.py \
#     https://raw.githubusercontent.com/sancelot/open-webui-multimodal-pipeline/main/colpali-pipeline.py.



FROM ghcr.io/open-webui/open-webui:main

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --upgrade \
    pdf2image \
    qdrant-client \
    colpali-engine \
    Pillow