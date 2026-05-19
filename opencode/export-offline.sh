#!/bin/bash
#
# Vision RAG — Offline Export
#
# Exports everything needed to run Vision RAG on an offline machine:
#   1. Git repo (code, configs, skills, MCP server)
#   2. Docker images (all 7 services)
#   3. Docker volumes (HF model cache, Qdrant data, Ollama models, Open WebUI data)
#
# Usage:
#   bash opencode/export-offline.sh /mnt/e/VisionRAG_Export
#
# On the target machine:
#   bash import-offline.sh
#
# Total export size: ~25-30 GB (mostly Docker images + HF model cache)

set -euo pipefail

EXPORT_DIR="${1:?Usage: $0 <export_directory>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"

echo "=== Vision RAG — Offline Export ==="
echo "Export directory: ${EXPORT_DIR}"
echo ""

mkdir -p "${EXPORT_DIR}"

# ── 1. Copy the repo ─────────────────────────────────────────────────
echo "[1/4] Copying repo..."
REPO_EXPORT="${EXPORT_DIR}/Vision_RAG_Git"
if [ -d "${REPO_EXPORT}" ]; then
    echo "  Removing old export..."
    rm -rf "${REPO_EXPORT}"
fi
# Use git archive to get clean copy, then copy untracked essentials
git -C "${REPO_DIR}" archive HEAD | tar -x -C "${EXPORT_DIR}" --one-top-level=Vision_RAG_Git
# Copy the opencode dir (may have been just added, not yet committed)
cp -r "${REPO_DIR}/opencode" "${REPO_EXPORT}/opencode" 2>/dev/null || true
echo "  Done: ${REPO_EXPORT}"

# ── 2. Copy PDFs ─────────────────────────────────────────────────────
echo "[2/4] Copying PDF files..."
mkdir -p "${REPO_EXPORT}/my-pdfs"
cp "${REPO_DIR}"/my-pdfs/*.pdf "${REPO_EXPORT}/my-pdfs/" 2>/dev/null || true
PDF_COUNT=$(ls -1 "${REPO_EXPORT}/my-pdfs/"*.pdf 2>/dev/null | wc -l)
echo "  Copied ${PDF_COUNT} PDF files"

# ── 3. Export Docker images ──────────────────────────────────────────
echo "[3/4] Exporting Docker images (this takes a while)..."
IMAGES_FILE="${EXPORT_DIR}/docker-images.tar.gz"

IMAGES=$(cd "${REPO_DIR}" && docker compose config --images 2>/dev/null | tr '\n' ' ')
echo "  Images: ${IMAGES}"
echo "  Saving to ${IMAGES_FILE}..."
docker save ${IMAGES} | gzip > "${IMAGES_FILE}"
IMAGE_SIZE=$(du -h "${IMAGES_FILE}" | cut -f1)
echo "  Done: ${IMAGE_SIZE}"

# ── 4. Export Docker volumes ─────────────────────────────────────────
echo "[4/4] Exporting Docker volumes..."
VOLUMES_DIR="${EXPORT_DIR}/volumes"
mkdir -p "${VOLUMES_DIR}"

for vol in vision_rag_git_hf-cache vision_rag_git_ollama vision_rag_git_open-webui vision_rag_git_qdrant_data; do
    short_name="${vol#vision_rag_git_}"
    out_file="${VOLUMES_DIR}/${short_name}.tar.gz"
    echo "  Exporting ${vol} → ${short_name}.tar.gz..."
    docker run --rm \
        -v "${vol}:/source:ro" \
        -v "${VOLUMES_DIR}:/dest" \
        alpine tar czf "/dest/${short_name}.tar.gz" -C /source .
    vol_size=$(du -h "${out_file}" | cut -f1)
    echo "    Size: ${vol_size}"
done

# ── 5. Create import script ─────────────────────────────────────────
cat > "${EXPORT_DIR}/import-offline.sh" << 'IMPORT_EOF'
#!/bin/bash
#
# Vision RAG — Offline Import
#
# Run this on the target offline machine after copying the export directory.
#
# Usage:
#   cd /path/to/VisionRAG_Export
#   bash import-offline.sh

set -euo pipefail

EXPORT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Vision RAG — Offline Import ==="
echo ""

# 1. Load Docker images
echo "[1/4] Loading Docker images..."
if [ -f "${EXPORT_DIR}/docker-images.tar.gz" ]; then
    docker load < "${EXPORT_DIR}/docker-images.tar.gz"
    echo "  Done."
else
    echo "  WARNING: docker-images.tar.gz not found — skipping."
fi

# 2. Import Docker volumes
echo "[2/4] Importing Docker volumes..."
VOLUMES_DIR="${EXPORT_DIR}/volumes"
if [ -d "${VOLUMES_DIR}" ]; then
    for archive in "${VOLUMES_DIR}"/*.tar.gz; do
        short_name="$(basename "${archive}" .tar.gz)"
        vol_name="vision_rag_git_${short_name}"
        echo "  Importing ${short_name} → ${vol_name}..."
        # Create volume if it doesn't exist
        docker volume create "${vol_name}" >/dev/null 2>&1 || true
        docker run --rm \
            -v "${vol_name}:/dest" \
            -v "${archive}:/source.tar.gz:ro" \
            alpine sh -c "cd /dest && tar xzf /source.tar.gz"
        echo "    Done."
    done
else
    echo "  WARNING: volumes/ directory not found — skipping."
fi

# 3. Copy repo to a working location
echo "[3/4] Vision RAG repo is at: ${EXPORT_DIR}/Vision_RAG_Git"
echo "  Copy it to your projects directory, then:"
echo "    cd Vision_RAG_Git"
echo "    cp .env.example .env   # Edit with your API key"
echo "    docker compose up -d   # No --build needed, images are loaded"

# 4. Install OpenCode integration
echo "[4/4] To install OpenCode integration:"
echo "    cd Vision_RAG_Git"
echo "    bash opencode/setup-opencode.sh"
echo "    # Then restart OpenCode"

echo ""
echo "=== Import complete ==="
IMPORT_EOF
chmod +x "${EXPORT_DIR}/import-offline.sh"

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "=== Export complete ==="
echo ""
TOTAL_SIZE=$(du -sh "${EXPORT_DIR}" | cut -f1)
echo "Total export size: ${TOTAL_SIZE}"
echo ""
echo "Contents:"
echo "  ${REPO_EXPORT}/              — Git repo + code + skills"
echo "  ${IMAGES_FILE}               — Docker images"
echo "  ${VOLUMES_DIR}/              — Docker volumes (models, data)"
echo "  ${EXPORT_DIR}/import-offline.sh  — Import script for target machine"
echo ""
echo "To deploy on the offline machine:"
echo "  1. Copy ${EXPORT_DIR} to the target machine"
echo "  2. Run: bash import-offline.sh"
echo "  3. Follow the on-screen instructions"
