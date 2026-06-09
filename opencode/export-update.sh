#!/bin/bash
#
# Vision RAG — Export Update Package (Source Machine)
#
# Exports updated Docker images and project code to a DOK or portable drive.
# Run this on the ONLINE source machine after making code changes.
#
# Usage:
#   bash opencode/export-update.sh /mnt/d/VisionRAG_Update
#
# The export can then be copied to a DOK and transferred to the offline target.

set -euo pipefail

EXPORT_DIR="${1:?Usage: $0 <export_directory>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() { echo -e "${BLUE}[INFO]${NC} $*"; }
ok() { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

echo "========================================"
echo "  Vision RAG — Export Update Package"
echo "========================================"
echo ""
info "Export directory: ${EXPORT_DIR}"
info "Source repo: ${REPO_DIR}"
echo ""

# Check prerequisites
if ! command -v docker &>/dev/null; then
    error "Docker is not installed or not in PATH"
    exit 1
fi

if ! docker info &>/dev/null; then
    error "Docker daemon is not running"
    exit 1
fi

# Create export directory
mkdir -p "${EXPORT_DIR}"
if [ -d "${EXPORT_DIR}/images" ] || [ -d "${EXPORT_DIR}/code" ]; then
    warn "Export directory already contains data. Removing old export..."
    rm -rf "${EXPORT_DIR}/images" "${EXPORT_DIR}/code" "${EXPORT_DIR}/manifest.json"
fi

mkdir -p "${EXPORT_DIR}/images"
mkdir -p "${EXPORT_DIR}/code"

# ── 1. Collect all images from docker-compose ─────────────────────────
echo ""
info "[1/4] Discovering Docker images..."

# Get all image names from compose config
IMAGES=$(cd "${REPO_DIR}" && docker compose config --images 2>/dev/null | sort -u | tr '\n' ' ')

# Add utility images needed for offline volume backup/restore operations
IMAGES="${IMAGES} busybox:latest alpine:latest"

if [ -z "${IMAGES}" ]; then
    error "No images found. Is docker-compose.yml valid?"
    exit 1
fi

info "Images to export:"
for img in ${IMAGES}; do
    # Check if image exists locally
    if docker inspect "${img}" &>/dev/null; then
        size=$(docker images --format "{{.Size}}" "${img}")
        echo "  - ${img} (${size})"
    else
        warn "  - ${img} (NOT FOUND locally — will be pulled on target or skipped)"
    fi
done

# ── 2. Export Docker images ───────────────────────────────────────────
echo ""
info "[2/4] Exporting Docker images (this may take a while)..."

for img in ${IMAGES}; do
    if ! docker inspect "${img}" &>/dev/null; then
        warn "Skipping ${img} — not found locally"
        continue
    fi
    
    # Create safe filename from image name
    safe_name=$(echo "${img}" | tr '/:' '-')
    out_file="${EXPORT_DIR}/images/${safe_name}.tar.gz"
    
    info "Exporting ${img}..."
    docker save "${img}" | gzip > "${out_file}"
    size=$(du -h "${out_file}" | cut -f1)
    ok "  Saved: ${safe_name}.tar.gz (${size})"
done

# ── 3. Export project code ────────────────────────────────────────────
echo ""
info "[3/4] Exporting project code..."

# Use rsync or tar to copy code, excluding sensitive/large files
info "Copying repository files..."

# List of files to exclude from code export
EXCLUDES=(
    '.git'
    '.env'
    '__pycache__'
    '*.pyc'
    '.ruff_cache'
    '*.egg-info'
    'node_modules'
    '.pytest_cache'
    '.mypy_cache'
    'docker-export'
    'dok-export'
    'vision-rag-backups'
)

# Build rsync exclude args
RSYNC_EXCLUDES=()
for ex in "${EXCLUDES[@]}"; do
    RSYNC_EXCLUDES+=(--exclude="${ex}")
done

# Copy everything from repo to code/ directory
rsync -a --delete "${RSYNC_EXCLUDES[@]}" "${REPO_DIR}/" "${EXPORT_DIR}/code/"

# Count what we copied
file_count=$(find "${EXPORT_DIR}/code" -type f | wc -l)
dir_size=$(du -sh "${EXPORT_DIR}/code" | cut -f1)
ok "  Copied ${file_count} files (${dir_size})"

# ── 4. Calculate checksums ────────────────────────────────────────────
echo ""
info "[4/4] Generating manifest..."

MANIFEST="${EXPORT_DIR}/manifest.json"

# Build JSON manifest
cat > "${MANIFEST}" << EOF
{
  "export_type": "vision-rag-update",
  "export_date": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "source_repo": "${REPO_DIR}",
  "images": [
EOF

first=true
for img in ${IMAGES}; do
    safe_name=$(echo "${img}" | tr '/:' '-')
    tar_file="${EXPORT_DIR}/images/${safe_name}.tar.gz"
    
    if [ -f "${tar_file}" ]; then
        checksum=$(sha256sum "${tar_file}" | cut -d' ' -f1)
        size=$(stat -c%s "${tar_file}" 2>/dev/null || stat -f%z "${tar_file}" 2>/dev/null || echo "0")
        
        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> "${MANIFEST}"
        fi
        
        cat >> "${MANIFEST}" << EOF
    {
      "name": "${img}",
      "file": "images/${safe_name}.tar.gz",
      "sha256": "${checksum}",
      "size_bytes": ${size}
    }
EOF
    fi
done

cat >> "${MANIFEST}" << EOF

  ],
  "code": {
    "file_count": ${file_count},
    "size_human": "${dir_size}"
  }
}
EOF

ok "  Manifest saved: manifest.json"

# ── 5. Copy import script ─────────────────────────────────────────────
echo ""
info "Copying import script..."
cp "${SCRIPT_DIR}/import-update.sh" "${EXPORT_DIR}/import-update.sh"
chmod +x "${EXPORT_DIR}/import-update.sh"
ok "  import-update.sh ready"

# ── 6. Summary ────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Export Complete!"
echo "========================================"
echo ""
TOTAL_SIZE=$(du -sh "${EXPORT_DIR}" | cut -f1)
info "Total export size: ${TOTAL_SIZE}"
echo ""
info "Export contents:"
echo "  ${EXPORT_DIR}/"
echo "    images/          — Docker images (.tar.gz)"
echo "    code/            — Updated project code"
echo "    manifest.json    — Inventory with checksums"
echo "    import-update.sh — Import script for target machine"
echo ""
info "Next steps:"
echo "  1. Copy this folder to your DOK/external drive"
echo "  2. Transfer to the offline target machine"
echo "  3. On the target, run: bash ${EXPORT_DIR}/import-update.sh"
echo ""
