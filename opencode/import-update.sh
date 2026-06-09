#!/bin/bash
#
# Vision RAG — Import Update Package (Target/Offline Machine)
#
# Run this on the OFFLINE target machine from the DOK export directory.
# It loads updated Docker images and code while preserving existing volumes.
#
# Usage:
#   cd /mnt/d/VisionRAG_Update   (or wherever the DOK is mounted)
#   bash import-update.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info() { echo -e "${BLUE}[INFO]${NC} $*"; }
ok() { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }
ask() { echo -e "${CYAN}[?]${NC} $*"; }

echo "========================================"
echo "  Vision RAG — Import Update Package"
echo "========================================"
echo ""
info "Export directory: ${SCRIPT_DIR}"
echo ""

# ── 0. Helper functions ───────────────────────────────────────────────

# Detect existing Docker volume prefix
detect_volume_prefix() {
    # Look for any volume ending in _hf-cache
    local vol
    vol=$(docker volume ls --format "{{.Name}}" | grep "_hf-cache$" | head -n 1)
    if [ -n "${vol}" ]; then
        # Extract prefix (remove _hf-cache suffix)
        echo "${vol%_hf-cache}"
        return 0
    fi
    return 1
}

# Check if critical volumes exist
check_critical_volumes() {
    local prefix="$1"
    local missing=()
    
    for vol in hf-cache qdrant_data ollama open-webui; do
        local full_name="${prefix}_${vol}"
        if ! docker volume inspect "${full_name}" &>/dev/null; then
            missing+=("${full_name}")
        fi
    done
    
    if [ ${#missing[@]} -gt 0 ]; then
        error "Missing critical volumes:"
        for v in "${missing[@]}"; do
            echo "  - ${v}"
        done
        return 1
    fi
    
    return 0
}

# ── 1. Prerequisites Check ────────────────────────────────────────────
echo ""
info "[1/6] Checking prerequisites..."

if ! command -v docker &>/dev/null; then
    error "Docker is not installed or not in PATH"
    error "Install Docker Desktop with WSL2 integration first"
    exit 1
fi

if ! docker info &>/dev/null; then
    error "Docker daemon is not running"
    error "Start Docker Desktop first"
    exit 1
fi

if ! command -v docker compose &>/dev/null; then
    error "Docker Compose plugin not found"
    exit 1
fi

ok "  Docker and Docker Compose are available"

# ── 2. Detect Existing Volumes ────────────────────────────────────────
echo ""
info "[2/6] Detecting existing Vision RAG volumes..."

VOLUME_PREFIX=""
if detect_volume_prefix; then
    VOLUME_PREFIX="$(detect_volume_prefix)"
    ok "  Found existing volume prefix: ${VOLUME_PREFIX}"
    info "  Existing volumes:"
    docker volume ls --format "{{.Name}}" | grep "^${VOLUME_PREFIX}_" | while read -r vol; do
        size=$(docker system df -v 2>/dev/null | grep "^${vol} " | awk '{print $4}' || echo "?")
        echo "    - ${vol} (${size})"
    done
else
    warn "  No existing Vision RAG volumes found!"
    warn "  This appears to be a fresh machine."
    echo ""
    ask "Do you want to continue anyway? (containers will be created fresh) [y/N]: "
    read -r response
    if [[ ! "${response}" =~ ^[Yy]$ ]]; then
        info "Aborted."
        exit 0
    fi
    VOLUME_PREFIX="vision_rag_git"
    info "  Will use default prefix: ${VOLUME_PREFIX}"
fi

# ── 3. Pre-flight Volume Check ────────────────────────────────────────
echo ""
info "[3/6] Pre-flight check: verifying critical volumes..."

if [ -n "${VOLUME_PREFIX}" ]; then
    if check_critical_volumes "${VOLUME_PREFIX}"; then
        ok "  All critical volumes present"
        
        # Show what's in the volumes to reassure user
        info "  Volume contents preview:"
        for vol in hf-cache qdrant_data ollama open-webui; do
            full_name="${VOLUME_PREFIX}_${vol}"
            if docker volume inspect "${full_name}" &>/dev/null; then
                # Count files in volume using a temporary container
                count=$(docker run --rm -v "${full_name}:/vol:ro" busybox sh -c "find /vol -type f 2>/dev/null | wc -l" 2>/dev/null || echo "?")
                echo "    - ${full_name}: ${count} files"
            fi
        done
    else
        error "Critical volumes missing! Aborting to prevent data loss."
        error "If this is intentional, run with --force flag."
        exit 1
    fi
fi

# ── 4. Ask for Target Directory ──────────────────────────────────────
echo ""
info "[4/6] Target project directory"

# Try to auto-detect existing project directory
EXISTING_PROJECT=""
if [ -n "${VOLUME_PREFIX}" ]; then
    # Common locations
    for check_dir in \
        "${HOME}/projects/Vision_RAG_Git" \
        "${HOME}/Vision_RAG_Git" \
        "/mnt/c/Users/*/projects/Vision_RAG_Git" \
        "/mnt/c/Users/*/Vision_RAG_Git"
    do
        for dir in ${check_dir}; do
            if [ -d "${dir}/.git" ] || [ -f "${dir}/docker-compose.yml" ]; then
                EXISTING_PROJECT="${dir}"
                break 2
            fi
        done
    done
fi

if [ -n "${EXISTING_PROJECT}" ]; then
    ask "Existing project found at: ${EXISTING_PROJECT}"
    ask "Use this directory? [Y/n]: "
    read -r response
    if [[ "${response}" =~ ^[Nn]$ ]]; then
        EXISTING_PROJECT=""
    fi
fi

if [ -z "${EXISTING_PROJECT}" ]; then
    ask "Enter the path to your Vision RAG project directory:"
    read -r EXISTING_PROJECT
fi

TARGET_DIR="${EXISTING_PROJECT}"

# Validate directory
if [ ! -d "${TARGET_DIR}" ]; then
    ask "Directory doesn't exist. Create it? [Y/n]: "
    read -r response
    if [[ ! "${response}" =~ ^[Nn]$ ]]; then
        mkdir -p "${TARGET_DIR}"
    else
        error "Aborted."
        exit 1
    fi
fi

info "  Target directory: ${TARGET_DIR}"

# ── 5. Load Docker Images ─────────────────────────────────────────────
echo ""
info "[5/6] Loading Docker images..."

if [ ! -d "${SCRIPT_DIR}/images" ]; then
    error "No images/ directory found in export"
    exit 1
fi

# Count images to load
image_count=$(find "${SCRIPT_DIR}/images" -name "*.tar.gz" | wc -l)
info "  Found ${image_count} images to load"

loaded_count=0
for img_file in "${SCRIPT_DIR}/images"/*.tar.gz; do
    if [ ! -f "${img_file}" ]; then
        continue
    fi
    
    basename=$(basename "${img_file}")
    info "  Loading ${basename}..."
    
    if docker load < "${img_file}"; then
        ok "    Loaded successfully"
        ((loaded_count++)) || true
    else
        error "    Failed to load ${basename}"
    fi
done

ok "  Loaded ${loaded_count}/${image_count} images"

# ── 6. Copy Project Code ──────────────────────────────────────────────
echo ""
info "[6/6] Updating project code..."

if [ ! -d "${SCRIPT_DIR}/code" ]; then
    error "No code/ directory found in export"
    exit 1
fi

# Backup existing .env if it exists
if [ -f "${TARGET_DIR}/.env" ]; then
    cp "${TARGET_DIR}/.env" "${TARGET_DIR}/.env.backup.$(date +%Y%m%d_%H%M%S)"
    ok "  Backed up existing .env"
fi

# Copy new code (rsync with delete to remove old files)
info "  Copying files to ${TARGET_DIR}..."
rsync -a --delete "${SCRIPT_DIR}/code/" "${TARGET_DIR}/"
ok "  Code updated"

# ── 7. Configure COMPOSE_PROJECT_NAME ─────────────────────────────────
echo ""
info "[Config] Setting up Docker Compose project name..."

# Update or add COMPOSE_PROJECT_NAME in .env
ENV_FILE="${TARGET_DIR}/.env"
if [ -f "${ENV_FILE}" ]; then
    if grep -q "^COMPOSE_PROJECT_NAME=" "${ENV_FILE}"; then
        # Update existing
        sed -i "s/^COMPOSE_PROJECT_NAME=.*/COMPOSE_PROJECT_NAME=${VOLUME_PREFIX}/" "${ENV_FILE}" 2>/dev/null || \
        sed -i.bak "s/^COMPOSE_PROJECT_NAME=.*/COMPOSE_PROJECT_NAME=${VOLUME_PREFIX}/" "${ENV_FILE}"
        ok "  Updated COMPOSE_PROJECT_NAME=${VOLUME_PREFIX} in .env"
    else
        # Add new line
        echo "" >> "${ENV_FILE}"
        echo "# Docker Compose project name — matches existing volume names" >> "${ENV_FILE}"
        echo "COMPOSE_PROJECT_NAME=${VOLUME_PREFIX}" >> "${ENV_FILE}"
        ok "  Added COMPOSE_PROJECT_NAME=${VOLUME_PREFIX} to .env"
    fi
else
    warn "  No .env file found. Creating from .env.example..."
    cp "${TARGET_DIR}/.env.example" "${ENV_FILE}"
    echo "" >> "${ENV_FILE}"
    echo "# Docker Compose project name — matches existing volume names" >> "${ENV_FILE}"
    echo "COMPOSE_PROJECT_NAME=${VOLUME_PREFIX}" >> "${ENV_FILE}"
    ok "  Created .env with COMPOSE_PROJECT_NAME=${VOLUME_PREFIX}"
    warn "  IMPORTANT: Edit ${ENV_FILE} and set your API keys!"
fi

# ── 8. Final Summary ──────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Import Complete!"
echo "========================================"
echo ""
ok "Updated:"
echo "  ${loaded_count} Docker images loaded"
echo "  Project code copied to: ${TARGET_DIR}"
echo "  COMPOSE_PROJECT_NAME set to: ${VOLUME_PREFIX}"
echo ""
info "Next Steps:"
echo ""
echo "  1. Review your environment variables:"
echo "     ${TARGET_DIR}/.env"
echo ""
echo "  2. Start the updated services:"
echo -e "     ${CYAN}cd ${TARGET_DIR}${NC}"
echo -e "     ${CYAN}docker compose up -d${NC}"
echo ""
echo "  3. Verify everything is running:"
echo -e "     ${CYAN}docker compose ps${NC}"
echo ""
echo "  4. (Optional) Install/update OpenCode integration:"
echo -e "     ${CYAN}cd ${TARGET_DIR}${NC}"
echo -e "     ${CYAN}bash opencode/setup-opencode.sh${NC}"
echo ""
echo "  5. Check that your indexed PDFs are still available:"
echo -e "     ${CYAN}curl http://localhost:8082/status${NC}"
echo "     (or use vision_rag_status in OpenCode)"
echo ""
warn "NOTE: docker compose up -d will recreate containers with new images"
warn "      but your volumes (${VOLUME_PREFIX}_*) will remain untouched."
echo ""

# ── 9. Post-import cleanup suggestion ─────────────────────────────────
ask "Would you like to see old image cleanup instructions? [y/N]: "
read -r response
if [[ "${response}" =~ ^[Yy]$ ]]; then
    echo ""
    info "After docker compose up -d, you can clean up old images:"
    echo ""
    echo "  # List all images"
    echo "  docker images"
    echo ""
    echo "  # Remove old untagged images (dangling)"
    echo "  docker image prune -f"
    echo ""
    echo "  # Or remove specific old images if you know the tags"
    echo "  docker rmi <old-image-name>:<tag>"
    echo ""
    echo "  # Remove ALL old images (USE WITH CAUTION)"
    echo "  docker image prune -a -f"
    echo ""
fi

info "Done!"
