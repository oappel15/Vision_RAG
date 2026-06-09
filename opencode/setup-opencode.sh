#!/bin/bash
#
# Vision RAG — OpenCode Integration Setup
#
# Installs the Vision RAG MCP server and ALL skills into the global
# OpenCode config directory (~/.config/opencode/). Run once on each
# machine where you want OpenCode to have access to Vision RAG.
#
# Installs:
#   - Vision RAG MCP server (6 tools for PDF indexing + search)
#   - Vision RAG skill (auto-triggers on PDF/schematic keywords)
#   - KiCad netlist-to-schematic skill
#
# Usage:
#   cd Vision_RAG_Git
#   bash opencode/setup-opencode.sh [--host <hostname>] [--dry-run]
#
# Options:
#   --host <hostname>   MCP server target host (default: localhost)
#   --dry-run           Preview changes without applying
#
# After running, restart OpenCode for changes to take effect.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENCODE_DIR="${HOME}/.config/opencode"

# Parse arguments
MCP_HOST="localhost"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            MCP_HOST="${2:-localhost}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--host <hostname>] [--dry-run]"
            exit 1
            ;;
    esac
done

echo "=== Vision RAG — OpenCode Integration Setup ==="
echo ""
echo "  Target host: ${MCP_HOST}"
if [ "${DRY_RUN}" = true ]; then
    echo "  Mode: DRY RUN (no changes will be made)"
fi
echo ""
echo "  Target host: ${MCP_HOST}"
if [ "${DRY_RUN}" = true ]; then
    echo "  Mode: DRY RUN (no changes will be made)"
fi
echo ""

# 1. Install Python MCP SDK (required by the MCP server)
echo "[1/6] Installing Python MCP SDK..."
if python3 -c "from mcp.server.fastmcp import FastMCP" 2>/dev/null; then
    echo "  Already installed."
else
    if [ "${DRY_RUN}" = true ]; then
        echo "  Would install: pip install mcp"
    else
        pip install --break-system-packages mcp 2>/dev/null \
            || pip install mcp 2>/dev/null \
            || { echo "ERROR: Failed to install mcp package. Install manually: pip install mcp"; exit 1; }
        echo "  Done."
    fi
fi

# 2. Check if Vision RAG services are reachable
echo ""
echo "[2/6] Checking Vision RAG services..."
INGEST_URL="http://${MCP_HOST}:8082"
PIPELINES_URL="http://${MCP_HOST}:9099"

services_ok=true
if command -v curl &>/dev/null; then
    if curl -s "${INGEST_URL}/status" &>/dev/null; then
        echo "  pdf-ingest (port 8082): REACHABLE"
    else
        echo "  pdf-ingest (port 8082): NOT REACHABLE (service may not be running)"
        services_ok=false
    fi
    
    if curl -s "${PIPELINES_URL}/v1/models" &>/dev/null || curl -s "${PIPELINES_URL}" &>/dev/null; then
        echo "  pipelines (port 9099): REACHABLE"
    else
        echo "  pipelines (port 9099): NOT REACHABLE (service may not be running)"
        services_ok=false
    fi
else
    echo "  curl not available — skipping connectivity check"
fi

if [ "${services_ok}" = false ]; then
    echo ""
    echo "WARNING: Some Vision RAG services are not reachable."
    echo "Make sure Docker containers are running:"
    echo "  cd $(dirname "${SCRIPT_DIR}")"
    echo "  docker compose up -d"
    echo ""
    echo "Continue anyway? [y/N]: "
    read -r response
    if [[ ! "${response}" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# 3. Copy MCP server
echo ""
echo "[3/6] Installing MCP server..."
if [ "${DRY_RUN}" = true ]; then
    echo "  Would copy: ${SCRIPT_DIR}/vision-rag-mcp.py → ${OPENCODE_DIR}/vision-rag-mcp.py"
else
    mkdir -p "${OPENCODE_DIR}"
    cp "${SCRIPT_DIR}/vision-rag-mcp.py" "${OPENCODE_DIR}/vision-rag-mcp.py"
    echo "  Installed: ${OPENCODE_DIR}/vision-rag-mcp.py"
fi

# 4. Copy skills
echo ""
echo "[4/6] Installing skills..."
for skill_dir in "${SCRIPT_DIR}"/skills/*/; do
    skill_name="$(basename "${skill_dir}")"
    dest="${OPENCODE_DIR}/skills/${skill_name}"
    if [ "${DRY_RUN}" = true ]; then
        echo "  Would copy skill: ${skill_name} → ${dest}/"
    else
        mkdir -p "${dest}"
        cp "${skill_dir}SKILL.md" "${dest}/SKILL.md"
        echo "  Installed skill: ${skill_name}"
    fi
done

# 5. Merge MCP config into opencode.json
echo ""
echo "[5/6] Registering MCP server in opencode.json..."
OPENCODE_JSON="${OPENCODE_DIR}/opencode.json"

# Detect the MCP server path (use $HOME so it's portable)
MCP_PATH="${OPENCODE_DIR}/vision-rag-mcp.py"

# Build the MCP config entry with the correct host
MCP_CONFIG=$(python3 -c "
import json
cfg = {
    'type': 'local',
    'command': ['python3', '${MCP_PATH}'],
    'enabled': True,
    'env': {
        'VISION_RAG_INGEST_URL': 'http://${MCP_HOST}:8082',
        'VISION_RAG_PIPELINES_URL': 'http://${MCP_HOST}:9099',
        'VISION_RAG_PIPELINES_API_KEY': '0p3n-w3bu!',
        'VISION_RAG_PIPELINE_MODEL': 'colpali-pipeline',
    }
}
print(json.dumps(cfg, indent=2))
")

if [ "${DRY_RUN}" = true ]; then
    echo "  Would register vision-rag MCP server with host: ${MCP_HOST}"
    echo "  Config preview:"
    echo "${MCP_CONFIG}" | sed 's/^/    /'
else
    if [ -f "${OPENCODE_JSON}" ]; then
        # Check if vision-rag is already registered
        if python3 -c "
import json
with open('${OPENCODE_JSON}') as f:
    cfg = json.load(f)
if 'vision-rag' in cfg.get('mcp', {}):
    exit(0)  # already registered
exit(1)
" 2>/dev/null; then
            echo "  Already registered in opencode.json."
            echo "  Updating host to: ${MCP_HOST}"
            python3 -c "
import json
with open('${OPENCODE_JSON}') as f:
    cfg = json.load(f)
cfg.setdefault('mcp', {})
cfg['mcp']['vision-rag'] = ${MCP_CONFIG}
with open('${OPENCODE_JSON}', 'w') as f:
    json.dump(cfg, f, indent=2)
print('  Updated vision-rag MCP server config.')
"
        else
            # Add the MCP server config
            python3 -c "
import json
with open('${OPENCODE_JSON}') as f:
    cfg = json.load(f)
cfg.setdefault('mcp', {})
cfg['mcp']['vision-rag'] = ${MCP_CONFIG}
with open('${OPENCODE_JSON}', 'w') as f:
    json.dump(cfg, f, indent=2)
print('  Added vision-rag MCP server to opencode.json.')
"
        fi
    else
        # Create a minimal opencode.json with just the MCP server
        python3 -c "
import json
cfg = {
    '\$schema': 'https://opencode.ai/config.json',
    'mcp': {
        'vision-rag': ${MCP_CONFIG}
    }
}
with open('${OPENCODE_JSON}', 'w') as f:
    json.dump(cfg, f, indent=2)
print('  Created opencode.json with vision-rag MCP server.')
"
    fi
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "[6/6] Summary:"
echo "  MCP server:  ${OPENCODE_DIR}/vision-rag-mcp.py"
echo "  Target host: ${MCP_HOST}"
echo "  Skills:"
for skill_dir in "${SCRIPT_DIR}"/skills/*/; do
    echo "    - $(basename "${skill_dir}")"
done
echo "  Config:      ${OPENCODE_DIR}/opencode.json"
echo ""

if [ "${DRY_RUN}" = true ]; then
    echo "DRY RUN complete — no changes were made."
    echo "Run without --dry-run to apply changes."
else
    echo "Restart OpenCode for changes to take effect."
fi

echo ""
echo "Make sure Vision RAG Docker services are running:"
echo "  cd $(dirname "${SCRIPT_DIR}")"
echo "  docker compose up -d"
echo ""
echo "To verify, after restarting OpenCode, ask:"
echo '  "What documents are in my Vision RAG index?"'
