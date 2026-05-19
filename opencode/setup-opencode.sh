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
#   bash opencode/setup-opencode.sh
#
# After running, restart OpenCode for changes to take effect.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENCODE_DIR="${HOME}/.config/opencode"

echo "=== Vision RAG — OpenCode Integration Setup ==="
echo ""

# 1. Install Python MCP SDK (required by the MCP server)
echo "[1/5] Installing Python MCP SDK..."
if python3 -c "from mcp.server.fastmcp import FastMCP" 2>/dev/null; then
    echo "  Already installed."
else
    pip install --break-system-packages mcp 2>/dev/null \
        || pip install mcp 2>/dev/null \
        || { echo "ERROR: Failed to install mcp package. Install manually: pip install mcp"; exit 1; }
    echo "  Done."
fi

# 2. Copy MCP server
echo "[2/5] Installing MCP server..."
mkdir -p "${OPENCODE_DIR}"
cp "${SCRIPT_DIR}/vision-rag-mcp.py" "${OPENCODE_DIR}/vision-rag-mcp.py"
echo "  Installed: ${OPENCODE_DIR}/vision-rag-mcp.py"

# 3. Copy skills
echo "[3/5] Installing skills..."
for skill_dir in "${SCRIPT_DIR}"/skills/*/; do
    skill_name="$(basename "${skill_dir}")"
    dest="${OPENCODE_DIR}/skills/${skill_name}"
    mkdir -p "${dest}"
    cp "${skill_dir}SKILL.md" "${dest}/SKILL.md"
    echo "  Installed skill: ${skill_name}"
done

# 4. Merge MCP config into opencode.json
echo "[4/5] Registering MCP server in opencode.json..."
OPENCODE_JSON="${OPENCODE_DIR}/opencode.json"

# Detect the MCP server path (use $HOME so it's portable)
MCP_PATH="${OPENCODE_DIR}/vision-rag-mcp.py"

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
    else
        # Add the MCP server config
        python3 -c "
import json
with open('${OPENCODE_JSON}') as f:
    cfg = json.load(f)
cfg.setdefault('mcp', {})
cfg['mcp']['vision-rag'] = {
    'type': 'local',
    'command': ['python3', '${MCP_PATH}'],
    'enabled': True,
    'env': {
        'VISION_RAG_INGEST_URL': 'http://localhost:8082',
        'VISION_RAG_PIPELINES_URL': 'http://localhost:9099',
        'VISION_RAG_PIPELINES_API_KEY': '0p3n-w3bu!',
        'VISION_RAG_PIPELINE_MODEL': 'colpali-pipeline',
    }
}
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
        'vision-rag': {
            'type': 'local',
            'command': ['python3', '${MCP_PATH}'],
            'enabled': True,
            'env': {
                'VISION_RAG_INGEST_URL': 'http://localhost:8082',
                'VISION_RAG_PIPELINES_URL': 'http://localhost:9099',
                'VISION_RAG_PIPELINES_API_KEY': '0p3n-w3bu!',
                'VISION_RAG_PIPELINE_MODEL': 'colpali-pipeline',
            }
        }
    }
}
with open('${OPENCODE_JSON}', 'w') as f:
    json.dump(cfg, f, indent=2)
print('  Created opencode.json with vision-rag MCP server.')
"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "[5/5] Summary:"
echo "  MCP server:  ${OPENCODE_DIR}/vision-rag-mcp.py"
echo "  Skills:"
for skill_dir in "${SCRIPT_DIR}"/skills/*/; do
    echo "    - $(basename "${skill_dir}")"
done
echo "  Config:      ${OPENCODE_DIR}/opencode.json"
echo ""
echo "Restart OpenCode for changes to take effect."
echo ""
echo "Make sure Vision RAG Docker services are running:"
echo "  cd $(dirname "${SCRIPT_DIR}")"
echo "  docker compose up -d"
echo ""
echo "To verify, after restarting OpenCode, ask:"
echo '  "What documents are in my Vision RAG index?"'
