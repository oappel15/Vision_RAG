#!/bin/bash
# Vision RAG Migration Wizard — Web UI Launcher
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo " Vision RAG Migration Wizard v4.0 (Web UI)"
echo "============================================================"
echo ""

# Install Flask if needed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "Installing Flask..."
    pip install --break-system-packages flask 2>/dev/null || pip install flask 2>/dev/null
fi

# Try to auto-open browser
sleep 2 && (xdg-open http://localhost:5555 2>/dev/null || sensible-browser http://localhost:5555 2>/dev/null || echo "Open your browser: http://localhost:5555") &

cd "$SCRIPT_DIR"
exec python3 vision_rag_web.py
