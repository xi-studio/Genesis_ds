#!/bin/bash
# Genesis Agent — Entry Script
# Usage: bash run.sh                     # skip pip install if deps already satisfied
#        FORCE_INSTALL=1 bash run.sh     # always reinstall dependencies

set -e

echo "============================================"
echo " Genesis Agent"
echo "============================================"
echo ""

# ── 1. Dependencies (skip if already installed) ──
if [ "$FORCE_INSTALL" = "1" ]; then
    echo "[1/2] FORCE_INSTALL=1 — installing dependencies..."
    pip install -q -r requirements.txt
elif python -c "import aiohttp, openai, tokenizers" 2>/dev/null; then
    : # dependencies already satisfied
else
    echo "[1/2] Installing dependencies..."
    pip install -q -r requirements.txt
fi

# ── 2. Copy config if not exists ──
if [ ! -f "config.json" ]; then
    echo "  config.json not found, copying from config.json.example..."
    cp config.json.example config.json
    echo "  ⚠️  Edit config.json with your API key, then re-run."
    exit 1
fi

# ── 3. Start agent ──
echo "[2/2] Starting agent..."
exec python main.py
