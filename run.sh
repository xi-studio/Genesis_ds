#!/bin/bash
# Genesis Agent — Entry Script
# Usage: bash run.sh

set -e

echo "============================================"
echo " Genesis Agent"
echo "============================================"
echo ""

# ── 1. Install deps ──
echo "[1/2] Installing dependencies..."
pip install -q -r requirements.txt

# ── 2. Copy config if not exists ──
if [ ! -f "config.json" ]; then
    echo "  config.json not found, copying from config.json.example..."
    cp config.json.example config.json
    echo "  ⚠️  Edit config.json with your API key, then re-run."
    exit 1
fi

# ── 3. Start agent ──
echo "[2/2] Starting agent..."
python main.py
