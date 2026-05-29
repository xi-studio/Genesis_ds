"""
UI event stub — overridden by main.py when UI is active.

This module provides the ``emit_ui_event`` function and shared state
for UI broadcast (TCP NDJSON + WebSocket). The server layer monkey-patches
``emit_ui_event`` with a real implementation.
"""
from __future__ import annotations

import asyncio

# Shared state — server layer reads/writes these
ui_broadcast_lock: asyncio.Lock | None = None
ui_clients: set[asyncio.StreamWriter] = set()
web_ws_clients: set[object] = set()


async def emit_ui_event(ev: dict) -> None:
    """Stub — overridden by main.py when UI is active."""
    pass
