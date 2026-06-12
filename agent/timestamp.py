"""
Unified timestamp utilities — single source of truth for all agent timestamps.

Produces ISO 8601 with explicit timezone offset: ``2026-06-12T11:25:50+08:00``
"""
from __future__ import annotations

from datetime import datetime, timezone


def now_local() -> str:
    """Return current local time as ISO 8601 string with timezone offset.

    Example: ``2026-06-12T11:25:50+08:00``
    """
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
