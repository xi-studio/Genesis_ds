"""
Output and logging — serialized prints with optional log file.

All terminal output goes through ``say()`` which is thread-safe and
optionally writes to a log file.
"""
from __future__ import annotations

import os
import sys
import threading

from agent.config import Config

_lock = threading.Lock()
_OUT = sys.stderr


def say(*args, file=None, **kwargs) -> None:
    """Thread-safe print: terminal (unless quiet) + log file."""
    cfg = Config.get()
    with _lock:
        if not cfg.terminal_quiet:
            tgt = file if file is not None else _OUT
            print(*args, file=tgt, **kwargs)
        lf = cfg.log_file
        if lf:
            parent = os.path.dirname(os.path.abspath(lf))
            if parent:
                try:
                    os.makedirs(parent, exist_ok=True)
                except OSError:
                    pass
            try:
                with open(lf, "a", encoding="utf-8", errors="replace") as logf:
                    print(*args, file=logf, **kwargs)
            except OSError:
                pass


def output_stream():
    """Return the current output stream (default stderr)."""
    return _OUT
