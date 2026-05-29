"""
Host primitives — trigger and signal handling.

These are the APIs exposed to exec via the curated globals namespace.
"""
from __future__ import annotations

import asyncio
import signal
import sys
import threading
from datetime import datetime

from agent.config import Config
from agent.consciousness import append
from agent.output import say
import agent.ui_stub as _us


# --- Shared state (set by main loop at startup) ---
_loop: asyncio.AbstractEventLoop | None = None
_trigger_inbox: asyncio.Queue | None = None
_current_infer_task: asyncio.Task | None = None

def init(loop: asyncio.AbstractEventLoop, trigger_inbox: asyncio.Queue) -> None:
    """Initialize host primitives with the running event loop and trigger queue."""
    global _loop, _trigger_inbox
    _loop = loop
    _trigger_inbox = trigger_inbox

    # Share inbox with exec_engine for batch interrupt check
    from agent.exec_engine import set_trigger_inbox
    set_trigger_inbox(trigger_inbox)


def set_current_infer_task(task: asyncio.Task | None) -> None:
    """Track the current infer task so SIGINT can cancel it."""
    global _current_infer_task
    _current_infer_task = task


# --- Trigger ---
def trigger(msg: str = "") -> None:
    """Sync, immediate — safe to call from any thread."""
    say(f"  [trigger] ({len(msg)} chars)")
    if _loop and _loop.is_running():
        asyncio.run_coroutine_threadsafe(_trigger_inbox.put(msg), _loop)
    else:
        _trigger_inbox.put_nowait(msg)


# --- Signal handling ---
def _cancel_infer_if_running() -> bool:
    """Return True if an infer task was scheduled for cancellation."""
    lp = _loop
    if lp is None or not lp.is_running():
        return False
    t = _current_infer_task
    if t is not None and not t.done():
        lp.call_soon_threadsafe(t.cancel)
        return True
    return False


def _sigint_handler(signum, frame) -> None:  # noqa: ARG001
    """First Ctrl+C cancels current infer(); otherwise propagate KeyboardInterrupt."""
    lp = _loop
    if lp is None or not lp.is_running():
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.raise_signal(signal.SIGINT)
        return
    if _cancel_infer_if_running():
        return

    def _raise_kb() -> None:
        raise KeyboardInterrupt

    lp.call_soon_threadsafe(_raise_kb)


def install_signal_handler() -> None:
    """Install SIGINT handler for graceful infer cancellation."""
    try:
        signal.signal(signal.SIGINT, _sigint_handler)
    except (ValueError, OSError):
        pass


# --- Drain triggers ---
def drain_triggers_to_consciousness() -> tuple[int, bool]:
    """
    Drain the trigger queue into consciousness before infer.

    Returns (message_count, had_human): ``had_human`` is True if any drained
    line contains ``[Human]``.
    """
    msgs: list[str] = []
    while True:
        try:
            msgs.append(_trigger_inbox.get_nowait())
        except asyncio.QueueEmpty:
            break
    if not msgs:
        return (0, False)
    merged = "\n".join(msgs)
    had_human = "[Human]" in merged
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    append(f"System - [Trigger] [{ts}] {merged}\n\n")
    say(
        f"  [trigger → consciousness] {len(msgs)} message(s) before infer, {len(merged)} chars"
    )
    return (len(msgs), had_human)


# --- Curated exec globals ---
def exec_globals() -> dict:
    """Build the curated globals dict exposed to /exec.

    Only safe APIs are exposed — not the full module namespace.
    """
    return {
        # Host primitives
        "trigger": trigger,
        # Standard libs (commonly needed)
        "__builtins__": __builtins__,
    }
