"""
Host Python execution for the ``exec`` function tool.

``run_exec_source_once`` compiles and runs code in the shared globals dict;
results return as tool message content (no duplicate consciousness mirroring).
"""
from __future__ import annotations

import asyncio
import ast
import inspect
import io
import os
from contextlib import redirect_stdout

from agent.config import Config
from agent.output import say
from agent.timestamp import now_local
import agent.ui_stub as _us


def _exec_stdout_max_chars() -> int:
    env = os.environ.get("EXEC_STDOUT_MAX_CHARS")
    if env and env.strip().isdigit():
        return max(1024, int(env.strip()))
    return max(1024, int(Config.get().exec_stdout_max_chars))


async def run_exec_source_once(source: str, exec_globals: dict) -> str:
    """Run one Python source string (compile + eval; top-level await). Return stdout/errors for the tool message."""
    await asyncio.sleep(0)

    cfg = Config.get()
    if (
        cfg.exec_batch_interrupt_on_human
        and _trigger_inbox_ref is not None
        and _trigger_inbox_ref.qsize() > 0
    ):
        from agent.host_primitives import drain_triggers_to_consciousness

        drain_triggers_to_consciousness()
        note = (
            "System - [ExecBatchInterrupted] "
            f"[{now_local()}] "
            "Human message(s) arrived — this exec was skipped "
            "(re-run in the next round if needed).\n\n"
        )
        say("  [exec batch] interrupted — human input merged; exec skipped")
        await _us.emit_ui_event({"event": "exec_batch_stopped", "reason": "human_input"})
        return note.strip()

    src = (source or "").strip()
    if not src:
        return "(empty code)"
    if len(src) > cfg.max_exec_source_chars:
        err = (
            f"System - [ExecError] [{now_local()}] "
            f"/exec block too large ({len(src)} chars, "
            f"max {cfg.max_exec_source_chars}). Use the read_file tool or open() instead of pasting."
        )
        say(f"  {err}")
        return err

    say(f"  [exec python] {src[:80]}...")
    await _us.emit_ui_event({"event": "exec", "preview": src[:80]})
    buf = io.StringIO()
    try:
        code = compile(
            src, "/exec_python", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
        )
        with redirect_stdout(buf):
            result = eval(code, exec_globals)
            if inspect.isawaitable(result):
                await result
    except Exception as e:
        detail = str(e)
        hint = ""
        if "SIGNAL:" in detail or "Source was saved" in detail:
            hint = (
                " [Hint: IDE/host often raises this when a giant paste looks like a save; "
                "use open() instead of inlining the file.]"
            )
        err = (
            f"System - [ExecError] [{now_local()}] "
            f"{type(e).__name__}: {detail}{hint}"
        )
        say(f"  {err}")
        return err
    else:
        raw_out = buf.getvalue()
        if raw_out:
            say(raw_out, end="", flush=True)
            if not raw_out.endswith("\n"):
                say("", flush=True)
        if raw_out.strip():
            lim = _exec_stdout_max_chars()
            body = (
                raw_out
                if len(raw_out) <= lim
                else raw_out[:lim] + f"\n\n[System - truncated exec stdout at {lim} chars]\n"
            )
            await _us.emit_ui_event({"event": "exec_stdout", "text": body})
            return body
        return "(no output)"


# --- Trigger inbox reference (set by host_primitives at init) ---
_trigger_inbox_ref = None


def set_trigger_inbox(inbox) -> None:
    """Called by host_primitives to share the trigger inbox reference."""
    global _trigger_inbox_ref
    _trigger_inbox_ref = inbox
