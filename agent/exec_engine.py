"""
Host Python execution for the ``exec`` function tool.

``run_exec_source_once`` compiles and runs code in the shared globals dict;
results return as tool message content (no duplicate consciousness mirroring).

**Two execution paths:**
* Async code (module-level ``await`` / ``async for`` / ``async with``) runs as an
  in-process loop task — cancellation and timeout cancel the task.
* Sync code runs in a **separate worker process** (``sys.executable -c helper``).
  On timeout the worker is SIGKILLed, so a ``while True: pass`` can never leave a
  zombie thread spinning forever. The shared globals dict is round-tripped
  (pickled) into the worker and merged back on success, so cross-call state keeps
  working; ``trigger()`` inside the worker is forwarded to the host after the run.

Known limitation (async path): a CPU-bound infinite loop *inside* a coroutine
(e.g. ``async def f(): while True: pass``) never yields to the event loop, so it
cannot be cancelled by timeout — avoid writing one; sync infinite loops are safe.
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import io
import os
import pickle
import shutil
import sys
import tempfile
from contextlib import redirect_stdout

from agent.config import Config
from agent.output import say
from agent.timestamp import now_local
from agent.host_primitives import trigger
import agent.ui_stub as _us


def _exec_stdout_max_chars() -> int:
    env = os.environ.get("EXEC_STDOUT_MAX_CHARS")
    if env and env.strip().isdigit():
        return max(1024, int(env.strip()))
    return max(1024, int(Config.get().exec_stdout_max_chars))


def _has_top_level_await(source: str) -> bool:
    """True if ``source`` awaits at module level (not inside a function/class)."""
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError:
        return False
    stack: list[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Await, ast.AsyncFor, ast.AsyncWith)):
                return True
            # Definitions don't run at module level — skip their bodies.
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            stack.append(child)
    return False


def _exec_error_message(e: Exception) -> str:
    detail = str(e)
    hint = ""
    if "SIGNAL:" in detail or "Source was saved" in detail:
        hint = (
            " [Hint: IDE/host often raises this when a giant paste looks like a save; "
            "use open() instead of inlining the file.]"
        )
    return (
        f"System - [ExecError] [{now_local()}] "
        f"{type(e).__name__}: {detail}{hint}"
    )


def _exec_timeout_message(timeout: float, kind: str) -> str:
    """Timeout error. ``kind``: "process" (sync worker, force-killed) or "task" (async, cancelled)."""
    extra = (
        "The worker process was killed (SIGKILL)."
        if kind == "process"
        else "The background task was cancelled."
    )
    return (
        f"System - [ExecError] [{now_local()}] "
        f"exec timed out after {timeout:g}s (exec_timeout_sec). {extra}"
    )


# ---------------------------------------------------------------------------
# Sync path: isolated worker process (killable on timeout)
# ---------------------------------------------------------------------------

# Runs inside the worker. Receives argv: src_path globals_path result_path trigger_path.
# Pickles: globals in, result (ok/error/globals) out; trigger() calls appended
# to trigger_path as pickled messages (size-capped so a runaway loop can't grow
# the file without bound — the worker is SIGKILLed on timeout anyway).
_CHILD_HELPER = r"""
import os, pickle, sys

_TRIGGER_PATH = ""
_TRIGGER_CAP = 1_000_000

def _trigger_stub(msg=""):
    try:
        sz = os.path.getsize(_TRIGGER_PATH) if os.path.exists(_TRIGGER_PATH) else 0
        if sz > _TRIGGER_CAP:
            raise RuntimeError("trigger queue full (>1MB)")
        with open(_TRIGGER_PATH, "ab") as f:
            pickle.dump(msg or "", f)
    except Exception as e:
        print(f"[exec child] trigger failed: {e}", file=sys.stderr)

def main():
    global _TRIGGER_PATH
    src_path, globals_path, result_path, trigger_path = sys.argv[1:5]
    _TRIGGER_PATH = trigger_path
    g = {}
    try:
        with open(globals_path, "rb") as f:
            g = pickle.load(f)
    except Exception:
        g = {}
    if not isinstance(g, dict):
        g = {}
    g.setdefault("__builtins__", __builtins__)
    g.setdefault("__name__", "__exec__")
    g["trigger"] = _trigger_stub  # never leak the host trigger into the child

    result = {"ok": False, "error": "worker died without result", "globals": None}
    try:
        with open(src_path, "r", encoding="utf-8") as f:
            src = f.read()
        code = compile(src, "/exec_python", "exec")
        exec(code, g)
        result["ok"] = True
        result["error"] = None
    except BaseException as e:  # noqa: BLE001 — report any error to the host
        result["ok"] = False
        result["error"] = f"{type(e).__name__}: {e}"
    try:
        # Exclude host-side primitives (unpicklable in this context): the local
        # trigger stub is a __main__ function, __builtins__ is a module object.
        # The host restores its own primitives when merging back.
        g_back = {k: v for k, v in g.items() if k not in ("trigger", "__builtins__")}
        result["globals"] = pickle.dumps(g_back)
    except Exception:
        result["globals"] = None
    with open(result_path, "wb") as f:
        pickle.dump(result, f)

if __name__ == "__main__":
    main()
"""


async def _run_sync_worker(source: str, exec_globals: dict, timeout: float) -> tuple:
    """Run sync code in a killable worker process.

    Returns (stdout_text, result_dict|None, triggers, dropped_keys, timed_out).
    ``result_dict`` is None when the worker exited without writing a result
    (e.g. ``os._exit``) — stdout is still returned.
    """
    tmp = tempfile.mkdtemp(prefix="genesis_exec_")
    try:
        src_path = os.path.join(tmp, "src.py")
        gl_path = os.path.join(tmp, "globals.pkl")
        res_path = os.path.join(tmp, "result.pkl")
        tr_path = os.path.join(tmp, "triggers.pkl")

        with open(src_path, "w", encoding="utf-8") as f:
            f.write(source)

        # Globals in: best-effort picklable snapshot. Host primitives are never
        # shipped — trigger is installed as a stub by the child, __builtins__ is
        # always re-derived there.
        send: dict = {}
        dropped: list[str] = []
        for k, v in exec_globals.items():
            if k in ("trigger", "__builtins__"):
                continue
            try:
                pickle.dumps(v)
                send[k] = v
            except Exception:
                dropped.append(k)
        with open(gl_path, "wb") as f:
            pickle.dump(send, f)

        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["GENESIS_EXEC_CHILD"] = "1"

        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-u", "-c", _CHILD_HELPER,
            src_path, gl_path, res_path, tr_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,  # child stderr passes through to the host console
            env=env,
        )
        timed_out = False
        try:
            out_b, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            timed_out = True
            proc.kill()
            out_b, _ = await proc.communicate()
        except asyncio.CancelledError:
            proc.kill()
            raise

        out = (out_b or b"").decode("utf-8", errors="replace")

        triggers: list[str] = []
        if os.path.exists(tr_path):
            try:
                with open(tr_path, "rb") as f:
                    while True:
                        try:
                            triggers.append(pickle.load(f))
                        except EOFError:
                            break
            except Exception:
                pass

        result = None
        if os.path.exists(res_path):
            try:
                with open(res_path, "rb") as f:
                    result = pickle.load(f)
            except Exception:
                result = None
        return out, result, triggers, dropped, timed_out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def run_exec_source_once(source: str, exec_globals: dict) -> str:
    """Run one Python source string (compile + eval; top-level await). Return stdout/errors for the tool message.

    Timeout: bounded by ``exec_timeout_sec`` (default 120s). Sync code runs in a
    worker process that is force-killed on timeout; async code runs as a loop
    task. Either way the event loop stays responsive, so Stop / triggers / new
    messages keep working while exec runs.
    """
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

    timeout = max(1.0, float(getattr(cfg, "exec_timeout_sec", 120.0) or 120.0))
    say(f"  [exec python] {src[:80]}...")
    await _us.emit_ui_event({"event": "exec", "preview": src[:80]})
    buf = io.StringIO()

    try:
        code = compile(
            src, "/exec_python", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
        )
    except SyntaxError as e:
        err = _exec_error_message(e)
        say(f"  {err}")
        return err

    try:
        if _has_top_level_await(src):
            # Async path: run as a loop task; timeout/Stop cancels the task.
            coro = eval(code, exec_globals)
            if inspect.isawaitable(coro):
                task = asyncio.create_task(coro)
                with redirect_stdout(buf):
                    await asyncio.wait_for(task, timeout=timeout)
        else:
            # Sync path: isolated worker process, force-killed on timeout.
            out, result, triggers, dropped, timed_out = await _run_sync_worker(
                src, exec_globals, timeout
            )
            for t_msg in triggers:
                try:
                    trigger(t_msg)
                except Exception:
                    pass
            if timed_out:
                err = _exec_timeout_message(timeout, "process")
                say(f"  {err}")
                return err

            note = ""
            if dropped:
                shown = ", ".join(str(k) for k in dropped[:8])
                more = f" (+{len(dropped) - 8} more)" if len(dropped) > 8 else ""
                note = (
                    f"\n\n[System - exec: {len(dropped)} non-picklable global(s) "
                    f"not passed to the worker: {shown}{more}]"
                )

            if result is None:
                # Worker exited without writing a result (e.g. os._exit) — stdout stands.
                body = out if out.strip() else "(no output)"
            elif result.get("ok") is False:
                err = f"System - [ExecError] [{now_local()}] {result.get('error') or 'unknown error'}"
                say(f"  {err}")
                return err + note
            else:
                # Merge worker globals back (never host primitives).
                gb = result.get("globals")
                if isinstance(gb, (bytes, bytearray)):
                    try:
                        merged = pickle.loads(gb)
                        if isinstance(merged, dict):
                            merged.pop("trigger", None)
                            merged.pop("__builtins__", None)
                            exec_globals.clear()
                            exec_globals.update(merged)
                            # Restore host primitives that were never shipped to the
                            # worker. Without this, exec_globals loses `trigger` and
                            # `__builtins__` after every sync exec, causing NameError
                            # in subsequent async exec calls that use trigger().
                            exec_globals["trigger"] = trigger
                            exec_globals["__builtins__"] = __builtins__
                    except Exception:
                        pass
                body = out if out.strip() else "(no output)"
            if body != "(no output)" and note:
                body = body + note
            if body != "(no output)":
                say(body, end="", flush=True)
                if not body.endswith("\n"):
                    say("", flush=True)
            lim = _exec_stdout_max_chars()
            if len(body) > lim:
                body = body[:lim] + f"\n\n[System - truncated exec stdout at {lim} chars]\n"
            await _us.emit_ui_event({"event": "exec_stdout", "text": body})
            return body
    except asyncio.TimeoutError:
        err = _exec_timeout_message(timeout, "task")
        say(f"  {err}")
        return err
    except asyncio.CancelledError:
        raise  # Stop/Ctrl+C — let the infer loop handle it
    except Exception as e:
        err = _exec_error_message(e)
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
