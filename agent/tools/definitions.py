"""
Built-in tool schemas (OpenAI-compatible) and handler registration.

Registers handlers with ``agent.tools.dispatch``.
"""
from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

from .dispatch import register_tool
from .grep_tool import run_grep

# ── Schema builder ──────────────────────────────────────────────────────

def _p(type: str, desc: str, *, required: bool = True, **extra) -> dict:
    """Build a parameter definition dict. Pass ``required=False`` for optional params."""
    d = {"type": type, "description": desc, **extra}
    d["_required"] = required
    return d

def _tool(name: str, desc: str, **params) -> dict:
    """Build a single OpenAI function-calling tool definition."""
    required = [k for k, v in params.items() if v.pop("_required", True)]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": params, "required": required},
        },
    }


# ── Tool definitions ────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    _tool("exec",
        "Execute Python on the host in the shared exec namespace (``trigger``, etc.). "
        "Top-level await is supported. For files use the ``read_file`` / ``write_file`` / "
        "``edit_file`` / ``grep`` tools or ``open()`` in code; prefer ``shell`` when it fits.",
        code=_p("string", "Python code to execute"),
    ),
    _tool("read_file",
        "Read a text file. Supports 1-based line offset/limit and optional line numbers.",
        path=_p("string", "File path to read"),
        offset=_p("integer", "Start line number, 1-based (default 1)", required=False),
        limit=_p("integer", "Max lines to return (default all remaining lines)", required=False),
        line_numbers=_p("boolean", "If true, prefix each returned line with its line number", required=False),
    ),
    _tool("write_file",
        "Write text to a file, creating parent directories if needed. Supports overwrite/append/prepend/create modes.",
        path=_p("string", "File path to write"),
        content=_p("string", "Content to write"),
        mode=_p("string", "Write mode: overwrite (default), append, prepend, or create", enum=["overwrite", "append", "prepend", "create"], required=False),
        show_diff=_p("boolean", "If true, include a truncated unified diff for text changes", required=False),
    ),
    _tool("edit_file",
        "Replace text in a file. Supports unique match, nth occurrence, or replace-all with optional count guard.",
        path=_p("string", "File path to edit"),
        old_text=_p("string", "Exact text to replace"),
        new_text=_p("string", "Replacement text"),
        replace_all=_p("boolean", "Replace all occurrences (default false)", required=False),
        occurrence=_p("integer", "When replace_all=false and old_text appears multiple times, replace this 1-based occurrence", required=False),
        expected_count=_p("integer", "Optional safety guard: require old_text to appear exactly this many times", required=False),
        context_lines=_p("integer", "Unified diff context lines (default 3, range 0-20)", required=False),
    ),
    _tool("shell",
        "Run a shell command; returns stdout+stderr. Long jobs: nohup + log + &; "
        "read the log later. Default timeout 30s.",
        command=_p("string", "Shell command to run"),
        timeout=_p("integer", "Timeout in seconds (default 30)", required=False),
        cwd=_p("string", "Working directory (default current)", required=False),
    ),
    _tool("grep",
        "Search file contents with a regex (or plain text if fixed_strings=true). "
        "Default output_mode is files_with_matches (paths only). Use output_mode=content "
        "for matching lines with optional context. Skips binary and files >2MB; ignores "
        ".git, node_modules, __pycache__, .venv. Paths are cwd-relative like read_file.",
        pattern=_p("string", "Regex pattern, or literal if fixed_strings=true"),
        path=_p("string", "File or directory to search (default '.')", required=False),
        glob=_p("string", "Optional path filter, e.g. '*.py' or 'tests/**/test_*.py'", required=False),
        type=_p("string", "Optional type shorthand: py, ts, md, json, yaml, ...", required=False),
        case_insensitive=_p("boolean", "", required=False),
        fixed_strings=_p("boolean", "If true, pattern is plain text (not regex)", required=False),
        output_mode=_p("string", "files_with_matches: list paths (default); content: lines + context; count: match counts per file",
            enum=["content", "files_with_matches", "count"], required=False),
        context_before=_p("integer", "Lines of context before each match in content mode (0-20)", required=False),
        context_after=_p("integer", "Lines of context after each match in content mode (0-20)", required=False),
        head_limit=_p("integer", "Max results per mode (default 250); 0 = no limit", required=False),
        max_matches=_p("integer", "Alias for head_limit in content mode", required=False),
        max_results=_p("integer", "Alias for head_limit in files_with_matches / count mode", required=False),
        offset=_p("integer", "Skip the first N hits before applying head_limit", required=False),
    ),
    _tool("core_memory_append",
        "Append one concise note to core memory (SQLite source table). "
        "priority: P1 = permanent (no passive expiry), P2 = kept 7 days after updated_at, "
        "P3 = kept 24 hours (passive purge when the injected snapshot syncs). Default P3.",
        content=_p("string", "Markdown-ready note body; keep short and high-signal"),
        priority=_p("string", "Retention tier (default P3)", enum=["P1", "P2", "P3"], required=False),
    ),
    _tool("core_memory_update",
        "Replace content of one core memory entry by id. Optional priority P1/P2/P3. "
        "To retire a note, set **P3** (and shorten content if needed); "
        "passive TTL removes it when the injected snapshot syncs — there is no delete tool.",
        id=_p("string", "Entry id from the injected Core Memory list"),
        content=_p("string", "New markdown body"),
        priority=_p("string", "If set, new retention tier", enum=["P1", "P2", "P3"], required=False),
    ),
]


# ── Tool Handlers ─────────────────────────────────────────────────────────

_exec_globals: dict[str, Any] = {"__builtins__": __builtins__}

def set_exec_globals(g: dict[str, Any]) -> None:
    _exec_globals.update(g)

async def _handle_exec(args: dict[str, Any]) -> str:
    from agent.exec_engine import run_exec_source_once
    return await run_exec_source_once(args.get("code", ""), _exec_globals)


# ── File-operation helpers ──────────────────────────────────────────────

_MAX_OUTPUT = 32000
_MAX_DIFF_CHARS = 12000


def _truncate(s: str, max_chars: int = _MAX_OUTPUT) -> str:
    return s if len(s) <= max_chars else s[:max_chars] + f"\n... (truncated, {len(s) - max_chars} chars omitted)"


def _to_int(value: Any, default: int, *, lo: int | None = None, hi: int | None = None) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = default
    if lo is not None:
        n = max(lo, n)
    if hi is not None:
        n = min(hi, n)
    return n


def _read_text(path: str) -> tuple[bool, str]:
    """Read a UTF-8-ish text file. Return (ok, content_or_error)."""
    if not str(path or "").strip():
        return False, "Error: path is required"
    p = Path(path).expanduser()
    if not p.exists():
        return False, f"Error: File not found: {path}"
    if not p.is_file():
        return False, f"Error: Not a file: {path}"
    try:
        return True, p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return False, f"Error: {type(e).__name__}: {e}"


def _normalize_lines_arg(args: dict[str, Any]) -> tuple[int, int | None]:
    offset = _to_int(args.get("offset"), 1, lo=1)
    raw_limit = args.get("limit")
    limit = None if raw_limit is None else _to_int(raw_limit, 0, lo=0)
    return offset, limit


def _unified_diff(old: str, new: str, path: str, *, context_lines: int = 3) -> str:
    context = _to_int(context_lines, 3, lo=0, hi=20)
    diff = "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"{path} (before)",
            tofile=f"{path} (after)",
            n=context,
        )
    )
    return _truncate(diff, _MAX_DIFF_CHARS) if diff else "(no textual diff)"


def _write_text_atomic(path: str, content: str) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _handle_read_file(args: dict[str, Any]) -> str:
    path = str(args.get("path") or "")
    offset, limit = _normalize_lines_arg(args)
    line_numbers = bool(args.get("line_numbers", False))
    ok, content = _read_text(path)
    if not ok:
        return content

    lines = content.splitlines()
    selected = lines[offset - 1:]
    if limit is not None:
        selected = selected[:limit]
    if line_numbers:
        selected = [f"{line_no}| {line}" for line_no, line in enumerate(selected, start=offset)]

    result = "\n".join(selected)
    if not result:
        return "(empty file or offset beyond end)"

    total = len(lines)
    end_line = offset + len(selected) - 1
    meta: list[str] = []
    if limit is not None and end_line < total:
        meta.append(f"(pagination: offset={offset}, limit={limit}, total_lines={total})")
    elif offset > 1:
        meta.append(f"(pagination: offset={offset}, total_lines={total})")
    out = _truncate(result)
    if meta:
        out += "\n\n" + "\n".join(meta)
    return out


def _handle_write_file(args: dict[str, Any]) -> str:
    path = str(args.get("path") or "")
    content = str(args.get("content") or "")
    mode = str(args.get("mode") or "overwrite").strip().lower()
    show_diff = bool(args.get("show_diff", False))
    if mode not in {"overwrite", "append", "prepend", "create"}:
        return f"Error: unsupported mode '{mode}' (expected overwrite, append, prepend, or create)"
    if not path.strip():
        return "Error: path is required"

    p = Path(path).expanduser()
    existed = p.exists()
    if existed and not p.is_file():
        return f"Error: Not a file: {path}"
    if mode == "create" and existed:
        return f"Error: File already exists: {path}"

    old = ""
    if existed:
        ok, old_or_err = _read_text(path)
        if not ok:
            return old_or_err
        old = old_or_err

    if mode in {"overwrite", "create"}:
        new = content
    elif mode == "append":
        new = old + content
    else:  # prepend
        new = content + old

    try:
        _write_text_atomic(path, new)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

    action = "Created" if (mode == "create" or not existed) else {"overwrite": "Written", "append": "Appended", "prepend": "Prepended"}[mode]
    msg = f"OK: {action} {len(content)} chars to {path} (final size {len(new)} chars)"
    if show_diff:
        msg += "\n\n" + _unified_diff(old, new, path)
    return msg


def _replace_nth(content: str, old_text: str, new_text: str, occurrence: int) -> str:
    if occurrence < 1:
        raise ValueError("occurrence must be >= 1")
    start = -1
    cursor = 0
    for _ in range(occurrence):
        start = content.find(old_text, cursor)
        if start < 0:
            raise ValueError(f"old_text occurrence {occurrence} not found")
        cursor = start + len(old_text)
    return content[:start] + new_text + content[start + len(old_text):]


def _handle_edit_file(args: dict[str, Any]) -> str:
    path = str(args.get("path") or "")
    old_text = str(args.get("old_text") or "")
    new_text = str(args.get("new_text") or "")
    replace_all = bool(args.get("replace_all", False))
    occurrence_arg = args.get("occurrence")
    expected_count_arg = args.get("expected_count")
    context_lines = _to_int(args.get("context_lines"), 3, lo=0, hi=20)

    if not path.strip():
        return "Error: path is required"
    if old_text == "":
        return "Error: old_text must not be empty"

    ok, content = _read_text(path)
    if not ok:
        return content

    count = content.count(old_text)
    if expected_count_arg is not None:
        expected = _to_int(expected_count_arg, -1, lo=0)
        if count != expected:
            return f"Error: expected old_text to appear {expected} time(s), found {count}"
    if count == 0:
        return f"Error: old_text not found in {path}"

    try:
        if replace_all:
            new_content = content.replace(old_text, new_text)
            replaced = count
        else:
            if occurrence_arg is not None:
                occurrence = _to_int(occurrence_arg, 1, lo=1)
                if occurrence > count:
                    return f"Error: occurrence {occurrence} requested, but old_text appears {count} time(s)"
                new_content = _replace_nth(content, old_text, new_text, occurrence)
                replaced = 1
            else:
                if count > 1:
                    return (
                        f"Error: old_text found {count} times in {path} (not unique). "
                        "Use replace_all=true, occurrence=N, expected_count, or provide more context."
                    )
                new_content = content.replace(old_text, new_text, 1)
                replaced = 1
        _write_text_atomic(path, new_content)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

    diff = _unified_diff(content, new_content, path, context_lines=context_lines)
    return f"OK: Replaced {replaced} occurrence(s) in {path}\n\n{diff}"


async def _handle_shell(args: dict[str, Any]) -> str:
    command = args.get("command", "")
    timeout = args.get("timeout", 30)
    cwd = args.get("cwd")
    if not command.strip():
        return "(empty command)"

    import asyncio

    proc = None
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            _kill_proc(proc)
            await _reap(proc)
            return f"Error: Command timed out after {timeout}s"

        stdout = (stdout_b or b"").decode("utf-8", errors="replace")
        stderr = (stderr_b or b"").decode("utf-8", errors="replace")
        output = stdout
        if stderr:
            output += f"\nSTDERR:\n{stderr}"
        if proc.returncode not in (0, None):
            output += f"\nReturn code: {proc.returncode}"
        return _truncate(output) if output.strip() else "(no output)"
    except asyncio.CancelledError:
        # User pressed Stop while the command was running — kill the child process
        # so it doesn't keep running in the background, then propagate cancel.
        if proc is not None:
            _kill_proc(proc)
            await _reap(proc)
        raise
    except Exception as e:
        if proc is not None:
            _kill_proc(proc)
        return f"Error: {type(e).__name__}: {e}"


def _kill_proc(proc) -> None:
    """Best-effort terminate→kill of an asyncio subprocess."""
    try:
        if proc.returncode is None:
            proc.terminate()
    except (ProcessLookupError, Exception):
        pass
    try:
        if proc.returncode is None:
            proc.kill()
    except (ProcessLookupError, Exception):
        pass


async def _reap(proc) -> None:
    """Wait briefly for the killed process to exit, ignoring errors."""
    import asyncio
    try:
        await asyncio.wait_for(proc.wait(), timeout=2.0)
    except (asyncio.TimeoutError, Exception):
        pass

def _handle_grep(args: dict[str, Any]) -> str:
    return run_grep(args)

def _handle_core_memory_append(args: dict[str, Any]) -> str:
    from agent.core_memory import disk_append
    return json.dumps(disk_append(str(args.get("content") or ""), args.get("priority")), ensure_ascii=False)

def _handle_core_memory_update(args: dict[str, Any]) -> str:
    from agent.core_memory import disk_update
    return json.dumps(disk_update(str(args.get("id") or ""), str(args.get("content") or ""), args.get("priority")), ensure_ascii=False)


# ── Registration ────────────────────────────────────────────────────────

_TOOLS = [
    ("exec", _handle_exec),
    ("read_file", _handle_read_file),
    ("write_file", _handle_write_file),
    ("edit_file", _handle_edit_file),
    ("shell", _handle_shell),
    ("grep", _handle_grep),
    ("core_memory_append", _handle_core_memory_append),
    ("core_memory_update", _handle_core_memory_update),
]

def register_all_tools() -> None:
    for name, handler in _TOOLS:
        register_tool(name, handler)

def get_tool_definitions() -> list[dict]:
    return TOOL_DEFINITIONS
