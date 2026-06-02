"""
Built-in tool schemas (OpenAI-compatible) and handler registration.

Registers handlers with ``agent.tools.dispatch``.
"""
from __future__ import annotations

import json
import subprocess
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
        "Read a file and return its contents. Supports line offset and limit.",
        path=_p("string", "File path to read"),
        offset=_p("integer", "Line number to start from (1-based, default 1)", required=False),
        limit=_p("integer", "Max lines to return (default all)", required=False),
    ),
    _tool("write_file",
        "Write content to a file, creating parent directories if needed.",
        path=_p("string", "File path to write"),
        content=_p("string", "Content to write"),
    ),
    _tool("edit_file",
        "Replace old_text with new_text in a file. Fails if old_text not found.",
        path=_p("string", "File path to edit"),
        old_text=_p("string", "Text to find (must be unique in file)"),
        new_text=_p("string", "Replacement text"),
        replace_all=_p("boolean", "Replace all occurrences (default false)", required=False),
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

def _read_path(path: str) -> str:
    """Read a file, return truncated content or error."""
    p = Path(path)
    if not p.exists():
        return f"Error: File not found: {path}"
    if not p.is_file():
        return f"Error: Not a file: {path}"
    return p.read_text(encoding="utf-8", errors="replace")

def _truncate(s: str, max_chars: int = _MAX_OUTPUT) -> str:
    return s if len(s) <= max_chars else s[:max_chars] + "\n... (truncated)"


def _handle_read_file(args: dict[str, Any]) -> str:
    path = args.get("path", "")
    offset = args.get("offset", 1)
    limit = args.get("limit")
    try:
        lines = _read_path(path).splitlines()
        if lines and lines[0].startswith("Error:"):
            return lines[0]
        selected = lines[offset - 1:]
        if limit is not None:
            selected = selected[:limit]
        result = "\n".join(selected)
        return _truncate(result) if result else "(empty file or offset beyond end)"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

def _handle_write_file(args: dict[str, Any]) -> str:
    path = args.get("path", "")
    content = args.get("content", "")
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"OK: Written {len(content)} chars to {path}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

def _handle_edit_file(args: dict[str, Any]) -> str:
    path = args.get("path", "")
    old_text = args.get("old_text", "")
    new_text = args.get("new_text", "")
    replace_all = args.get("replace_all", False)
    try:
        content = _read_path(path)
        if content.startswith("Error:"):
            return content
        if old_text not in content:
            return f"Error: old_text not found in {path}"
        count = content.count(old_text)
        if not replace_all and count > 1:
            return f"Error: old_text found {count} times in {path} (not unique). Use replace_all=true or provide more context."
        new_content = content.replace(old_text, new_text) if replace_all else content.replace(old_text, new_text, 1)
        Path(path).write_text(new_content, encoding="utf-8")
        return f"OK: Replaced {count} occurrence(s) in {path}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

def _handle_shell(args: dict[str, Any]) -> str:
    command = args.get("command", "")
    timeout = args.get("timeout", 30)
    cwd = args.get("cwd")
    if not command.strip():
        return "(empty command)"
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if result.returncode != 0:
            output += f"\nReturn code: {result.returncode}"
        return _truncate(output) if output.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

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
