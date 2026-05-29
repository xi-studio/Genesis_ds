"""
Built-in tool schemas (OpenAI-compatible) and handler registration.

Registers handlers with ``agent.tools.dispatch``.
"""
from __future__ import annotations

import json
import subprocess
from typing import Any

from .dispatch import register_tool
from .grep_tool import run_grep

# ── Tool Schemas (OpenAI function calling format) ─────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "exec",
            "description": (
                "Execute Python on the host in the shared exec namespace "
                "(``trigger``, etc.). "
                "Top-level await is supported. For files use the ``read_file`` / ``write_file`` / "
                "``edit_file`` / ``grep`` tools or ``open()`` in code; prefer ``shell`` when it fits."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file and return its contents. Supports line offset and limit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                    "offset": {"type": "integer", "description": "Line number to start from (1-based, default 1)"},
                    "limit": {"type": "integer", "description": "Max lines to return (default all)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating parent directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace old_text with new_text in a file. Fails if old_text not found.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to edit"},
                    "old_text": {"type": "string", "description": "Text to find (must be unique in file)"},
                    "new_text": {"type": "string", "description": "Replacement text"},
                    "replace_all": {"type": "boolean", "description": "Replace all occurrences (default false)"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": (
                "Run a shell command; returns stdout+stderr. Long jobs: nohup + log + &; read the log later. "
                "Default timeout 30s."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
                    "cwd": {"type": "string", "description": "Working directory (default current)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search file contents with a regex (or plain text if fixed_strings=true). "
                "Default output_mode is files_with_matches (paths only). Use output_mode=content "
                "for matching lines with optional context. Skips binary and files >2MB; ignores "
                ".git, node_modules, __pycache__, .venv. Paths are cwd-relative like read_file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern, or literal if fixed_strings=true",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search (default '.')",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Optional path filter, e.g. '*.py' or 'tests/**/test_*.py'",
                    },
                    "type": {
                        "type": "string",
                        "description": "Optional type shorthand: py, ts, md, json, yaml, ...",
                    },
                    "case_insensitive": {"type": "boolean"},
                    "fixed_strings": {
                        "type": "boolean",
                        "description": "If true, pattern is plain text (not regex)",
                    },
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "files_with_matches", "count"],
                        "description": (
                            "files_with_matches: list paths (default); content: lines + context; "
                            "count: match counts per file"
                        ),
                    },
                    "context_before": {
                        "type": "integer",
                        "description": "Lines of context before each match in content mode (0-20)",
                    },
                    "context_after": {
                        "type": "integer",
                        "description": "Lines of context after each match in content mode (0-20)",
                    },
                    "head_limit": {
                        "type": "integer",
                        "description": "Max results per mode (default 250); 0 = no limit",
                    },
                    "max_matches": {
                        "type": "integer",
                        "description": "Alias for head_limit in content mode",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Alias for head_limit in files_with_matches / count mode",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Skip the first N hits before applying head_limit",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "core_memory_append",
            "description": (
                "Append one concise note to core memory (SQLite source table). "
                "priority: P1 = permanent (no passive expiry), P2 = kept 7 days after updated_at, "
                "P3 = kept 24 hours (passive purge when the injected snapshot syncs). Default P3."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Markdown-ready note body; keep short and high-signal",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["P1", "P2", "P3"],
                        "description": "Retention tier (default P3)",
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "core_memory_update",
            "description": (
                "Replace content of one core memory entry by id. "
                "Optional priority P1/P2/P3. To retire a note, set **P3** (and shorten content if needed); "
                "passive TTL removes it when the injected snapshot syncs — there is no delete tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Entry id from the injected Core Memory list"},
                    "content": {"type": "string", "description": "New markdown body"},
                    "priority": {
                        "type": "string",
                        "enum": ["P1", "P2", "P3"],
                        "description": "If set, new retention tier",
                    },
                },
                "required": ["id", "content"],
            },
        },
    },
]

_exec_globals: dict[str, Any] = {
    "__builtins__": __builtins__,
}


def set_exec_globals(g: dict[str, Any]) -> None:
    """Inject host primitives and shared state into exec globals."""
    _exec_globals.update(g)


# ── Tool Handlers ─────────────────────────────────────────────────────────

async def _handle_exec(args: dict[str, Any]) -> str:
    """Run host Python via ``exec_engine`` (same semantics as legacy /exec)."""
    from agent.exec_engine import run_exec_source_once

    code = args.get("code", "")
    return await run_exec_source_once(code, _exec_globals)


def _handle_read_file(args: dict[str, Any]) -> str:
    """Read a file, return contents."""
    from pathlib import Path

    path = args.get("path", "")
    offset = args.get("offset", 1)
    limit = args.get("limit")

    p = Path(path)
    if not p.exists():
        return f"Error: File not found: {path}"
    if not p.is_file():
        return f"Error: Not a file: {path}"

    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        selected = lines[offset - 1 :]
        if limit is not None:
            selected = selected[:limit]
        result = "\n".join(selected)
        if not result:
            return "(empty file or offset beyond end)"
        # Truncate
        max_chars = 32000
        if len(result) > max_chars:
            result = result[:max_chars] + f"\n... (truncated)"
        return result
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def _handle_write_file(args: dict[str, Any]) -> str:
    """Write content to a file."""
    from pathlib import Path

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
    """Edit a file by replacing old_text with new_text."""
    from pathlib import Path

    path = args.get("path", "")
    old_text = args.get("old_text", "")
    new_text = args.get("new_text", "")
    replace_all = args.get("replace_all", False)

    try:
        p = Path(path)
        if not p.exists():
            return f"Error: File not found: {path}"

        content = p.read_text(encoding="utf-8")
        if old_text not in content:
            return f"Error: old_text not found in {path}"

        if replace_all:
            new_content = content.replace(old_text, new_text)
            count = content.count(old_text)
        else:
            # Check uniqueness
            count = content.count(old_text)
            if count > 1:
                return f"Error: old_text found {count} times in {path} (not unique). Use replace_all=true or provide more context."
            new_content = content.replace(old_text, new_text, 1)

        p.write_text(new_content, encoding="utf-8")
        return f"OK: Replaced {count} occurrence(s) in {path}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def _handle_shell(args: dict[str, Any]) -> str:
    """Run a shell command."""
    command = args.get("command", "")
    timeout = args.get("timeout", 30)
    cwd = args.get("cwd")

    if not command.strip():
        return "(empty command)"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if result.returncode != 0:
            output += f"\nReturn code: {result.returncode}"
        # Truncate
        max_chars = 32000
        if len(output) > max_chars:
            output = output[:max_chars] + f"\n... (truncated)"
        return output if output.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def _handle_grep(args: dict[str, Any]) -> str:
    return run_grep(args)


def _handle_core_memory_append(args: dict[str, Any]) -> str:
    from agent.core_memory import disk_append

    return json.dumps(
        disk_append(str(args.get("content") or ""), args.get("priority")),
        ensure_ascii=False,
    )


def _handle_core_memory_update(args: dict[str, Any]) -> str:
    from agent.core_memory import disk_update

    return json.dumps(
        disk_update(
            str(args.get("id") or ""),
            str(args.get("content") or ""),
            args.get("priority"),
        ),
        ensure_ascii=False,
    )



def register_all_tools() -> None:
    """Register all built-in tools."""
    register_tool("exec", _handle_exec)
    register_tool("read_file", _handle_read_file)
    register_tool("write_file", _handle_write_file)
    register_tool("edit_file", _handle_edit_file)
    register_tool("shell", _handle_shell)
    register_tool("grep", _handle_grep)
    register_tool("core_memory_append", _handle_core_memory_append)
    register_tool("core_memory_update", _handle_core_memory_update)


def get_tool_definitions() -> list[dict]:
    """Return tool definitions for the LLM API."""
    return TOOL_DEFINITIONS
