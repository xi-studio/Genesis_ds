"""Dispatch model function/tool calls to Python handlers (name -> callable)."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

JsonDict = dict[str, Any]

Handler = Callable[[JsonDict], Any] | Callable[[JsonDict], Awaitable[Any]]

_REGISTRY: dict[str, Handler] = {}


def register_tool(name: str, fn: Handler) -> None:
    _REGISTRY[name.strip()] = fn


def registered_tool_names() -> frozenset[str]:
    return frozenset(_REGISTRY.keys())


async def run_tool(name: str, arguments_json: str) -> str:
    """Parse JSON arguments, invoke handler, return string for ``tool`` message."""
    key = (name or "").strip()
    fn = _REGISTRY.get(key)
    if fn is None:
        return json.dumps(
            {"error": f"unknown tool {key!r} — register via agent.tools.dispatch.register_tool"},
            ensure_ascii=False,
        )
    try:
        args: JsonDict = json.loads(arguments_json) if (arguments_json or "").strip() else {}
        if not isinstance(args, dict):
            args = {"_args": args}
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"invalid tool arguments JSON: {e}"}, ensure_ascii=False)

    try:
        result = fn(args)
        if asyncio.iscoroutine(result):
            result = await result
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 — tool boundary
        return json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)

    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


def _register_builtin_tools() -> None:
    def echo(args: JsonDict) -> str:
        return json.dumps({"echo": args.get("text", "")}, ensure_ascii=False)

    register_tool("echo", echo)


_register_builtin_tools()
