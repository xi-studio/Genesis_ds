"""
Infer — LLM Chat Completions with OpenAI-compatible APIs.

Infer history comes from ``perceive()``: a **slice** of the SQLite consciousness log,
``consciousness_messages.id`` from ``agent_state.infer_context_start_id`` through ``infer_context_end_id``
(inclusive). New turns are appended only to ``consciousness_messages``; the window extends ``end_id``.
**Trimming** advances ``infer_context_start_id`` when over ``context_window_max_tokens`` (ref.tok,
optionally scaled by the last chat API ``usage.prompt_tokens``), until about
``context_window_tail_tokens`` heuristic sum, then the prefix is kept API-valid. The chat request is
``system`` + one Core Memory snapshot ``user`` turn + infer slice.

When ``tool_definitions`` is non-empty, runs a multi-turn loop until the model returns
text without ``tool_calls``. Each LLM round respects ``stream``: streaming rounds
accumulate ``tool_calls`` from deltas; non-streaming uses one blocking response.
Cancellation (Ctrl+C / ``/cancel``) is honored at ``await`` boundaries between tools;
long **synchronous** tool handlers still block until they return.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from openai import BadRequestError

from agent.config import Config
from agent.timestamp import now_local
from agent.consciousness import (
    compose_infer_messages,
    extend_messages,
    perceive,
    record_infer_prompt_usage,
)
from agent.output import say
from agent.prompt import infer_system_content
from agent.tools import run_tool as run_tool_fn
from agent.tokenizer import update_ratio_from_usage
import agent.ui_stub as _us


# ---------------------------------------------------------------------------
# Small helpers — unified dict/object access
# ---------------------------------------------------------------------------

def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Dict or object access — handles OpenAI SDK's mixed pydantic/dict shapes."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _maybe_model_dump(obj: Any) -> dict[str, Any] | None:
    """Try model_dump on a pydantic object; return None on failure."""
    if obj is None or not hasattr(obj, "model_dump"):
        return None
    try:
        return obj.model_dump(mode="python")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Reasoning / content extraction (unified delta + message)
# ---------------------------------------------------------------------------

_REASONING_KEYS = ("reasoning_content", "reasoningContent", "reasoning", "thinking", "thought")
_REASONING_SUBKEYS = ("content", "text", "value", "reasoning", "message")


def _extract_reasoning(obj: Any) -> str:
    """Extract reasoning/thinking text from a dict, pydantic object, or None."""
    if obj is None:
        return ""
    # Try model_dump first (covers pydantic + __pydantic_extra__ in one pass)
    d = _maybe_model_dump(obj)
    if d is not None:
        for key in _REASONING_KEYS:
            v = d.get(key)
            if isinstance(v, str) and v:
                return v
            if isinstance(v, dict):
                for sub in _REASONING_SUBKEYS:
                    inner = v.get(sub)
                    if isinstance(inner, str) and inner:
                        return inner
        return ""
    if isinstance(obj, dict):
        for key in _REASONING_KEYS:
            v = obj.get(key)
            if isinstance(v, str) and v:
                return v
            if isinstance(v, dict):
                for sub in _REASONING_SUBKEYS:
                    inner = v.get(sub)
                    if isinstance(inner, str) and inner:
                        return inner
        return ""
    # Fallback: getattr
    for key in ("reasoning_content", "reasoning", "thinking"):
        v = getattr(obj, key, None)
        if isinstance(v, str) and v:
            return v
    return ""


def _extract_content(obj: Any) -> str:
    """Normalize ``content`` from SDK objects (str, list of content parts, or None)."""
    c = _get(obj, "content") if obj is not None else None
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for p in c:
            if isinstance(p, dict):
                if str(p.get("type") or "") == "text" or "text" in p:
                    parts.append(str(p.get("text") or ""))
            elif hasattr(p, "model_dump"):
                try:
                    pd = p.model_dump(mode="python")
                    if str(pd.get("type") or "") == "text" or pd.get("text") is not None:
                        parts.append(str(pd.get("text") or ""))
                except Exception:
                    pass
        return "".join(parts)
    return str(c)


def _extract_reasoning_and_content(obj: Any) -> tuple[str, str]:
    """Unified extraction: reasoning + content from delta or message."""
    if obj is None:
        return "", ""
    return _extract_reasoning(obj), _extract_content(obj)


# ---------------------------------------------------------------------------
# Stream creation
# ---------------------------------------------------------------------------

async def _create_stream(client, **kwargs) -> Any:
    """Create a stream, with optional ``stream_options`` for usage tracking."""
    try:
        return await client.chat.completions.create(
            **kwargs, stream_options={"include_usage": True}
        )
    except (TypeError, BadRequestError):
        return await client.chat.completions.create(**kwargs)


# ---------------------------------------------------------------------------
# Chat message assembly
# ---------------------------------------------------------------------------

def _compose_api_messages(
    sys_content: str, infer_window: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    from agent.core_memory import user_message_dict
    sys_msg: dict[str, Any] = {"role": "system", "content": sys_content}
    return [sys_msg, user_message_dict(), *_sanitize_chat_messages(infer_window)]


def _sanitize_chat_messages(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only fields required for chat.completions. DeepSeek: reasoning_content
    only for assistant turns with tool_calls."""
    clean: list[dict[str, Any]] = []
    for m in msgs:
        if not isinstance(m, dict) or not m.get("role"):
            continue
        d: dict[str, Any] = {"role": m["role"]}
        for k in ("name", "tool_calls", "tool_call_id"):
            if m.get(k) is not None:
                d[k] = m[k]
        if m.get("role") == "tool":
            d["tool_call_id"] = str(m.get("tool_call_id", ""))
        # reasoning_content: only assistant + tool_calls
        if m.get("role") == "assistant" and m.get("tool_calls") and "reasoning_content" in m:
            rc = m["reasoning_content"]
            d["reasoning_content"] = rc if isinstance(rc, str) else str(rc) if rc is not None else ""
        c = m.get("content")
        if m.get("tool_calls") and (c is None or c == ""):
            d["content"] = None
        else:
            d["content"] = c if c is not None else ""
        clean.append(d)
    return clean


# ---------------------------------------------------------------------------
# Token calibration
# ---------------------------------------------------------------------------

def _messages_json_for_estimate(messages: list[dict[str, Any]]) -> str:
    try:
        return json.dumps(messages, ensure_ascii=False)
    except (TypeError, ValueError):
        return "".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))


def _calibrate_from_messages(messages: list[dict], usage) -> None:
    if usage is None:
        return
    pt = getattr(usage, "prompt_tokens", None)
    if pt is None or pt <= 0:
        return
    update_ratio_from_usage(_messages_json_for_estimate(messages), pt)


# ---------------------------------------------------------------------------
# Shared stream processing
# ---------------------------------------------------------------------------

_TOOL_CALLS_UI_PLACEHOLDER = "(assistant: tool_calls)\n"


def _clamp_ui_text(s: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        return "", False
    if len(s) <= max_chars:
        return s, False
    return s[:max_chars], True


async def _emit_tool_call_ui(cfg: Config, name: str, tool_call_id: str, arguments: str) -> None:
    if (name or "").strip() == "exec":
        return
    cap = max(4096, min(48_000, int(cfg.exec_stdout_max_chars)))
    arg_show, arg_trunc = _clamp_ui_text(arguments or "", cap)
    await _us.emit_ui_event({
        "event": "tool_call", "name": name, "tool_call_id": tool_call_id or "",
        "arguments": arg_show, "truncated": arg_trunc,
    })


async def _emit_tool_result_ui(cfg: Config, name: str, tool_call_id: str, out: str) -> None:
    if (name or "").strip() == "exec":
        return
    cap = max(1024, int(cfg.exec_stdout_max_chars))
    body, trunc = _clamp_ui_text(out or "", cap)
    await _us.emit_ui_event({
        "event": "tool_result", "name": name, "tool_call_id": tool_call_id or "",
        "text": body, "truncated": trunc,
    })


async def _emit_assistant_round_ui(reasoning: str, content_piece: str | None, *, has_tool_calls: bool) -> None:
    c = content_piece if isinstance(content_piece, str) else ""
    has_reasoning = bool((reasoning or "").strip())
    if reasoning:
        await _us.emit_ui_event({"event": "think_delta", "text": reasoning})
    if c:
        if has_tool_calls and not has_reasoning:
            await _us.emit_ui_event({"event": "think_delta", "text": c})
        else:
            await _us.emit_ui_event({"event": "reply_delta", "text": c})
    elif has_tool_calls and not has_reasoning:
        await _us.emit_ui_event({"event": "reply_delta", "text": _TOOL_CALLS_UI_PLACEHOLDER})


def _feed_tool_delta_fragments(acc: dict[int, dict[str, Any]], delta_tool_calls) -> None:
    if not delta_tool_calls:
        return
    for tc in delta_tool_calls:
        idx = _get(tc, "index")
        if idx is None:
            continue
        slot = acc.setdefault(idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
        tid = _get(tc, "id")
        if isinstance(tid, str) and tid.strip():
            slot["id"] = tid
        ttype = _get(tc, "type")
        if isinstance(ttype, str) and ttype.strip():
            slot["type"] = ttype
        fn = _get(tc, "function")
        if fn is None:
            continue
        nm = _get(fn, "name") if isinstance(fn, dict) else getattr(fn, "name", None)
        arg = _get(fn, "arguments") if isinstance(fn, dict) else getattr(fn, "arguments", None)
        if isinstance(nm, str) and nm:
            slot["function"]["name"] = (slot["function"]["name"] or "") + nm
        if isinstance(arg, str) and arg:
            slot["function"]["arguments"] = (slot["function"]["arguments"] or "") + arg


def _tool_acc_to_messages_list(acc: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx in sorted(acc.keys()):
        slot = acc[idx]
        fn = slot.get("function") or {}
        name = (fn.get("name") or "").strip()
        arguments = fn.get("arguments")
        if not isinstance(arguments, str):
            arguments = ""
        tid = (slot.get("id") or "").strip()
        if not name and not tid:
            continue
        out.append({
            "id": tid or f"call_idx_{idx}", "type": slot.get("type") or "function",
            "function": {"name": name, "arguments": arguments},
        })
    return out


async def _process_stream(
    stream, cfg: Config, *, has_tools: bool
) -> tuple[str, str, dict[int, dict[str, Any]], Any, str | None, bool]:
    """Process one streaming response. Returns (reasoning, content, tool_acc, usage, finish_reason, sent_think).

    When ``has_tools``, all visible content goes to think_delta (tool rounds).
    Otherwise, content goes to reply_delta.
    """
    thr = cfg.stream_flush_chars
    label = "tools" if has_tools else ""
    batched = "batched" if thr > 0 else ""
    parts = [s for s in ["assistant", "streaming", label, batched] if s]
    say(f"--- {' ('.join(parts[1:]) + ')' if len(parts) > 2 else ''} ---" if len(parts) > 1 else "--- assistant (streaming) ---")

    buf: list[str] = []
    blen = 0

    def _flush_buf():
        nonlocal buf, blen
        if not buf:
            return
        say("".join(buf), end="", flush=True)
        buf.clear()
        blen = 0

    usage_seen = None
    finish_reason: str | None = None
    tool_acc: dict[int, dict[str, Any]] = {}
    reasoning_acc = ""
    content_acc = ""
    sent_think = False

    try:
        async for chunk in stream:
            u = getattr(chunk, "usage", None)
            if u is not None:
                usage_seen = u
            if not chunk.choices:
                continue
            ch0 = chunk.choices[0]
            fr = getattr(ch0, "finish_reason", None)
            if isinstance(fr, str) and fr:
                finish_reason = fr
            delta = ch0.delta
            if delta is None:
                continue

            if has_tools:
                dtools = _get(delta, "tool_calls")
                if dtools:
                    _feed_tool_delta_fragments(tool_acc, dtools)

            r, c = _extract_reasoning_and_content(delta)
            event_type = "think_delta" if has_tools else "reply_delta"
            if r:
                reasoning_acc += r
                sent_think = True
                await _us.emit_ui_event({"event": "think_delta", "text": r})
                await asyncio.sleep(0)
                if thr <= 0:
                    say(r, end="", flush=True)
                else:
                    buf.append(r)
                    blen += len(r)
            if c:
                content_acc += c
                sent_think = True
                await _us.emit_ui_event({"event": event_type, "text": c})
                await asyncio.sleep(0)
                if thr <= 0:
                    say(c, end="", flush=True)
                else:
                    buf.append(c)
                    blen += len(c)
            if thr > 0 and blen >= thr:
                _flush_buf()
        if thr > 0:
            _flush_buf()
        say("", flush=True)
    finally:
        await stream.close()

    # Post-stream: handle edge case where think deltas weren't sent but content exists
    combined = (reasoning_acc + content_acc).strip()
    tcs_list = _tool_acc_to_messages_list(tool_acc)
    if tcs_list and combined and not sent_think:
        await _us.emit_ui_event({"event": "think_set", "text": combined})
        await asyncio.sleep(0)
    if tcs_list and not reasoning_acc.strip() and not content_acc.strip():
        await _us.emit_ui_event({"event": "reply_delta", "text": _TOOL_CALLS_UI_PLACEHOLDER})

    return reasoning_acc, content_acc, tool_acc, usage_seen, finish_reason, sent_think


# ---------------------------------------------------------------------------
# UI payload helpers
# ---------------------------------------------------------------------------

def _usage_for_emit(usage) -> dict[str, int]:
    if usage is None:
        return {}
    out: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        v = getattr(usage, key, None)
        if v is not None:
            try:
                out[key] = int(v)
            except (TypeError, ValueError):
                pass
    return out


def _infer_end_ok_payload(usage, *, display_text=None, think_text=None, reply_text=None) -> dict[str, Any]:
    ev: dict[str, Any] = {"event": "infer_end", "ok": True, "server_time": now_local(), **_usage_for_emit(usage)}
    for key, val in (("think_text", think_text), ("reply_text", reply_text), ("display_text", display_text)):
        if val is not None and str(val).strip():
            ev[key] = str(val)
    return ev


# ---------------------------------------------------------------------------
# SDK message serialization
# ---------------------------------------------------------------------------

def _sdk_tool_call_to_dict(tc) -> dict[str, Any]:
    d = _maybe_model_dump(tc)
    if d is not None:
        return d
    fn = getattr(tc, "function", None)
    return {
        "id": getattr(tc, "id", "") or "",
        "type": getattr(tc, "type", None) or "function",
        "function": {
            "name": getattr(fn, "name", "") if fn else "",
            "arguments": (getattr(fn, "arguments", None) or "") if fn else "",
        },
    }


def sdk_assistant_message_to_dict(msg: Any) -> dict[str, Any]:
    if msg is None:
        return {"role": "assistant", "content": ""}
    d = _maybe_model_dump(msg)
    if d is not None:
        return d
    out: dict[str, Any] = {
        "role": getattr(msg, "role", None) or "assistant",
        "content": getattr(msg, "content", None),
    }
    rc = getattr(msg, "reasoning_content", None)
    if isinstance(rc, str):
        out["reasoning_content"] = rc
    tcs = getattr(msg, "tool_calls", None)
    if tcs:
        out["tool_calls"] = [_sdk_tool_call_to_dict(tc) for tc in tcs]
    return out


def _normalize_assistant_tool_call_ids(adict: dict[str, Any]) -> None:
    tcs = adict.get("tool_calls")
    if not isinstance(tcs, list):
        return
    for idx, tc in enumerate(tcs):
        if isinstance(tc, dict) and not str(tc.get("id") or "").strip():
            tc["id"] = f"call_{idx}"


# ---------------------------------------------------------------------------
# Build API kwargs (shared)
# ---------------------------------------------------------------------------

def _api_kwargs(cfg: Config, messages: list[dict[str, Any]], *,
                stream: bool, tools: list[dict[str, Any]] | None = None,
                _extra: dict[str, Any] | None = None) -> dict[str, Any]:
    kw: dict[str, Any] = dict(
        model=cfg.model, messages=messages,
        temperature=cfg.temperature, max_tokens=cfg.max_tokens,
        stream=stream,
    )
    if tools:
        kw["tools"] = tools
        kw["tool_choice"] = "auto"
    if _extra:
        kw["extra_body"] = _extra
    return kw


# ---------------------------------------------------------------------------
# Top-level infer entry
# ---------------------------------------------------------------------------

async def infer(history: list[dict[str, Any]]) -> str:
    cfg = Config.get()
    client = cfg.openai_client
    await _us.emit_ui_event({"event": "infer_begin", "server_time": now_local()})

    if client is None:
        await _us.emit_ui_event({"event": "infer_end", "ok": False, "server_time": now_local(), "error": "OpenAI client not initialized"})
        return "[Error] OpenAI client not initialized (call apply_config first)"
    if not cfg.api_key:
        await _us.emit_ui_event({"event": "infer_end", "ok": False, "server_time": now_local(), "error": "missing api_key"})
        return "[Error] Missing api_key"

    sys_content = infer_system_content()
    tools = cfg.tool_definitions if cfg.tool_definitions else None
    _extra = dict(cfg.openai_extra_body) if cfg.openai_extra_body else {}

    if tools:
        return await _infer_with_tools_loop(client, cfg, history, sys_content, tools, _extra)

    hist = compose_infer_messages(history)
    messages = _compose_api_messages(sys_content, hist)
    if cfg.stream:
        return await _infer_streaming(client, cfg, messages, _extra, infer_window=hist)
    return await _infer_nonstream(client, cfg, messages, _extra, infer_window=hist)


# ---------------------------------------------------------------------------
# Non-streaming (plain + tool rounds)
# ---------------------------------------------------------------------------

async def _infer_nonstream(client, cfg: Config, messages: list[dict[str, Any]],
                           _extra: dict[str, Any], tools: list[dict[str, Any]] | None = None,
                           *, infer_window: list[dict[str, Any]] | None = None) -> str | tuple:
    """Single non-streaming completion. Returns str (plain) or tuple (tool round)."""
    kw = _api_kwargs(cfg, messages, stream=False, tools=tools, _extra=_extra)
    resp = await client.chat.completions.create(**kw)
    usage = getattr(resp, "usage", None)
    ch = resp.choices[0] if resp.choices else None
    raw_msg = ch.message if ch else None

    if tools:
        adict = sdk_assistant_message_to_dict(raw_msg)
        reasoning, content_piece = _extract_reasoning_and_content(raw_msg)
        disp = reasoning + (content_piece or "")
        if not disp.strip() and adict.get("tool_calls"):
            disp = _TOOL_CALLS_UI_PLACEHOLDER
        say(disp if disp.strip() else "\n", flush=True)
        await _emit_assistant_round_ui(reasoning, content_piece, has_tool_calls=bool(adict.get("tool_calls")))
        return adict, usage, getattr(ch, "finish_reason", None) if ch else None

    # Plain non-stream
    reasoning, reply = _extract_reasoning_and_content(raw_msg)
    ai_text = reasoning + reply
    await _emit_assistant_round_ui(reasoning, reply, has_tool_calls=False)
    say(ai_text, flush=True)
    row: dict[str, Any] = {"role": "assistant", "content": reply if reply else None}
    if row.get("content") is None and not reasoning:
        row["content"] = ""
    extend_messages([row])
    await _us.emit_ui_event(_infer_end_ok_payload(usage, display_text=ai_text,
                                                   think_text=reasoning or None, reply_text=reply or None))
    _calibrate_from_messages(messages + [row], usage)
    record_infer_prompt_usage(messages, usage, infer_window=infer_window)
    return ai_text


# ---------------------------------------------------------------------------
# Streaming (plain)
# ---------------------------------------------------------------------------

async def _infer_streaming(client, cfg: Config, messages: list[dict[str, Any]],
                           _extra: dict[str, Any],
                           *, infer_window: list[dict[str, Any]] | None = None) -> str:
    stream = await _create_stream(client, **_api_kwargs(cfg, messages, stream=True, _extra=_extra))
    reasoning_acc, content_acc, _, usage_seen, _, _ = await _process_stream(stream, cfg, has_tools=False)

    ai_text = reasoning_acc + content_acc
    if reasoning_acc or content_acc:
        row: dict[str, Any] = {"role": "assistant", "content": content_acc if content_acc else None}
        if row.get("content") is None and not reasoning_acc:
            row["content"] = ""
        extend_messages([row])
        await _us.emit_ui_event(_infer_end_ok_payload(
            usage_seen, display_text=ai_text,
            think_text=reasoning_acc or None, reply_text=content_acc or None))
        _calibrate_from_messages(messages + [row], usage_seen)
        record_infer_prompt_usage(messages, usage_seen, infer_window=infer_window)
    return ai_text


# ---------------------------------------------------------------------------
# Tool loop
# ---------------------------------------------------------------------------

async def _run_tool_round(client, cfg: Config, messages: list[dict[str, Any]],
                          _extra: dict[str, Any], tools: list[dict[str, Any]]) -> tuple[dict[str, Any], Any]:
    """One LLM round (stream or non-stream) with tools. Returns (assistant_dict, usage)."""
    if cfg.stream:
        stream = await _create_stream(client, **_api_kwargs(cfg, messages, stream=True, tools=tools, _extra=_extra))
        reasoning, content, tool_acc, usage, _, _ = await _process_stream(stream, cfg, has_tools=True)
        adict: dict[str, Any] = {"role": "assistant"}
        if reasoning:
            adict["reasoning_content"] = reasoning
        tcs_list = _tool_acc_to_messages_list(tool_acc)
        if tcs_list:
            adict["tool_calls"] = tcs_list
            adict["content"] = content if content else None
        else:
            adict["content"] = content
        return adict, usage
    else:
        return (await _infer_nonstream(client, cfg, messages, _extra, tools=tools))[:2]


async def _infer_with_tools_loop(client, cfg: Config, history: list[dict[str, Any]],
                                  sys_content: str, tools: list[dict[str, Any]],
                                  _extra: dict[str, Any]) -> str:
    first = True
    last_usage: Any = None

    for _round in range(max(1, int(cfg.max_tool_rounds))):
        hist = compose_infer_messages(history) if first else compose_infer_messages(perceive())
        first = False
        messages = _compose_api_messages(sys_content, hist)

        try:
            adict, last_usage = await _run_tool_round(client, cfg, messages, _extra, tools)
        except asyncio.CancelledError:
            say("\n  [infer cancelled — Ctrl+C]\n", flush=True)
            await _us.emit_ui_event({"event": "infer_end", "ok": False, "server_time": now_local(), "cancelled": True})
            return "[System - infer cancelled by user]\n"
        except Exception as e:
            await _us.emit_ui_event({"event": "infer_end", "ok": False, "server_time": now_local(), "error": f"{type(e).__name__}: {e}"})
            return f"[Error] {type(e).__name__}: {e}"

        reasoning = str(adict.get("reasoning_content") or "")
        c_str = adict.get("content")
        if not isinstance(c_str, str):
            c_str = ""

        tcs = adict.get("tool_calls")
        if tcs:
            _normalize_assistant_tool_call_ids(adict)
            extend_messages([adict])
            tool_rows: list[dict[str, Any]] = []
            for tc in tcs:
                await asyncio.sleep(0)
                fn = tc.get("function") or {}
                name = (fn.get("name") or "").strip()
                args = fn.get("arguments")
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False) if args is not None else "{}"
                tc_id = str(tc.get("id") or "")
                try:
                    await _emit_tool_call_ui(cfg, name, tc_id, args)
                    out = await run_tool_fn(name, args)
                except asyncio.CancelledError:
                    say("\n  [infer cancelled — during tool run]\n", flush=True)
                    await _us.emit_ui_event({"event": "infer_end", "ok": False, "server_time": now_local(), "cancelled": True})
                    return "[System - infer cancelled by user]\n"
                await _emit_tool_result_ui(cfg, name, tc_id, out)
                tool_rows.append({"role": "tool", "tool_call_id": tc.get("id") or "", "content": out})
            extend_messages(tool_rows)
            continue

        # Final text reply
        persist: dict[str, Any] = dict(adict)
        persist.pop("reasoning_content", None)
        extend_messages([persist])
        final_piece = reasoning + c_str
        _calibrate_from_messages(messages + [persist], last_usage)
        record_infer_prompt_usage(messages, last_usage, infer_window=hist)
        await _us.emit_ui_event(_infer_end_ok_payload(
            last_usage, display_text=final_piece,
            think_text=reasoning or None, reply_text=c_str or None))
        return final_piece

    # Tool round limit
    ts = now_local()
    final_piece = (
        f"System - [ToolRoundLimit] [{ts}] "
        f"max_tool_rounds ({cfg.max_tool_rounds}) reached without a final text reply. "
        f"Continuing host cycle.\n\n/next\n"
    )
    hist_tail = compose_infer_messages(perceive())
    tail_messages = _compose_api_messages(sys_content, hist_tail)
    _calibrate_from_messages(tail_messages, last_usage)
    record_infer_prompt_usage(tail_messages, last_usage, infer_window=hist_tail)
    extend_messages([{"role": "assistant", "content": final_piece}])
    say(final_piece, flush=True)
    ev = _infer_end_ok_payload(last_usage, display_text=final_piece, reply_text=final_piece)
    ev["tool_round_limit"] = True
    await _us.emit_ui_event(ev)
    return final_piece
