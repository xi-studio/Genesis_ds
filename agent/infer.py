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
from datetime import datetime
from typing import Any

from openai import BadRequestError

from agent.config import Config
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


def _compose_api_messages(
    sys_content: str, infer_window: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """``[system, user(core memory snapshot), …history]``."""
    from agent.core_memory import user_message_dict

    sys_msg: dict[str, Any] = {"role": "system", "content": sys_content}
    hist = _sanitize_chat_messages(infer_window)
    return [sys_msg, user_message_dict(), *hist]


def _sanitize_chat_messages(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep fields required for chat.completions.

    DeepSeek: only assistant turns **with** ``tool_calls`` must include ``reasoning_content``
    on the wire; plain replies may omit it (we do not persist those chains either).
    """
    clean: list[dict[str, Any]] = []
    for m in msgs:
        if not isinstance(m, dict) or not m.get("role"):
            continue
        d: dict[str, Any] = {"role": m["role"]}
        if m.get("name") is not None:
            d["name"] = m["name"]
        if m.get("tool_calls") is not None:
            d["tool_calls"] = m["tool_calls"]
        if m.get("role") == "tool":
            tid = m.get("tool_call_id")
            d["tool_call_id"] = "" if tid is None else str(tid)
        elif m.get("tool_call_id") is not None:
            d["tool_call_id"] = m["tool_call_id"]
        if (
            m.get("role") == "assistant"
            and m.get("tool_calls")
            and "reasoning_content" in m
        ):
            rc = m.get("reasoning_content")
            if isinstance(rc, str):
                d["reasoning_content"] = rc
            elif rc is not None:
                d["reasoning_content"] = str(rc)
        c = m.get("content")
        if m.get("tool_calls") and (c is None or c == ""):
            d["content"] = None
        elif c is not None:
            d["content"] = c
        else:
            d["content"] = ""
        clean.append(d)
    return clean


def _messages_json_for_estimate(messages: list[dict[str, Any]]) -> str:
    """Same blob as :func:`_messages_char_estimate` length; used for tokenizer calibration."""
    try:
        return json.dumps(messages, ensure_ascii=False)
    except (TypeError, ValueError):
        parts: list[str] = []
        for m in messages:
            c = m.get("content")
            if isinstance(c, str):
                parts.append(c)
        return "".join(parts)


def _messages_char_estimate(messages: list[dict[str, Any]]) -> int:
    return len(_messages_json_for_estimate(messages))


def _calibrate_from_messages(messages: list[dict], usage) -> None:
    if usage is None:
        return
    pt = getattr(usage, "prompt_tokens", None)
    if pt is None or pt <= 0:
        return
    update_ratio_from_usage(_messages_json_for_estimate(messages), pt)


def _stringify_content_piece(c: Any) -> str:
    """Normalize ``content`` from SDK objects (str or list of content parts)."""
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


def _reasoning_from_mapping(d: dict[str, Any]) -> str:
    """Extract streaming / message reasoning text from provider-specific dict shapes."""
    for key in (
        "reasoning_content",
        "reasoningContent",
        "reasoning",
        "thinking",
        "thought",
    ):
        v = d.get(key)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, dict):
            for sub in ("content", "text", "value", "reasoning", "message"):
                inner = v.get(sub)
                if isinstance(inner, str) and inner:
                    return inner
    return ""


def _delta_reasoning_and_content(delta) -> tuple[str, str]:
    if delta is None:
        return "", ""
    if isinstance(delta, dict):
        r = _reasoning_from_mapping(delta)
        c = _stringify_content_piece(delta.get("content"))
        return r, c
    if hasattr(delta, "model_dump"):
        try:
            d = delta.model_dump(mode="python")
            r = _reasoning_from_mapping(d)
            c = _stringify_content_piece(d.get("content"))
            if r or c:
                return r, c
        except Exception:
            pass
        extras = getattr(delta, "__pydantic_extra__", None)
        if isinstance(extras, dict) and extras:
            r2 = _reasoning_from_mapping(extras)
            if r2:
                c2 = _stringify_content_piece(extras.get("content"))
                return (r2, c2) if c2 else (r2, "")
    r = getattr(delta, "reasoning_content", None)
    if not isinstance(r, str) or r == "":
        r = getattr(delta, "reasoning", None)
    if not isinstance(r, str):
        r = getattr(delta, "thinking", None)
    reasoning = r if isinstance(r, str) else ""
    c = getattr(delta, "content", None)
    content = _stringify_content_piece(c)
    return reasoning, content


def _message_reasoning_and_content(message) -> tuple[str, str]:
    if message is None:
        return "", ""
    if hasattr(message, "model_dump"):
        try:
            d = message.model_dump(mode="python")
            reasoning = _reasoning_from_mapping(d)
            content = _stringify_content_piece(d.get("content"))
            return reasoning, content
        except Exception:
            pass
    r = getattr(message, "reasoning_content", None)
    if not isinstance(r, str) or r == "":
        r = getattr(message, "reasoning", None)
    if not isinstance(r, str):
        r = getattr(message, "thinking", None)
    reasoning = r if isinstance(r, str) else ""
    c = getattr(message, "content", None)
    content = _stringify_content_piece(c)
    return reasoning, content


_TOOL_CALLS_UI_PLACEHOLDER = "(assistant: tool_calls)\n"


def _clamp_ui_text(s: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        return "", False
    if len(s) <= max_chars:
        return s, False
    return s[:max_chars], True


async def _emit_tool_call_ui(cfg: Config, name: str, tool_call_id: str, arguments: str) -> None:
    """Web UI: tool name + JSON args. Skip ``exec`` — :mod:`exec_engine` emits ``exec`` / ``exec_stdout``."""
    if (name or "").strip() == "exec":
        return
    cap = max(4096, min(48_000, int(cfg.exec_stdout_max_chars)))
    arg_show, arg_trunc = _clamp_ui_text(arguments or "", cap)
    await _us.emit_ui_event(
        {
            "event": "tool_call",
            "name": name,
            "tool_call_id": tool_call_id or "",
            "arguments": arg_show,
            "truncated": arg_trunc,
        }
    )


async def _emit_tool_result_ui(cfg: Config, name: str, tool_call_id: str, out: str) -> None:
    if (name or "").strip() == "exec":
        return
    cap = max(1024, int(cfg.exec_stdout_max_chars))
    body, trunc = _clamp_ui_text(out or "", cap)
    await _us.emit_ui_event(
        {
            "event": "tool_result",
            "name": name,
            "tool_call_id": tool_call_id or "",
            "text": body,
            "truncated": trunc,
        }
    )


async def _emit_assistant_round_ui(reasoning: str, content_piece: str | None, *, has_tool_calls: bool) -> None:
    """Mirror terminal output to WebSocket (think_delta / reply_delta); stub no-ops if no UI."""
    c = content_piece if isinstance(content_piece, str) else ""
    has_reasoning = bool((reasoning or "").strip())
    if reasoning:
        await _us.emit_ui_event({"event": "think_delta", "text": reasoning})
    if c:
        # Tool rounds: some APIs stream readable preamble only in ``content`` (no ``reasoning_content``).
        # Log shows it via ``say``; web must use ``think_delta`` or the 「思考」 block stays empty.
        if has_tool_calls and not has_reasoning:
            await _us.emit_ui_event({"event": "think_delta", "text": c})
        else:
            await _us.emit_ui_event({"event": "reply_delta", "text": c})
    elif has_tool_calls and not has_reasoning:
        await _us.emit_ui_event({"event": "reply_delta", "text": _TOOL_CALLS_UI_PLACEHOLDER})


def _delta_tool_calls(delta) -> Any:
    if delta is None:
        return None
    if isinstance(delta, dict):
        return delta.get("tool_calls")
    return getattr(delta, "tool_calls", None)


def _feed_tool_delta_fragments(
    acc: dict[int, dict[str, Any]], delta_tool_calls
) -> None:
    if not delta_tool_calls:
        return
    for tc in delta_tool_calls:
        if isinstance(tc, dict):
            idx = tc.get("index")
            tid_raw = tc.get("id")
            ttype = tc.get("type")
            fn = tc.get("function")
        else:
            idx = getattr(tc, "index", None)
            tid_raw = getattr(tc, "id", None)
            ttype = getattr(tc, "type", None)
            fn = getattr(tc, "function", None)
        if idx is None:
            continue
        slot = acc.setdefault(
            idx,
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
        )
        tid = tid_raw
        if isinstance(tid, str) and tid.strip():
            slot["id"] = tid
        if isinstance(ttype, str) and ttype.strip():
            slot["type"] = ttype
        if fn is None:
            continue
        if isinstance(fn, dict):
            nm = fn.get("name")
            arg = fn.get("arguments")
        else:
            nm = getattr(fn, "name", None)
            arg = getattr(fn, "arguments", None)
        if isinstance(nm, str) and nm:
            slot["function"]["name"] = (slot["function"]["name"] or "") + nm
        if isinstance(arg, str) and arg:
            f = slot["function"]
            f["arguments"] = (f["arguments"] or "") + arg


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
        out.append(
            {
                "id": tid or f"call_idx_{idx}",
                "type": slot.get("type") or "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return out


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


def _infer_end_ok_payload(
    usage,
    *,
    display_text: str | None = None,
    think_text: str | None = None,
    reply_text: str | None = None,
) -> dict[str, Any]:
    """WS payload for a successful infer.

    ``think_text`` / ``reply_text`` help the web UI when stream deltas were missed; keys
    starting with ``_`` are avoided. ``display_text`` remains ``reasoning + reply`` for
    simple fallbacks.
    """
    ev: dict[str, Any] = {"event": "infer_end", "ok": True, **_usage_for_emit(usage)}
    if think_text is not None and str(think_text).strip():
        ev["think_text"] = str(think_text)
    if reply_text is not None and str(reply_text).strip():
        ev["reply_text"] = str(reply_text)
    if display_text is not None and str(display_text).strip():
        ev["display_text"] = str(display_text)
    return ev


def sdk_assistant_message_to_dict(msg: Any) -> dict[str, Any]:
    """Serializable assistant message for persistence (tool_calls included)."""
    if msg is None:
        return {"role": "assistant", "content": ""}
    if hasattr(msg, "model_dump"):
        try:
            return msg.model_dump(mode="json", exclude_none=True)
        except TypeError:
            d = msg.model_dump(exclude_none=True)
            return json.loads(json.dumps(d, default=str))

    role = getattr(msg, "role", None) or "assistant"
    content = getattr(msg, "content", None)
    out: dict[str, Any] = {"role": role, "content": content}
    rc = getattr(msg, "reasoning_content", None)
    if isinstance(rc, str):
        out["reasoning_content"] = rc
    tcs = getattr(msg, "tool_calls", None)
    if not tcs:
        return out
    ser: list[dict[str, Any]] = []
    for tc in tcs:
        if hasattr(tc, "model_dump"):
            try:
                ser.append(tc.model_dump(mode="json", exclude_none=True))
            except TypeError:
                ser.append(json.loads(json.dumps(tc.model_dump(exclude_none=True), default=str)))
        else:
            fn = getattr(tc, "function", None)
            ser.append(
                {
                    "id": getattr(tc, "id", "") or "",
                    "type": getattr(tc, "type", None) or "function",
                    "function": {
                        "name": getattr(fn, "name", "") if fn else "",
                        "arguments": (getattr(fn, "arguments", None) or "") if fn else "",
                    },
                }
            )
    out["tool_calls"] = ser
    return out


def _normalize_assistant_tool_call_ids(adict: dict[str, Any]) -> None:
    """Ensure every tool call has a non-empty id so matching tool rows persist correctly."""
    tcs = adict.get("tool_calls")
    if not isinstance(tcs, list):
        return
    for idx, tc in enumerate(tcs):
        if not isinstance(tc, dict):
            continue
        tid = str(tc.get("id") or "").strip()
        if not tid:
            tc["id"] = f"call_{idx}"


async def infer(history: list[dict[str, Any]]) -> str:
    """Call the LLM; return the final assistant text (reasoning + content) for this turn."""
    cfg = Config.get()
    client = cfg.openai_client
    await _us.emit_ui_event({"event": "infer_begin"})

    if client is None:
        await _us.emit_ui_event(
            {"event": "infer_end", "ok": False, "error": "OpenAI client not initialized"}
        )
        return "[Error] OpenAI client not initialized (call apply_config first)"
    if not cfg.api_key:
        await _us.emit_ui_event({"event": "infer_end", "ok": False, "error": "missing api_key"})
        return "[Error] Missing api_key in config.json or OPENAI_API_KEY / DASHSCOPE_API_KEY / DEEPSEEK_API_KEY"

    sys_content = infer_system_content()
    tools = cfg.tool_definitions if cfg.tool_definitions else None
    _extra = dict(cfg.openai_extra_body) if cfg.openai_extra_body else {}

    if tools:
        return await _infer_with_tools_loop(
            client, cfg, history, sys_content, tools, _extra
        )

    messages = _compose_api_messages(sys_content, compose_infer_messages(history))

    if cfg.stream:
        return await _infer_streaming(client, cfg, messages, _extra)

    return await _infer_single_nonstream(client, cfg, messages, _extra)


async def _complete_tool_round_nonstream(
    client,
    cfg: Config,
    messages: list[dict[str, Any]],
    _extra: dict[str, Any],
    tools: list[dict[str, Any]],
) -> tuple[dict[str, Any], Any, str | None]:
    _ns_kw: dict[str, Any] = dict(
        model=cfg.model,
        messages=messages,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        stream=False,
        tools=tools,
        tool_choice="auto",
    )
    if _extra:
        _ns_kw["extra_body"] = _extra
    resp = await client.chat.completions.create(**_ns_kw)
    last_usage = getattr(resp, "usage", None)
    ch = resp.choices[0] if resp.choices else None
    raw_msg = ch.message if ch else None
    adict = sdk_assistant_message_to_dict(raw_msg)
    reasoning, content_piece = _message_reasoning_and_content(raw_msg)
    disp = reasoning + (content_piece or "")
    if not disp.strip() and adict.get("tool_calls"):
        disp = "(assistant: tool_calls)\n"
    say(disp if disp.strip() else "\n", flush=True)
    await _emit_assistant_round_ui(
        reasoning, content_piece, has_tool_calls=bool(adict.get("tool_calls"))
    )
    finish_reason = getattr(ch, "finish_reason", None) if ch else None
    return adict, last_usage, finish_reason


async def _stream_tool_round(
    client,
    cfg: Config,
    messages: list[dict[str, Any]],
    _extra: dict[str, Any],
    tools: list[dict[str, Any]],
) -> tuple[dict[str, Any], Any, str | None]:
    """One streamed completion with ``tools``; aggregates ``tool_calls`` from deltas.

    Web UI: only **thinking** streams (``think_delta``). All visible ``content`` tokens in
    this round go to the thinking fold so 思考 stays live even after ``tool_calls`` deltas
    begin; tool invocation/results are shown separately via ``tool_call`` / ``tool_result``.
    """
    _stream_kw: dict[str, Any] = dict(
        model=cfg.model,
        messages=messages,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        stream=True,
        tools=tools,
        tool_choice="auto",
    )
    if _extra:
        _stream_kw["extra_body"] = _extra
    try:
        try:
            stream = await client.chat.completions.create(
                **_stream_kw, stream_options={"include_usage": True}
            )
        except (TypeError, BadRequestError):
            stream = await client.chat.completions.create(**_stream_kw)
    except Exception:
        raise

    thr = cfg.stream_flush_chars
    say(
        "--- assistant (streaming; tools) ---"
        if thr <= 0
        else "--- assistant (streaming; tools; batched) ---"
    )
    buf: list[str] = []
    blen = 0

    def _flush_buf() -> None:
        nonlocal buf, blen
        if not buf:
            return
        say("".join(buf), end="", flush=True)
        buf = []
        blen = 0

    usage_seen = None
    finish_reason: str | None = None
    tool_acc: dict[int, dict[str, Any]] = {}
    reasoning_acc = ""
    content_acc = ""
    sent_think_piece = False
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
            dtools = _delta_tool_calls(delta)
            if dtools:
                _feed_tool_delta_fragments(tool_acc, dtools)
            # Stream all assistant ``content`` into 思考 (``think_delta``), not ``reply_delta`` —
            # otherwise, once ``tool_calls`` fragments appear, ``pre_tool`` becomes false and any
            # CoT arriving only in ``content`` would render outside the thinking fold.
            r, c = _delta_reasoning_and_content(delta)
            if r:
                reasoning_acc += r
                sent_think_piece = True
                await _us.emit_ui_event({"event": "think_delta", "text": r})
                await asyncio.sleep(0)
                if thr <= 0:
                    say(r, end="", flush=True)
                else:
                    buf.append(r)
                    blen += len(r)
            if c:
                content_acc += c
                sent_think_piece = True
                await _us.emit_ui_event({"event": "think_delta", "text": c})
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

    tcs_list = _tool_acc_to_messages_list(tool_acc)
    combined = (reasoning_acc + content_acc).strip()
    # Do not emit ``think_set`` after normal streaming: it replaces the whole ``thinkBuf`` on
    # the client and wipes prior tool-rounds' thinking when this infer has multiple model passes.
    if tcs_list and combined and not sent_think_piece:
        await _us.emit_ui_event(
            {"event": "think_set", "text": reasoning_acc + content_acc}
        )
        await asyncio.sleep(0)
    if tcs_list and not reasoning_acc.strip() and not content_acc.strip():
        await _us.emit_ui_event({"event": "reply_delta", "text": _TOOL_CALLS_UI_PLACEHOLDER})
    adict: dict[str, Any] = {"role": "assistant"}
    if reasoning_acc:
        adict["reasoning_content"] = reasoning_acc
    if tcs_list:
        adict["tool_calls"] = tcs_list
        adict["content"] = content_acc if content_acc else None
    else:
        adict["content"] = content_acc
    return adict, usage_seen, finish_reason


async def _infer_with_tools_loop(
    client,
    cfg: Config,
    history: list[dict[str, Any]],
    sys_content: str,
    tools: list[dict[str, Any]],
    _extra: dict[str, Any],
) -> str:
    """Multi-turn tools: each model round streams or blocks per ``cfg.stream``."""
    first = True
    last_usage: Any = None

    for _round in range(max(1, int(cfg.max_tool_rounds))):
        if first:
            hist = compose_infer_messages(history)
            first = False
        else:
            hist = compose_infer_messages(perceive())
        messages = _compose_api_messages(sys_content, hist)

        try:
            if cfg.stream:
                adict, last_usage, _finish_reason = await _stream_tool_round(
                    client, cfg, messages, _extra, tools
                )
            else:
                adict, last_usage, _finish_reason = await _complete_tool_round_nonstream(
                    client, cfg, messages, _extra, tools
                )
        except asyncio.CancelledError:
            say("\n  [infer cancelled — Ctrl+C]\n", flush=True)
            await _us.emit_ui_event({"event": "infer_end", "ok": False, "cancelled": True})
            return "[System - infer cancelled by user]\n"
        except Exception as e:
            await _us.emit_ui_event(
                {"event": "infer_end", "ok": False, "error": f"{type(e).__name__}: {e}"}
            )
            return f"[Error] {type(e).__name__}: {e}"

        reasoning = str(adict.get("reasoning_content") or "")
        raw_c = adict.get("content")
        c_str = raw_c if isinstance(raw_c, str) else ""

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
                    await _us.emit_ui_event({"event": "infer_end", "ok": False, "cancelled": True})
                    return "[System - infer cancelled by user]\n"
                await _emit_tool_result_ui(cfg, name, tc_id, out)
                tool_rows.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id") or "",
                        "content": out,
                    }
                )
            extend_messages(tool_rows)
            continue

        persist: dict[str, Any] = dict(adict)
        persist.pop("reasoning_content", None)
        extend_messages([persist])
        final_piece = reasoning + c_str
        _calibrate_from_messages(messages + [persist], last_usage)
        record_infer_prompt_usage(messages, last_usage)
        await _us.emit_ui_event(
            _infer_end_ok_payload(
                last_usage,
                display_text=final_piece,
                think_text=reasoning or None,
                reply_text=c_str or None,
            )
        )
        return final_piece

    # Model still returning tool_calls after max rounds: do not treat as hard [Error] (that would
    # idle-wait); append a synthetic tail with /next so the host chains after self_continue_gap_sec.
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    final_piece = (
        f"System - [ToolRoundLimit] [{ts}] "
        f"max_tool_rounds ({cfg.max_tool_rounds}) reached without a final text reply "
        "(still had tool calls). Continuing host cycle.\n\n"
        "/next\n"
    )
    hist_tail = compose_infer_messages(perceive())
    tail_messages = _compose_api_messages(sys_content, hist_tail)
    _calibrate_from_messages(tail_messages, last_usage)
    record_infer_prompt_usage(tail_messages, last_usage)
    extend_messages([{"role": "assistant", "content": final_piece}])
    say(final_piece, flush=True)
    ev = _infer_end_ok_payload(
        last_usage,
        display_text=final_piece,
        reply_text=final_piece,
    )
    ev["tool_round_limit"] = True
    await _us.emit_ui_event(ev)
    return final_piece


async def _infer_single_nonstream(
    client, cfg: Config, messages: list[dict[str, Any]], _extra: dict[str, Any]
) -> str:
    _ns_kw = dict(
        model=cfg.model,
        messages=messages,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        stream=False,
    )
    if _extra:
        _ns_kw["extra_body"] = _extra
    try:
        resp = await client.chat.completions.create(**_ns_kw)
    except Exception as e:
        await _us.emit_ui_event(
            {"event": "infer_end", "ok": False, "error": f"{type(e).__name__}: {e}"}
        )
        return f"[Error] {type(e).__name__}: {e}"

    ch = resp.choices[0] if resp.choices else None
    msg = ch.message if ch else None
    reasoning, reply = _message_reasoning_and_content(msg)
    ai_text = reasoning + reply
    await _emit_assistant_round_ui(reasoning, reply, has_tool_calls=False)
    say(ai_text, flush=True)
    row: dict[str, Any] = {"role": "assistant", "content": reply if reply else None}
    if row.get("content") is None and not reasoning:
        row["content"] = ""
    extend_messages([row])
    usage = getattr(resp, "usage", None)
    await _us.emit_ui_event(
        _infer_end_ok_payload(
            usage,
            display_text=ai_text,
            think_text=reasoning or None,
            reply_text=reply or None,
        )
    )
    _calibrate_from_messages(messages + [row], usage)
    record_infer_prompt_usage(messages, usage)
    return ai_text


async def _infer_streaming(
    client, cfg: Config, messages: list[dict[str, Any]], _extra: dict[str, Any]
) -> str:
    reasoning_acc = ""
    content_acc = ""
    _stream_kw = dict(
        model=cfg.model,
        messages=messages,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        stream=True,
    )
    if _extra:
        _stream_kw["extra_body"] = _extra
    try:
        try:
            stream = await client.chat.completions.create(
                **_stream_kw, stream_options={"include_usage": True}
            )
        except (TypeError, BadRequestError):
            stream = await client.chat.completions.create(**_stream_kw)

        thr = cfg.stream_flush_chars
        say(
            "--- assistant (streaming) ---"
            if thr <= 0
            else "--- assistant (streaming; batched — pause typing if it scrolls) ---"
        )
        buf: list[str] = []
        blen = 0

        def _flush_buf() -> None:
            nonlocal buf, blen
            if not buf:
                return
            say("".join(buf), end="", flush=True)
            buf = []
            blen = 0

        usage_seen = None
        try:
            async for chunk in stream:
                u = getattr(chunk, "usage", None)
                if u is not None:
                    usage_seen = u
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                r, c = _delta_reasoning_and_content(delta)
                if r:
                    reasoning_acc += r
                    await _us.emit_ui_event({"event": "think_delta", "text": r})
                    await asyncio.sleep(0)
                    if thr <= 0:
                        say(r, end="", flush=True)
                    else:
                        buf.append(r)
                        blen += len(r)
                if c:
                    content_acc += c
                    await _us.emit_ui_event({"event": "reply_delta", "text": c})
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
            ai_text = reasoning_acc + content_acc
            if reasoning_acc or content_acc:
                row: dict[str, Any] = {
                    "role": "assistant",
                    "content": content_acc if content_acc else None,
                }
                if row.get("content") is None and not reasoning_acc:
                    row["content"] = ""
                extend_messages([row])
                end_ev = _infer_end_ok_payload(
                    usage_seen,
                    display_text=ai_text,
                    think_text=reasoning_acc or None,
                    reply_text=content_acc or None,
                )
                await _us.emit_ui_event(end_ev)
                _calibrate_from_messages(messages + [row], usage_seen)
                record_infer_prompt_usage(messages, usage_seen)
            else:
                end_ev = _infer_end_ok_payload(
                    usage_seen,
                    display_text=ai_text,
                    think_text=reasoning_acc or None,
                    reply_text=content_acc or None,
                )
                await _us.emit_ui_event(end_ev)
                _calibrate_from_messages(messages, usage_seen)
                record_infer_prompt_usage(messages, usage_seen)
            return ai_text
        finally:
            await stream.close()

    except asyncio.CancelledError:
        say("\n  [infer cancelled — Ctrl+C]\n", flush=True)
        await _us.emit_ui_event({"event": "infer_end", "ok": False, "cancelled": True})
        if reasoning_acc or content_acc:
            row_cancel: dict[str, Any] = {
                "role": "assistant",
                "content": content_acc if content_acc else None,
            }
            if row_cancel.get("content") is None and not reasoning_acc:
                row_cancel["content"] = ""
            extend_messages([row_cancel])
        ai_text = reasoning_acc + content_acc
        if ai_text.strip():
            return ai_text + "\n\n[System - infer cancelled by user]\n"
        return "[System - infer cancelled by user]\n"

    except Exception as e:
        await _us.emit_ui_event(
            {"event": "infer_end", "ok": False, "error": f"{type(e).__name__}: {e}"}
        )
        return f"[Error] {type(e).__name__}: {e}"
