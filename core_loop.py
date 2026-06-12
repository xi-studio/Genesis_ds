"""
Core Loop (DeepSeek) — perceive → infer (tools) → chain/wait.

Use ``config.json`` with ``api.deepseek.com`` and optional ``llm_extra_body`` for
``thinking`` / ``reasoning_effort``. API key: ``DEEPSEEK_API_KEY`` or ``api_key`` in config.

Every cycle: perceive → infer (tool loop) → chain/wait. Code runs only via the ``exec`` tool.

**Chain:** ``/next`` waits ``self_continue_gap_sec`` (default 15s) before the next infer;
``/next <sec>`` overrides the gap. First successful completion also chains once unless
the tail is a **wait** tail.

**Wait**: ``/sleep`` or ``/sleep <sec>`` (defaults
and cap from ``sleep_default_sec`` / ``sleep_max_sec``), or an implicit wait when not chaining — all
use the same trigger inbox; human input wakes early.

Gap / wait sleeps are interruptible: triggers wake immediately.
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from agent.timestamp import now_local

from agent.config import Config
from agent.output import say
from agent.consciousness import (
    append,
    perceive_and_ref_token_len,
)
from agent.infer import infer
from agent.tools import register_all_tools, set_exec_globals
from agent.host_primitives import (
    init as hp_init,
    drain_triggers_to_consciousness,
    install_signal_handler,
    set_current_infer_task,
    exec_globals,
)
from agent.loop_control import (
    should_chain,
    parse_tail,
    chain_gap_for_tail_mode,
    wait_timeout_for_tail,
)
import agent.ui_stub as _ui_stub


def _strip_digital_being_header(text: str) -> str:
    return re.sub(
        r"^\*{0,2}Digital Being\s*[-–—]\s*\[.*?\]\*{0,2}\n?",
        "",
        text or "",
    )


def _append_digital_being(ai_text: str, ts: str, tag: str) -> None:
    clean = _strip_digital_being_header(ai_text).strip()
    if clean:
        append(
            f"**Digital Being - [{ts}] {tag}**\n{clean}\n\n",
            role="assistant",
        )


async def _interruptible_sleep(seconds: float, inbox: asyncio.Queue) -> None:
    """Sleep for *seconds*, but wake immediately if a trigger arrives.

    If woken early, the trigger is put back so the next drain picks it up.
    """
    if seconds <= 0:
        return
    try:
        msg = await asyncio.wait_for(inbox.get(), timeout=seconds)
        # Woken by trigger — put it back for the next cycle's drain
        inbox.put_nowait(msg)
    except asyncio.TimeoutError:
        pass  # normal expiry


async def main():
    cfg = Config.get()
    cfg.load_and_apply()

    if cfg.terminal_quiet and not cfg.log_file:
        print(
            "[core_loop] terminal_quiet with no log_file — set log_file in config for debug",
            file=sys.stderr,
            flush=True,
        )

    os.makedirs(cfg.workspace_rel, exist_ok=True)

    # [Evolution Plasmid Hook]
    if os.path.exists("evolution_patch.py"):
        try:
            with open("evolution_patch.py", "r", encoding="utf-8") as _f:
                exec(_f.read(), globals())
            say("[System] Evolution plasmid injected successfully.")
        except Exception as _e:
            say(f"[System] Failed to inject plasmid: {_e}")

    loop = asyncio.get_running_loop()
    trigger_inbox: asyncio.Queue = asyncio.Queue()
    hp_init(loop, trigger_inbox)
    install_signal_handler()

    # Shared ``exec`` tool globals (host primitives + curated namespace)
    _exec_globals = exec_globals()
    set_exec_globals(_exec_globals)
    register_all_tools()

    # Startup info
    key_hint = (
        f"{cfg.api_key[:8]}…"
        if len(cfg.api_key) >= 8
        else ("(missing)" if not cfg.api_key else "(short)")
    )
    say(f"  config: {cfg.config_path}")
    say(f"  base_url: {cfg.api_base_url}")
    say(f"  model: {cfg.model}")
    say(f"  stream: {cfg.stream}")
    if cfg.stream:
        say(f"  stream_flush_chars: {cfg.stream_flush_chars} (0 = each chunk)")
    say(f"  api_key: {key_hint}")
    say(f"  chain: /next → {cfg.self_continue_gap_sec}s · /next <sec> overrides")
    say(
        f"  wait: /sleep default → {cfg.sleep_default_sec}s (cap {cfg.sleep_max_sec}s)"
    )
    say(f"  exec_batch_interrupt_on_human: {cfg.exec_batch_interrupt_on_human}")
    if cfg.log_file:
        say(f"  log_file: {os.path.abspath(cfg.log_file)}")
    say(
        f"=== MolAgent Core Loop ===\n"
        f"  agent SQLite: {cfg.agent_db_file}\n"
        f"    tables: consciousness_messages, agent_state; core_memory_entries, core_memory_snapshot\n"
        f"  infer context: agent_state.infer_context_start_id … end_id → consciousness_messages.id\n"
        f"  context cap: ≤{cfg.context_window_max_tokens} ref. tok in windows; over cap → drop whole "
        f"messages from head until ≤{cfg.context_window_tail_tokens}\n"
        f"  tools: {len(cfg.tool_definitions)} (each model round follows config.stream)\n"
        f"  workspace: {os.path.abspath(cfg.workspace_rel)}\n"
    )

    # --- Main loop ---
    boot_auto_chain = True  # first successful infer chains like /next

    while True:
        _, had_human_this_round = drain_triggers_to_consciousness()

        # Perceive
        ctx, rtoks = await asyncio.to_thread(perceive_and_ref_token_len)
        await _ui_stub.emit_ui_event(
            {
                "event": "phase",
                "phase": "perceive",
                "messages": len(ctx),
                "approx_ref_tokens": rtoks,
            }
        )
        say(f"--- perceive: {len(ctx)} messages · ~{rtoks} tok ---")

        # Infer
        say("--- infer ---")
        infer_t = asyncio.create_task(infer(ctx))
        set_current_infer_task(infer_t)
        try:
            ai_text = await infer_t
        finally:
            set_current_infer_task(None)

        # Parse tail
        tp = parse_tail(ai_text or "")

        # Log assistant turn: infer persists via extend_messages (reasoning_content only when tool_calls).
        if (ai_text or "").strip():
            ts = now_local()
            tag = "[Being]"
            if (ai_text or "").strip().startswith("[Error]"):
                _append_digital_being(ai_text, ts, tag)
        else:
            say("  [loop] empty model response; skip append")

        # Chain or wait
        cancelled = "[System - infer cancelled by user]" in (ai_text or "")
        if not (ai_text or "").strip():
            say("  [loop] stop: empty — waiting for trigger")
            cont = False
            tp = parse_tail("")
        else:
            cont = should_chain(ai_text, boot_auto_chain=boot_auto_chain)
            if boot_auto_chain and (ai_text or "").strip() and not cancelled:
                boot_auto_chain = False

        tm = tp.get("tail_mode")
        explicit_chain = tm == "next"
        boot_chain = bool(
            cont and not explicit_chain and tm != "sleep"
        )

        say(
            f"  [loop] after_human={had_human_this_round} "
            f"tail_mode={tm} boot_chain={boot_chain} → {'chain' if cont else 'wait'}"
        )

        await _ui_stub.emit_ui_event(
            {
                "event": "host_decision",
                "chain": cont,
                "after_human": had_human_this_round,
                "tail_mode": tm,
                "tail_next": tm == "next",
                "tail_sleep": tm == "sleep",
                "boot_chain": boot_chain,
            }
        )

        if cont:
            gap, mode = chain_gap_for_tail_mode(tp)
            await _ui_stub.emit_ui_event(
                {
                    "event": "phase",
                    "phase": "chain",
                    "gap_sec": gap,
                    "gap_mode": mode,
                }
            )
            say(f"--- chain (gap {gap}s · {mode}) ---")
            await _interruptible_sleep(gap, trigger_inbox)
            continue

        wait_sec, wait_lbl = wait_timeout_for_tail(tp)
        await _ui_stub.emit_ui_event(
            {
                "event": "phase",
                "phase": "wait",
                "watchdog_sec": wait_sec,
                "wait_timeout_sec": wait_sec,
                "wait_mode": wait_lbl,
            }
        )
        say(f"--- wait ({wait_lbl} · timeout {wait_sec}s) ---")
        try:
            msg = await asyncio.wait_for(trigger_inbox.get(), timeout=wait_sec)
        except asyncio.TimeoutError:
            msg = f"[wait_timeout] {wait_sec}s ({wait_lbl}), auto-waking"
        # Grace period to collect concurrent triggers
        await asyncio.sleep(0.5)
        msgs = [msg]
        while not trigger_inbox.empty():
            msgs.append(trigger_inbox.get_nowait())
        merged = "\n".join(msgs)
        ts = now_local()
        append(f"System - [Trigger] [{ts}] {merged}\n\n", role="user")
        say(f"--- triggered ({len(msgs)} msgs, {len(merged)} chars) ---\n{merged}\n---")


if __name__ == "__main__":
    _cfg = Config.get()
    try:
        _cfg.load_and_apply()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        raise SystemExit(1) from e
    # web_listen / ui_listen 需要 monkey-patch emit_ui_event；仅在本进程启动 aiohttp / TCP ui 时才会注入。
    if _cfg.web_port > 0 or _cfg.ui_port > 0:
        import main  # noqa: PLC0415 — same package, loads aiohttp only when needed

        raise SystemExit(main.main())
    asyncio.run(main())
