"""
Loop control — tail parsing, chain decision, gap timing.

Tail modes (last matching line in the tail wins): ``/next`` / ``/next <sec>``,
``/sleep`` / ``/sleep <sec>``.

**Process start:** the *first* successful completion also chains once like ``/next`` unless
the tail requests **wait** (``/sleep``) or a hard stop.
"""
from __future__ import annotations

import re
from typing import Any

from agent.config import Config

_CHAIN_GAP_ABS_MAX = 86_400.0


def clamp_chain_gap_seconds(sec: float) -> float:
    return max(0.0, min(float(sec), _CHAIN_GAP_ABS_MAX))


def clamp_wait_seconds(sec: float, max_sec: float) -> float:
    cap = max(0.0, float(max_sec))
    return max(0.0, min(float(sec), cap))


def tail_after_last_code_fence(B: str) -> str:
    """Substring after the last ``\\n```\\n`` in B."""
    if not B:
        return ""
    i = B.rfind("\n```\n")
    return B[i:] if i != -1 else B


# Line-start or end-of-line (allows ``... prose /next 30`` on one line).
# Groups 1 vs 2 capture the optional number. The optional unit is accepted so the
# prompt can say "xx 秒" naturally while the host still stores plain seconds.
_RE_NEXT = re.compile(
    r"(?m)^/next(?:\s+(\d+)\s*(?:秒|s|sec)?)?(?=\s*$)|/next(?:\s+(\d+)\s*(?:秒|s|sec)?)?\s*$",
    re.IGNORECASE,
)
_RE_SLEEP = re.compile(
    r"(?m)^/sleep(?:\s+(\d+)\s*(?:秒|s|sec)?)?(?=\s*$)|/sleep(?:\s+(\d+)\s*(?:秒|s|sec)?)?\s*$",
    re.IGNORECASE,
)


def _last_regex_match(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    last: re.Match[str] | None = None
    for m in pattern.finditer(text):
        last = m
    return last


def parse_tail(ai_text: str) -> dict[str, Any]:
    """Parse assistant tail. The last recognized tail token wins."""
    tail = tail_after_last_code_fence(ai_text or "")
    candidates: list[tuple[int, str, str | None]] = []

    m = _last_regex_match(_RE_NEXT, tail)
    if m:
        num = m.group(1) or m.group(2)
        candidates.append((m.end(), "next", num))

    m = _last_regex_match(_RE_SLEEP, tail)
    if m:
        num = m.group(1) or m.group(2)
        candidates.append((m.end(), "sleep", num))

    if not candidates:
        return {
            "tail_text": tail,
            "tail_mode": None,
            "next_arg_sec": None,
            "sleep_arg_sec": None,
        }

    _, mode, extra = max(candidates, key=lambda x: x[0])
    next_arg: int | None = None
    sleep_arg: int | None = None
    if mode == "next" and extra is not None:
        next_arg = int(extra)
    if mode == "sleep" and extra is not None:
        sleep_arg = int(extra)
    return {
        "tail_text": tail,
        "tail_mode": mode,
        "next_arg_sec": next_arg,
        "sleep_arg_sec": sleep_arg,
    }


def should_chain(ai_text: str, *, boot_auto_chain: bool = False) -> bool:
    """Chain on ``/next`` and legacy chain tails, or once at process start when not waiting."""
    B = ai_text or ""
    if not B.strip():
        return False
    if B.lstrip().startswith("[Error]"):
        return False
    if "[System - infer cancelled by user]" in B:
        return False

    tp = parse_tail(B)
    mode = tp["tail_mode"]
    if mode == "sleep":
        return False
    if mode == "next":
        return True
    if boot_auto_chain:
        return True
    return False


def chain_gap_for_tail_mode(tail: dict[str, Any] | str | None) -> tuple[float, str]:
    """Seconds before the next chained infer (interruptible by triggers)."""
    cfg = Config.get()
    if isinstance(tail, dict):
        tail_mode = tail.get("tail_mode")
        if tail_mode == "next":
            raw = tail.get("next_arg_sec")
            base = float(raw) if raw is not None else float(cfg.self_continue_gap_sec)
            return clamp_chain_gap_seconds(base), "next"
    else:
        tail_mode = tail
    return clamp_chain_gap_seconds(cfg.self_continue_gap_sec), "next"


def wait_timeout_for_tail(tp: dict[str, Any]) -> tuple[float, str]:
    """(seconds, label) for the wait phase; human input or timeout ends the wait."""
    cfg = Config.get()
    mode = tp.get("tail_mode")
    if mode == "sleep":
        raw = tp.get("sleep_arg_sec")
        base = float(raw) if raw is not None else float(cfg.sleep_default_sec)
        sec = clamp_wait_seconds(base, cfg.sleep_max_sec)
        return sec, "sleep"
    sec = clamp_wait_seconds(cfg.sleep_default_sec, cfg.sleep_max_sec)
    return sec, "fallback"
