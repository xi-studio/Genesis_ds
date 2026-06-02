"""
Consciousness + infer context live in **the agent SQLite file** (``agent_db_file``):

* Table ``consciousness_messages`` — append-only full log: ``id`` is the primary key; ``body`` is JSON
  (chat-shaped dict **without** ``id``; runtime attaches ``id`` from the row).
* Table ``agent_state`` (singleton row) — ``infer_context_start_id`` / ``infer_context_end_id`` point at a
  **contiguous slice** of ``consciousness_messages.id`` for the chat API. Trimming only advances
  ``infer_context_start_id``; rows are **not** deleted from consciousness.

**Cold start:** schema only; empty DB is seeded with :func:`_boot_messages`.

Token trim / API-prefix repair: see :func:`_maybe_trim_infer_window`.

**Budget check:** :func:`record_infer_prompt_usage` scales ref.tok like before.

Token estimate: :func:`_single_message_tokens` (approximate).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any

from agent.config import Config
from agent.output import say
from agent.tokenizer import count_tokens

_STORE_LOCK = threading.RLock()
_USAGE_ANCHOR_LOCK = threading.Lock()
_STATE_VERSION = 2

_last_usage_anchor: tuple[int, int] | None = None


# ── DB helpers ──────────────────────────────────────────────────────────

def _db_path() -> str:
    return os.path.abspath(Config.get().agent_db_file)

@contextmanager
def _db():
    path = _db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS consciousness_messages (
          id INTEGER PRIMARY KEY AUTOINCREMENT, body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS agent_state (
          singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
          infer_context_start_id INTEGER, infer_context_end_id INTEGER);
        INSERT OR IGNORE INTO agent_state (singleton, infer_context_start_id, infer_context_end_id)
        VALUES (1, NULL, NULL);
    """)
    _ensure_boot_rows(conn)
    _sync_window_cover_all(conn)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ── Token calibration ───────────────────────────────────────────────────

def record_infer_prompt_usage(messages_for_request: list[dict[str, Any]], usage: Any) -> None:
    global _last_usage_anchor
    pt = getattr(usage, "prompt_tokens", None)
    if pt is None:
        return
    try:
        pi = int(pt)
    except (TypeError, ValueError):
        return
    if pi <= 0:
        return
    ref = sum(_single_message_tokens(m) for m in messages_for_request if isinstance(m, dict))
    if ref > 0:
        with _USAGE_ANCHOR_LOCK:
            _last_usage_anchor = (pi, ref)


def _window_trigger_total(msgs: list[dict[str, Any]]) -> int:
    window_est = sum(_single_message_tokens(m) for m in msgs)
    with _USAGE_ANCHOR_LOCK:
        anchor = _last_usage_anchor
    if anchor is None:
        return window_est
    api_pt, full_ref = anchor
    return max(0, int(window_est * api_pt / full_ref)) if api_pt > 0 and full_ref > 0 else window_est


def _single_message_tokens(m: dict[str, Any]) -> int:
    n = 0
    c = m.get("content")
    if isinstance(c, str) and c:
        n += count_tokens(c)
    rc = m.get("reasoning_content")
    if m.get("tool_calls") and isinstance(rc, str) and rc:
        n += count_tokens(rc)
    if m.get("tool_calls"):
        try:
            n += count_tokens(json.dumps(m["tool_calls"], ensure_ascii=False))
        except (TypeError, ValueError):
            n += 32
    if m.get("role") == "tool" and (tid := m.get("tool_call_id")):
        n += count_tokens(str(tid))
    return max(n, 1)


# ── Boot / Schema ───────────────────────────────────────────────────────

def _boot_messages() -> list[dict[str, Any]]:
    return [{"role": "user", "content": "System - [Boot] Being awakened.\n\n"}]

def _count_messages(conn: sqlite3.Connection) -> int:
    r = conn.execute("SELECT COUNT(*) FROM consciousness_messages").fetchone()
    return int(r[0]) if r else 0

def _ensure_boot_rows(conn: sqlite3.Connection) -> bool:
    if _count_messages(conn) > 0:
        return False
    for bm in _boot_messages():
        row = _normalize_stored_message(bm)
        conn.execute("INSERT INTO consciousness_messages (body) VALUES (?)",
                     (json.dumps(row, ensure_ascii=False),))
    return True


# ── Message I/O ─────────────────────────────────────────────────────────

def _row_to_message(row_id: int, body_raw: str) -> dict[str, Any]:
    try:
        data = json.loads(body_raw)
    except (json.JSONDecodeError, TypeError):
        data = {"role": "user", "content": ""}
    if not isinstance(data, dict):
        data = {"role": "user", "content": ""}
    return {**data, "id": row_id}


def _fetch_range(conn: sqlite3.Connection, start_id: int, end_id: int) -> list[dict[str, Any]]:
    cur = conn.execute(
        "SELECT id, body FROM consciousness_messages WHERE id >= ? AND id <= ? ORDER BY id ASC",
        (start_id, end_id))
    return [_row_to_message(int(r[0]), str(r[1])) for r in cur.fetchall()]


def _window_bounds(conn: sqlite3.Connection) -> tuple[int | None, int | None]:
    row = conn.execute(
        "SELECT infer_context_start_id, infer_context_end_id FROM agent_state WHERE singleton = 1"
    ).fetchone()
    if not row or row[0] is None or row[1] is None:
        return None, None
    try:
        return int(row[0]), int(row[1])
    except (TypeError, ValueError):
        return None, None


def _set_window_bounds(conn: sqlite3.Connection, start_id: int | None, end_id: int | None) -> None:
    conn.execute(
        "UPDATE agent_state SET infer_context_start_id=?, infer_context_end_id=? WHERE singleton=1",
        (start_id, end_id))


def _sync_window_cover_all(conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT MIN(id), MAX(id) FROM consciousness_messages").fetchone()
    if row and row[0] is not None:
        lo, hi = int(row[0]), int(row[1])
        s, e = _window_bounds(conn)
        if s is None or e is None:
            _set_window_bounds(conn, lo, hi)


# ── Message normalization ───────────────────────────────────────────────

_PERSIST_KEYS = {"role", "content", "name", "tool_call_id", "tool_calls"}

def _normalize_stored_message(m: dict[str, Any]) -> dict[str, Any]:
    """Normalize a message dict for persistence (flat API-shaped dict)."""
    m = {k: v for k, v in m.items() if k != "id"}
    role = str(m.get("role") or "user")
    content = m.get("content")
    if not isinstance(content, str):
        content = str(content) if content is not None else ""

    out: dict[str, Any] = {"role": role, "content": content}
    for k in ("name", "tool_call_id", "tool_calls"):
        if m.get(k) is not None:
            out[k] = m[k]

    tcs = out.get("tool_calls")
    for k, v in m.items():
        if k in _PERSIST_KEYS:
            continue
        if role == "assistant" and k == "reasoning_content" and not tcs:
            continue
        if v is not None:
            out[k] = v
    return out


def _strip_msg_id(m: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in m.items() if k != "id"}

def _strip_infer_keys(m: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in m.items() if k != "id" and not str(k).startswith("_")}


# ── Window trimming ─────────────────────────────────────────────────────

def _suffix_is_valid_chat_completions(msgs: list[dict[str, Any]]) -> bool:
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if m.get("role") == "tool":
            return False
        if m.get("role") == "assistant" and (tcs := m.get("tool_calls")):
            if not isinstance(tcs, list) or not tcs:
                i += 1; continue
            n = len(tcs)
            req_ids = [str(tc["id"]) for tc in tcs if isinstance(tc, dict) and str(tc.get("id") or "").strip()]
            if i + n >= len(msgs):
                return False
            got = [str(msgs[i + 1 + k].get("tool_call_id", "")) for k in range(n)
                   if isinstance(msgs[i + 1 + k], dict) and msgs[i + 1 + k].get("role") == "tool"]
            if len(got) != n or (req_ids and set(req_ids) != set(got)):
                return False
            i += 1 + n
            continue
        i += 1
    return True


def _left_trim_to_valid_chat_prefix(msgs: list[dict[str, Any]]) -> int:
    for drop in range(len(msgs) + 1):
        if _suffix_is_valid_chat_completions(msgs[drop:]):
            return drop
    return len(msgs)


def _tail_assistant_awaits_tool_rows(msgs: list[dict[str, Any]]) -> bool:
    if not msgs:
        return False
    last = msgs[-1]
    return last.get("role") == "assistant" and isinstance(last.get("tool_calls"), list) and bool(last["tool_calls"])


def _maybe_trim_infer_window(conn: sqlite3.Connection, *, quiet: bool,
                              with_tool_chain_trim: bool = True) -> bool:
    cfg = Config.get()
    cap = max(1024, int(cfg.context_window_max_tokens))
    tail_target = max(512, int(cfg.context_window_tail_tokens))
    if tail_target >= cap:
        tail_target = max(512, cap // 2)

    start_id, end_id = _window_bounds(conn)
    if start_id is None or end_id is None:
        return False

    msgs = [m for m in _fetch_range(conn, start_id, end_id) if isinstance(m, dict)]
    if not msgs:
        return False

    total = sum(_single_message_tokens(m) for m in msgs)
    trigger = _window_trigger_total(msgs)
    if trigger <= cap and _suffix_is_valid_chat_completions(msgs):
        return False

    dropped_token = 0
    if trigger > cap:
        while len(msgs) > 1 and not (total <= tail_target and _window_trigger_total(msgs) <= cap):
            total -= _single_message_tokens(msgs[0])
            msgs.pop(0)
            dropped_token += 1

    api_trim = 0
    if with_tool_chain_trim:
        if not (_suffix_is_valid_chat_completions(msgs)
                and _window_trigger_total(msgs) <= cap
                and _tail_assistant_awaits_tool_rows(msgs)):
            api_trim = _left_trim_to_valid_chat_prefix(msgs)
            if api_trim and msgs:
                api_trim = min(api_trim, len(msgs))
                del msgs[:api_trim]
    elif dropped_token > 0 and not _suffix_is_valid_chat_completions(msgs):
        t = _left_trim_to_valid_chat_prefix(msgs)
        if t and msgs:
            del msgs[:min(t, len(msgs))]
            api_trim = t

    if dropped_token == 0 and api_trim == 0:
        return False

    _set_window_bounds(conn, msgs[0]["id"] if msgs else end_id, end_id)

    from agent import core_memory as _cm
    _cm.sync_snapshot_from_core_memory_conn(conn)

    tail_tok = sum(_single_message_tokens(m) for m in msgs)
    if not quiet and (dropped_token or api_trim):
        parts = []
        if dropped_token:
            parts.append(f"over {cap} ref. tok: dropped {dropped_token} msg → ≤{tail_target} (~{tail_tok} tok)")
        if api_trim:
            parts.append(f"tool-chain fix: dropped {api_trim} msg (~{tail_tok} tok)")
        say(f"  [windows] {'; '.join(parts)}")
    return True


# ── Public API ──────────────────────────────────────────────────────────

def load_messages() -> list[dict[str, Any]]:
    with _STORE_LOCK, _db() as conn:
        cur = conn.execute("SELECT id, body FROM consciousness_messages ORDER BY id ASC")
        return [_strip_msg_id(_row_to_message(int(r[0]), str(r[1]))) for r in cur.fetchall()]


def save_messages(messages: list[dict[str, Any]]) -> None:
    with _STORE_LOCK, _db() as conn:
        conn.execute("DELETE FROM consciousness_messages")
        try:
            conn.execute("DELETE FROM sqlite_sequence WHERE name = 'consciousness_messages'")
        except sqlite3.OperationalError:
            pass
        for m in messages:
            if isinstance(m, dict) and m.get("role"):
                conn.execute("INSERT INTO consciousness_messages (body) VALUES (?)",
                             (json.dumps(_normalize_stored_message(m), ensure_ascii=False),))
        _sync_window_cover_all(conn)


def append(text: str, *, role: str = "user", **extra: Any) -> None:
    if not (text or "").strip() and not extra.get("tool_calls"):
        return
    msg: dict[str, Any] = {"role": role, "content": text or ""}
    for k, v in extra.items():
        if v is not None:
            msg[k] = v
    extend_messages([msg])


def extend_messages(msgs: list[dict[str, Any]]) -> None:
    if not msgs:
        return
    with _STORE_LOCK, _db() as conn:
        before_max = conn.execute("SELECT COALESCE(MAX(id), 0) FROM consciousness_messages").fetchone()[0]
        for m in (dict(m) for m in msgs):
            conn.execute("INSERT INTO consciousness_messages (body) VALUES (?)",
                         (json.dumps(_normalize_stored_message(m), ensure_ascii=False),))
        last_id = conn.execute("SELECT MAX(id) FROM consciousness_messages").fetchone()[0] or before_max
        w_start, _ = _window_bounds(conn)
        _set_window_bounds(conn, w_start if w_start is not None else before_max + 1, last_id)
        _maybe_trim_infer_window(conn, quiet=False, with_tool_chain_trim=False)


def perceive() -> list[dict[str, Any]]:
    with _STORE_LOCK, _db() as conn:
        _maybe_trim_infer_window(conn, quiet=True)
        start_id, end_id = _window_bounds(conn)
        if start_id is None or end_id is None:
            return []
        return [_strip_infer_keys(dict(m)) for m in _fetch_range(conn, start_id, end_id)]


def perceive_and_ref_token_len() -> tuple[list[dict[str, Any]], int]:
    msgs = perceive()
    return msgs, sum(_single_message_tokens(m) for m in msgs)


def get_window() -> dict[str, Any]:
    with _STORE_LOCK, _db() as conn:
        _maybe_trim_infer_window(conn, quiet=True)
        start_id, end_id = _window_bounds(conn)
        slice_msgs = [_strip_infer_keys(dict(m)) for m in _fetch_range(conn, start_id, end_id)] \
            if start_id is not None and end_id is not None else []
        return {
            "version": _STATE_VERSION, "agent_db_file": _db_path(),
            "infer_context": {"start_id": start_id, "end_id": end_id},
            "messages": slice_msgs, "message_count": len(slice_msgs),
        }


def compose_infer_messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(m) for m in history]
