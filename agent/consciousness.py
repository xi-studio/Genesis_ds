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
from typing import Any

from agent.config import Config
from agent.output import say
from agent.tokenizer import count_tokens

_STORE_LOCK = threading.RLock()
_USAGE_ANCHOR_LOCK = threading.Lock()
_STATE_VERSION = 2

# Last successful chat.completions input: (prompt_tokens from usage, sum _single_message_tokens)
_last_usage_anchor: tuple[int, int] | None = None


def record_infer_prompt_usage(messages_for_request: list[dict[str, Any]], usage: Any) -> None:
    """Remember provider ``prompt_tokens`` vs local ref-sum for the same API ``messages`` list."""
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
    ref = 0
    for m in messages_for_request:
        if isinstance(m, dict):
            ref += _single_message_tokens(m)
    if ref <= 0:
        return
    with _USAGE_ANCHOR_LOCK:
        _last_usage_anchor = (pi, ref)


def _window_trigger_total(msgs: list[dict[str, Any]]) -> int:
    """Infer-window ref.tok for cap check; scales by last provider usage when available."""
    window_est = sum(_single_message_tokens(m) for m in msgs)
    with _USAGE_ANCHOR_LOCK:
        anchor = _last_usage_anchor
    if anchor is None:
        return window_est
    api_pt, full_ref = anchor
    if api_pt <= 0 or full_ref <= 0:
        return window_est
    scale = api_pt / float(full_ref)
    return max(0, int(window_est * scale))


def _boot_messages() -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": "System - [Boot] Being awakened.\n\n",
        }
    ]


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _db_path() -> str:
    return os.path.abspath(Config.get().agent_db_file)


def _connect() -> sqlite3.Connection:
    path = _db_path()
    _ensure_parent(path)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS consciousness_messages (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          body TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS agent_state (
          singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
          infer_context_start_id INTEGER,
          infer_context_end_id INTEGER
        );

        INSERT OR IGNORE INTO agent_state (singleton, infer_context_start_id, infer_context_end_id)
        VALUES (1, NULL, NULL);
        """
    )


def _row_to_message(row_id: int, body_raw: str) -> dict[str, Any]:
    try:
        data = json.loads(body_raw)
    except (json.JSONDecodeError, TypeError):
        data = {"role": "user", "content": ""}
    if not isinstance(data, dict):
        data = {"role": "user", "content": ""}
    out = dict(data)
    out["id"] = row_id
    return out


def _fetch_range(conn: sqlite3.Connection, start_id: int, end_id: int) -> list[dict[str, Any]]:
    cur = conn.execute(
        "SELECT id, body FROM consciousness_messages WHERE id >= ? AND id <= ? ORDER BY id ASC",
        (start_id, end_id),
    )
    return [_row_to_message(int(r[0]), str(r[1])) for r in cur.fetchall()]


def _window_bounds(conn: sqlite3.Connection) -> tuple[int | None, int | None]:
    row = conn.execute(
        "SELECT infer_context_start_id, infer_context_end_id FROM agent_state WHERE singleton = 1"
    ).fetchone()
    if not row:
        return None, None
    a, b = row[0], row[1]
    if a is None or b is None:
        return None, None
    try:
        return int(a), int(b)
    except (TypeError, ValueError):
        return None, None


def _set_window_bounds(conn: sqlite3.Connection, start_id: int | None, end_id: int | None) -> None:
    conn.execute(
        "UPDATE agent_state SET infer_context_start_id = ?, infer_context_end_id = ? WHERE singleton = 1",
        (start_id, end_id),
    )


def _sync_window_cover_all(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT MIN(id), MAX(id) FROM consciousness_messages",
    ).fetchone()
    if not row or row[0] is None:
        return
    lo, hi = int(row[0]), int(row[1])
    s, e = _window_bounds(conn)
    if s is None or e is None:
        _set_window_bounds(conn, lo, hi)


def _count_messages(conn: sqlite3.Connection) -> int:
    r = conn.execute("SELECT COUNT(*) FROM consciousness_messages").fetchone()
    return int(r[0]) if r else 0


def _ensure_boot_rows(conn: sqlite3.Connection) -> bool:
    if _count_messages(conn) > 0:
        return False
    for bm in _boot_messages():
        row = _normalize_stored_message(bm)
        conn.execute(
            "INSERT INTO consciousness_messages (body) VALUES (?)",
            (json.dumps(row, ensure_ascii=False),),
        )
    _sync_window_cover_all(conn)
    return True


def _prepare_store(conn: sqlite3.Connection) -> None:
    _init_schema(conn)
    if _count_messages(conn) == 0:
        _ensure_boot_rows(conn)
    _sync_window_cover_all(conn)


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
    if m.get("role") == "tool":
        tid = m.get("tool_call_id")
        if tid:
            n += count_tokens(str(tid))
    return max(n, 1)


def _suffix_is_valid_chat_completions(msgs: list[dict[str, Any]]) -> bool:
    """True if this slice can follow ``system`` in chat.completions (tool_calls ↔ tool rows)."""
    i = 0
    while i < len(msgs):
        m = msgs[i]
        role = m.get("role")
        if role == "tool":
            return False
        if role == "assistant" and m.get("tool_calls"):
            tcs = m["tool_calls"]
            if not isinstance(tcs, list) or not tcs:
                i += 1
                continue
            n_calls = len(tcs)
            req_ids = [
                str(tc["id"])
                for tc in tcs
                if isinstance(tc, dict) and str(tc.get("id") or "").strip()
            ]
            if i + n_calls >= len(msgs):
                return False
            got: list[str] = []
            for k in range(n_calls):
                tm = msgs[i + 1 + k]
                if not isinstance(tm, dict) or tm.get("role") != "tool":
                    return False
                got.append(str(tm.get("tool_call_id") or ""))
            if req_ids and set(req_ids) != set(got):
                return False
            i += 1 + n_calls
            continue
        i += 1
    return True


def _left_trim_to_valid_chat_prefix(msgs: list[dict[str, Any]]) -> int:
    """Smallest number of **whole** leading messages to drop so the remainder is API-valid."""
    for drop in range(len(msgs) + 1):
        if _suffix_is_valid_chat_completions(msgs[drop:]):
            return drop
    return len(msgs)


def _normalize_stored_message(m: dict[str, Any]) -> dict[str, Any]:
    """Normalize a message dict for persistence (flat API-shaped dict)."""
    m = {k: v for k, v in m.items() if k != "id"}
    role = str(m.get("role") or "user")
    content = m.get("content")
    if content is None:
        content = ""
    elif not isinstance(content, str):
        content = str(content)
    out: dict[str, Any] = {"role": role, "content": content}
    if m.get("name") is not None:
        out["name"] = m["name"]
    if m.get("tool_call_id") is not None:
        out["tool_call_id"] = m["tool_call_id"]
    if m.get("tool_calls") is not None:
        out["tool_calls"] = m["tool_calls"]
    tool_calls = out.get("tool_calls")
    for k, v in m.items():
        if k in ("role", "content", "name", "tool_call_id", "tool_calls"):
            continue
        if role == "assistant" and k == "reasoning_content" and not tool_calls:
            continue
        if v is not None:
            out[k] = v
    return out


def _strip_msg_id(m: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in m.items() if k != "id"}


def _strip_infer_keys(m: dict[str, Any]) -> dict[str, Any]:
    """Chat-API payload: strip persisted ``id`` and any leading-underscore keys."""
    return {
        k: v
        for k, v in m.items()
        if k != "id" and not str(k).startswith("_")
    }


def _tail_assistant_awaits_tool_rows(msgs: list[dict[str, Any]]) -> bool:
    """True if the last message is an assistant turn with ``tool_calls`` (tool rows follow next)."""
    if not msgs:
        return False
    last = msgs[-1]
    if last.get("role") != "assistant":
        return False
    tcs = last.get("tool_calls")
    return isinstance(tcs, list) and bool(tcs)


def _maybe_trim_infer_window(
    conn: sqlite3.Connection,
    *,
    quiet: bool,
    with_tool_chain_trim: bool = True,
) -> bool:
    """Advance ``infer_context_start_id`` until the window fits token / API-prefix rules."""
    cfg = Config.get()
    cap = max(1024, int(cfg.context_window_max_tokens))
    tail_target = max(512, int(cfg.context_window_tail_tokens))
    if tail_target >= cap:
        tail_target = max(512, cap // 2)

    start_id, end_id = _window_bounds(conn)
    if start_id is None or end_id is None:
        return False

    msgs = _fetch_range(conn, start_id, end_id)
    msgs = [m for m in msgs if isinstance(m, dict)]
    if not msgs:
        return False

    total = sum(_single_message_tokens(m) for m in msgs)
    trigger = _window_trigger_total(msgs)
    if trigger <= cap and _suffix_is_valid_chat_completions(msgs):
        return False

    dropped_token = 0
    if trigger > cap:
        while len(msgs) > 1:
            if total <= tail_target and _window_trigger_total(msgs) <= cap:
                break
            total -= _single_message_tokens(msgs[0])
            msgs.pop(0)
            dropped_token += 1

    api_trim = 0
    if with_tool_chain_trim:
        prefix_ok = _suffix_is_valid_chat_completions(msgs)
        skip_chain = (
            prefix_ok
            and _window_trigger_total(msgs) <= cap
            and _tail_assistant_awaits_tool_rows(msgs)
        )
        if not skip_chain:
            api_trim = _left_trim_to_valid_chat_prefix(msgs)
            if api_trim and msgs:
                api_trim = min(api_trim, len(msgs))
                if api_trim:
                    del msgs[:api_trim]

    elif dropped_token > 0 and not _suffix_is_valid_chat_completions(msgs):
        t = _left_trim_to_valid_chat_prefix(msgs)
        if t and msgs:
            t = min(t, len(msgs))
            if t:
                del msgs[:t]
                api_trim = t

    if dropped_token == 0 and api_trim == 0:
        return False

    new_start = msgs[0]["id"] if msgs else end_id
    _set_window_bounds(conn, new_start, end_id)

    from agent import core_memory as _core_memory

    _core_memory.sync_snapshot_from_core_memory_conn(conn)

    tail_tok = sum(_single_message_tokens(m) for m in msgs)
    if not quiet:
        parts: list[str] = []
        if dropped_token:
            parts.append(
                f"over {cap} ref. tok (provider-scaled est.): dropped {dropped_token} whole message(s) from head "
                f"→ ≤{tail_target} heuristic target (~{tail_tok} ref. tok left)"
            )
        if api_trim:
            parts.append(
                f"tool-chain fix: dropped {api_trim} whole message(s) from head "
                f"(~{tail_tok} ref. tok left)"
            )
        say(f"  [windows] {'; '.join(parts)}")
    elif dropped_token or api_trim:
        if dropped_token:
            say(
                f"  [windows] infer ref.tok over {cap} (provider-scaled est.): dropped {dropped_token} whole message(s) "
                f"from head → ≤{tail_target} heuristic target (~{tail_tok} ref. tok est. left)"
            )
        elif api_trim:
            say(
                f"  [windows] infer API-prefix repair (still ≤{cap} ref. tok est.): "
                f"dropped {api_trim} whole message(s) from head (~{tail_tok} left)"
            )
    return True


def load_messages() -> list[dict[str, Any]]:
    """Full history from consciousness only (no ``id`` in dicts)."""
    with _STORE_LOCK:
        conn = _connect()
        try:
            _prepare_store(conn)
            cur = conn.execute("SELECT id, body FROM consciousness_messages ORDER BY id ASC")
            rows = [_row_to_message(int(r[0]), str(r[1])) for r in cur.fetchall()]
            conn.commit()
        finally:
            conn.close()
        return [_strip_msg_id(m) for m in rows]


def save_messages(messages: list[dict[str, Any]]) -> None:
    """Replace the full consciousness log and reset the infer window to the full range."""
    with _STORE_LOCK:
        conn = _connect()
        try:
            _prepare_store(conn)
            conn.execute("DELETE FROM consciousness_messages")
            try:
                conn.execute("DELETE FROM sqlite_sequence WHERE name = 'consciousness_messages'")
            except sqlite3.OperationalError:
                pass
            for m in messages:
                if not isinstance(m, dict) or not m.get("role"):
                    continue
                row = _normalize_stored_message(m)
                conn.execute(
                    "INSERT INTO consciousness_messages (body) VALUES (?)",
                    (json.dumps(row, ensure_ascii=False),),
                )
            _sync_window_cover_all(conn)
            conn.commit()
        finally:
            conn.close()


def append(
    text: str,
    *,
    role: str = "user",
    **extra: Any,
) -> None:
    """Append one chat message (and optional extra keys, e.g. tool_call_id)."""
    if not (text or "").strip() and not extra.get("tool_calls"):
        return
    msg: dict[str, Any] = {"role": role, "content": text or ""}
    for k, v in extra.items():
        if v is not None:
            msg[k] = v
    extend_messages([msg])


def extend_messages(msgs: list[dict[str, Any]]) -> None:
    """Append turns to consciousness; extend infer window ``end_id``; trim window if over cap."""
    if not msgs:
        return
    new = [dict(m) for m in msgs]
    with _STORE_LOCK:
        conn = _connect()
        try:
            _prepare_store(conn)
            before_max_row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM consciousness_messages").fetchone()
            before_max = int(before_max_row[0]) if before_max_row else 0

            for m in new:
                base = _normalize_stored_message(m)
                conn.execute(
                    "INSERT INTO consciousness_messages (body) VALUES (?)",
                    (json.dumps(base, ensure_ascii=False),),
                )

            last_row = conn.execute("SELECT MAX(id) FROM consciousness_messages").fetchone()
            last_id = int(last_row[0]) if last_row and last_row[0] is not None else before_max

            w_start, w_end = _window_bounds(conn)
            first_new_row = conn.execute(
                "SELECT MIN(id) FROM consciousness_messages WHERE id > ?",
                (before_max,),
            ).fetchone()
            first_new_id = int(first_new_row[0]) if first_new_row and first_new_row[0] is not None else last_id

            if w_start is None or w_end is None:
                _set_window_bounds(conn, first_new_id, last_id)
            else:
                _set_window_bounds(conn, w_start, last_id)

            _maybe_trim_infer_window(conn, quiet=False, with_tool_chain_trim=False)
            conn.commit()
        finally:
            conn.close()


def perceive() -> list[dict[str, Any]]:
    """Messages sent to infer — slice ``[infer_context_start_id, infer_context_end_id]`` over consciousness."""
    with _STORE_LOCK:
        conn = _connect()
        try:
            _prepare_store(conn)
            _maybe_trim_infer_window(conn, quiet=True)
            start_id, end_id = _window_bounds(conn)
            if start_id is None or end_id is None:
                conn.commit()
                return []
            msgs = _fetch_range(conn, start_id, end_id)
            conn.commit()
        finally:
            conn.close()
        return [_strip_infer_keys(dict(m)) for m in msgs]


def perceive_and_ref_token_len() -> tuple[list[dict[str, Any]], int]:
    msgs = perceive()
    total = sum(_single_message_tokens(m) for m in msgs)
    return msgs, total


def get_window() -> dict[str, Any]:
    """Infer rolling context + bounds for hosts / patches."""
    with _STORE_LOCK:
        conn = _connect()
        try:
            _prepare_store(conn)
            _maybe_trim_infer_window(conn, quiet=True)
            start_id, end_id = _window_bounds(conn)
            slice_msgs: list[dict[str, Any]] = []
            if start_id is not None and end_id is not None:
                slice_msgs = [_strip_infer_keys(dict(m)) for m in _fetch_range(conn, start_id, end_id)]
            conn.commit()
        finally:
            conn.close()
        return {
            "version": _STATE_VERSION,
            "agent_db_file": _db_path(),
            "infer_context": {"start_id": start_id, "end_id": end_id},
            "messages": slice_msgs,
            "message_count": len(slice_msgs),
        }


def compose_infer_messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shallow copy of window history for the chat API (system is added in ``infer``)."""
    return [dict(m) for m in history]
