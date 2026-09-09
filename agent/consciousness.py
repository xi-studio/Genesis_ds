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

_last_usage_anchor: tuple[int, int, int, int] | None = None  # (api_pt, full_ref, window_ref, tools_ref)


# ── DB helpers ──────────────────────────────────────────────────────────

def _db_path() -> str:
    return os.path.abspath(Config.get().agent_db_file)

@contextmanager
def _db():
    path = _db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    # WAL: readers don't block writers and vice-versa — needed because
    # consciousness and core_memory hold separate locks but write the same file.
    # busy_timeout: wait-and-retry on contention instead of raising "database is locked".
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
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

def _tools_token_estimate() -> int:
    """Heuristic for tool-schema tokens included in API prompt but not in messages."""
    tools = Config.get().tool_definitions
    if not tools:
        return 0
    try:
        return count_tokens(json.dumps(tools, ensure_ascii=False))
    except (TypeError, ValueError):
        return 0


def record_infer_prompt_usage(
    messages_for_request: list[dict[str, Any]],
    usage: Any,
    *,
    infer_window: list[dict[str, Any]] | None = None,
) -> None:
    """Record API prompt usage for scaling window estimates during trim.

    ``infer_window`` is the consciousness slice only (no system / core-memory user).
    Fixed request overhead (system, injected core memory, tool schemas) is kept
    separate so trimming the window does not over-shrink it.
    """
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
    full_ref = sum(_single_message_tokens(m) for m in messages_for_request if isinstance(m, dict))
    if full_ref <= 0:
        return
    if infer_window is None:
        window_ref = full_ref
    else:
        window_ref = sum(_single_message_tokens(m) for m in infer_window if isinstance(m, dict))
        window_ref = max(0, min(window_ref, full_ref))
    if window_ref <= 0:
        return
    tools_ref = _tools_token_estimate()
    with _USAGE_ANCHOR_LOCK:
        _last_usage_anchor = (pi, full_ref, window_ref, tools_ref)


def _window_trigger_total(msgs: list[dict[str, Any]]) -> int:
    window_est = sum(_single_message_tokens(m) for m in msgs)
    with _USAGE_ANCHOR_LOCK:
        anchor = _last_usage_anchor
    if anchor is None:
        return window_est
    api_pt, full_ref, window_ref_anchor, tools_ref_anchor = anchor
    if api_pt <= 0 or full_ref <= 0 or window_ref_anchor <= 0:
        return window_est
    tools_ref = _tools_token_estimate() or tools_ref_anchor
    full_ref_adj = full_ref + max(0, tools_ref)
    non_window_ref = max(0, full_ref - window_ref_anchor) + max(0, tools_ref)
    non_window_api = int(api_pt * non_window_ref / full_ref_adj) if full_ref_adj > 0 else 0
    non_window_api = max(0, min(api_pt, non_window_api))
    window_api = max(0, api_pt - non_window_api)
    window_scale = window_api / window_ref_anchor
    return max(0, int(non_window_api + window_est * window_scale))


def _compute_message_tokens(m: dict[str, Any]) -> int:
    n = 0
    c = m.get("content")
    if isinstance(c, str) and c:
        n += count_tokens(c)
    rc = m.get("reasoning_content")
    if m.get("role") == "assistant" and isinstance(rc, str) and rc:
        n += count_tokens(rc)
    if m.get("tool_calls"):
        try:
            n += count_tokens(json.dumps(m["tool_calls"], ensure_ascii=False))
        except (TypeError, ValueError):
            n += 32
    if m.get("role") == "tool" and (tid := m.get("tool_call_id")):
        n += count_tokens(str(tid))
    return max(n, 1)


def _single_message_tokens(m: dict[str, Any]) -> int:
    """Token size of one message. Uses the cached ``_tokens`` field when present
    (messages are immutable once stored, so the count never changes), otherwise
    computes it on the fly. Caching is populated at write time in
    :func:`_normalize_stored_message`."""
    cached = m.get("_tokens")
    if isinstance(cached, int) and cached > 0:
        return cached
    return _compute_message_tokens(m)



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

    # Cache the token size once at write time (messages are immutable in the
    # append-only log, so this never needs recomputing during trim). Stored under
    # the ``_tokens`` key, which _strip_infer_keys() removes before the API sees it.
    out.pop("_tokens", None)
    out["_tokens"] = _compute_message_tokens(out)
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
    """Drop messages from the left until the suffix is valid for chat.completions.

    Skips complete atomic blocks (assistant+tool_calls+tool_results) instead of
    individual messages, preserving maximum context.  When a broken tool_calls
    block is encountered (orphaned tool results, or missing tool results), the
    entire broken block is dropped in one step rather than one message at a time.
    """
    drop = 0
    while drop < len(msgs):
        if _suffix_is_valid_chat_completions(msgs[drop:]):
            return drop
        # Skip the next atomic block to reach a potentially valid prefix.
        block_len = _atomic_block_len(msgs[drop:])
        if block_len <= 0:
            block_len = 1
        drop += block_len
    return len(msgs)


def _atomic_block_len(msgs: list[dict[str, Any]]) -> int:
    """Left-most trim unit: one message, or assistant + following tool rows.

    For assistant(tool_calls), counts the actual number of consecutive ``tool``
    messages that follow (capped at ``len(tool_calls)``), rather than blindly
    assuming all ``len(tool_calls)`` results are present.  This prevents
    miscalculation when tool results are missing or interleaved with other roles.
    """
    if not msgs:
        return 0
    m = msgs[0]
    if m.get("role") == "tool":
        return 1
    if m.get("role") == "assistant" and (tcs := m.get("tool_calls")):
        if isinstance(tcs, list) and tcs:
            expected = len(tcs)
            actual = 0
            for k in range(1, min(expected + 1, len(msgs))):
                if msgs[k].get("role") == "tool":
                    actual += 1
                else:
                    break
            return min(len(msgs), 1 + actual)
    return 1


def _pop_left_atomic(msgs: list[dict[str, Any]]) -> tuple[int, int]:
    """Drop one atomic unit from the left; return (tokens_removed, messages_removed)."""
    n = _atomic_block_len(msgs)
    if n <= 0:
        return 0, 0
    block = msgs[:n]
    del msgs[:n]
    return sum(_single_message_tokens(m) for m in block), n


def _atomic_block_shrink(msgs: list[dict[str, Any]], cut: int) -> int:
    """Shrink a left-cut index back to the previous atomic-block boundary."""
    if cut <= 1:
        return 1
    i = 0
    while True:
        n = _atomic_block_len(msgs[i:])
        if n <= 0 or i + n >= cut:
            break
        i += n
    return max(1, i)


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

    dropped_msgs = 0
    if trigger > cap:
        while len(msgs) > 1 and total > tail_target:
            n = _atomic_block_len(msgs)
            if n <= 0:
                break
            block_tok = sum(_single_message_tokens(m) for m in msgs[:n])
            if total - block_tok < tail_target:
                break  # tail_target is a floor: keep >= tail_target, stop before overshooting
            removed_tok, removed_n = _pop_left_atomic(msgs)
            if removed_n <= 0:
                break
            total -= removed_tok
            dropped_msgs += removed_n
        while len(msgs) > 1 and _window_trigger_total(msgs) > cap:
            n = _atomic_block_len(msgs)
            if n <= 0:
                break
            block_tok = sum(_single_message_tokens(m) for m in msgs[:n])
            if total - block_tok < tail_target:
                break  # floor: never collapse below tail_target (bug: few-k windows)
            removed_tok, removed_n = _pop_left_atomic(msgs)
            if removed_n <= 0:
                break
            total -= removed_tok
            dropped_msgs += removed_n

    api_trim = 0
    needs_prefix_fix = (
        not _suffix_is_valid_chat_completions(msgs)
        and not _tail_assistant_awaits_tool_rows(msgs)
    )
    if needs_prefix_fix and (with_tool_chain_trim or dropped_msgs > 0):
        api_trim = _left_trim_to_valid_chat_prefix(msgs)
        if api_trim and msgs:
            api_trim = min(api_trim, len(msgs))
            # Floor guard: don't let prefix repair collapse the window.
            while api_trim > 0:
                remain_tok = sum(_single_message_tokens(m) for m in msgs[api_trim:])
                if remain_tok >= tail_target or api_trim <= 1:
                    break
                api_trim = max(1, _atomic_block_shrink(msgs, api_trim))
            del msgs[:api_trim]

    if dropped_msgs == 0 and api_trim == 0:
        return False

    if not msgs:
        # Degenerate: nothing left to point at — keep previous bounds entirely.
        say("  [windows] trim would empty the window; bounds unchanged")
        return False

    _set_window_bounds(conn, msgs[0]["id"], end_id)

    from agent import core_memory as _cm
    _cm.sync_snapshot_from_core_memory_conn(conn)

    tail_tok = sum(_single_message_tokens(m) for m in msgs)
    if not quiet and (dropped_msgs or api_trim):
        parts = []
        if dropped_msgs:
            parts.append(
                f"over {cap} est. tok: dropped {dropped_msgs} msg → ≥{tail_target} (floor) (~{tail_tok} ref.tok)"
            )
        if api_trim:
            parts.append(f"tool-chain fix: dropped {api_trim} msg (~{tail_tok} tok)")
            if tail_tok < tail_target:
                parts.append(f"⚠ below tail floor {tail_target}")
        say(f"  [windows] {'; '.join(parts)}")
    return True


# ── Public API ──────────────────────────────────────────────────────────

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
        if w_start is None:
            # NULL bounds mid-run (crash/cancel/external reset): recover the FULL
            # window, not a 1-message one. Trim will converge it to tail_target.
            w_start = conn.execute(
                "SELECT COALESCE(MIN(id), ?) FROM consciousness_messages",
                (before_max + 1,)).fetchone()[0]
            say(f"  [windows] recovered NULL bounds → full window from #{w_start}")
        _set_window_bounds(conn, w_start, last_id)
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


def recent_messages(limit: int = 100) -> list[dict[str, Any]]:
    """Most recent ``limit`` rows of the consciousness log (oldest→newest).

    Returns chat-shaped dicts with ``id``, without internal ``_tokens`` —
    safe for display (e.g. the web history endpoint).
    """
    n = max(1, min(int(limit), 1000))
    with _STORE_LOCK, _db() as conn:
        rows = conn.execute(
            "SELECT id, body FROM consciousness_messages ORDER BY id DESC LIMIT ?",
            (n,),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in reversed(rows):
        m = _row_to_message(int(r[0]), str(r[1]))
        out.append({k: v for k, v in m.items() if not str(k).startswith("_")})
    return out
