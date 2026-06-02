"""
Core memory — source table ``core_memory_entries`` and injected ``core_memory_snapshot``
in ``agent_db_file``.

Each LLM request injects the stable snapshot. The source table is synced into the
snapshot at infer-window trim time.

**Passive retention** (applied during snapshot sync): entries are
``priority`` **P1** (keep forever), **P2** (drop if ``updated_at`` older than 7 days), **P3** (drop
if older than 24 hours). Missing ``priority`` in legacy rows is treated as **P1**. Pruned rows are
removed from ``core_memory_entries`` as well.

Mutations (**tools**): ``core_memory_append`` / ``core_memory_update`` touch ``core_memory_entries``.
There is no agent **delete** tool — demote to **P3** (and optional short ``content``) for passive TTL.

Schema per source entry::

    {"id": "cm_1", "updated_at": "...", "content": "...", "priority": "P1"|"P2"|"P3"}

Snapshot row (singleton)::

    content= "## Core Memory\\n...", synced_at=ISO, passive_dropped=int
"""
from __future__ import annotations

import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

from agent.config import Config
from agent.tokenizer import count_tokens

_LOCK = threading.RLock()

P1, P2, P3 = "P1", "P2", "P3"
_TTL = {P2: timedelta(days=7), P3: timedelta(hours=24)}
_HEADER = "## Core Memory\n\n---\n\n"


# ── DB helpers ──────────────────────────────────────────────────────────

def _db_path() -> str:
    return os.path.abspath(os.path.expanduser(Config.get().agent_db_file))

@contextmanager
def _db():
    """Context manager: open DB, init schema, yield conn, commit, close."""
    path = _db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS core_memory_entries (
          id TEXT PRIMARY KEY, updated_at TEXT NOT NULL,
          content TEXT NOT NULL, priority TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS core_memory_snapshot (
          singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
          content TEXT NOT NULL, synced_at TEXT NOT NULL,
          passive_dropped INTEGER NOT NULL DEFAULT 0);
        INSERT OR IGNORE INTO core_memory_snapshot (singleton, content, synced_at, passive_dropped)
        VALUES (1, '', '', 0);
    """)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ── Priority helpers ────────────────────────────────────────────────────

_PRIORITIES = {P1, P2, P3}

def _valid_priority(raw: Any) -> str | None:
    """Return canonical priority string if valid, else None."""
    p = str(raw or "").strip().upper()
    return p if p in _PRIORITIES else None

def _norm_priority(raw: Any) -> str:
    """Normalize to valid priority; legacy/missing defaults to P1."""
    return _valid_priority(raw) or P1


# ── Time helpers ────────────────────────────────────────────────────────

def _parse_dt(ts: str) -> datetime | None:
    s = (ts or "").strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None

def _sort_key(updated_at: str) -> float:
    dt = _parse_dt(updated_at)
    return dt.timestamp() if dt else 0.0

def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# ── Entry I/O ───────────────────────────────────────────────────────────

def _next_id(conn: sqlite3.Connection) -> str:
    best = 0
    for (sid,) in conn.execute("SELECT id FROM core_memory_entries").fetchall():
        m = re.match(r"^cm_(\d+)$", str(sid))
        if m:
            best = max(best, int(m.group(1)))
    return f"cm_{best + 1}"

def _canonical_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(item["id"]).strip(),
        "updated_at": str(item.get("updated_at") or "").strip(),
        "content": str(item.get("content") or ""),
        "priority": _norm_priority(item.get("priority")),
    }

def _load_entries(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for eid, ua, content, prio in conn.execute(
        "SELECT id, updated_at, content, priority FROM core_memory_entries"
    ).fetchall():
        eid_s = str(eid or "").strip()
        if eid_s and isinstance(content, str):
            rows.append(_canonical_row({"id": eid_s, "updated_at": ua, "content": content, "priority": prio}))
    rows.sort(key=lambda r: _sort_key(str(r.get("updated_at") or "")))
    return rows


# ── Passive retention ───────────────────────────────────────────────────

def _passive_keep(row: dict[str, Any], now_utc: datetime) -> bool:
    prio = _norm_priority(row.get("priority"))
    if prio == P1:
        return True
    dt = _parse_dt(str(row.get("updated_at") or ""))
    return dt is None or (now_utc - dt) <= _TTL.get(prio, timedelta.max)

def _passive_prune(entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    now_utc = datetime.now(timezone.utc)
    kept, dropped = [], []
    for r in entries:
        if _passive_keep(r, now_utc):
            kept.append(r)
        elif eid := str(r.get("id") or "").strip():
            dropped.append(eid)
    kept.sort(key=lambda r: _sort_key(str(r.get("updated_at") or "")))
    return kept, dropped


# ── Markdown rendering ──────────────────────────────────────────────────

def _render_entries_to_markdown(entries: list[dict[str, Any]]) -> str:
    cfg = Config.get()
    max_tok = max(128, int(cfg.core_memory_max_tokens))

    by_prio = {p: [r for r in entries if r.get("priority") == p] for p in (P1, P2, P3)}

    def _short_ts(raw: str) -> str:
        s = (raw or "").strip()
        return s[:16] if len(s) >= 16 else s or "(no time)"

    def _build(prio_rows: dict) -> list[str]:
        parts: list[str] = []
        for p in (P1, P2, P3):
            if not prio_rows[p]:
                continue
            parts.append(f"### {p}")
            for r in prio_rows[p]:
                if body := str(r.get("content") or "").strip():
                    parts.append(f"- **{_short_ts(str(r.get('updated_at') or ''))}** — {body}")
        return parts

    parts = _build(by_prio)
    text = _HEADER + ("\n".join(parts) if parts else "")

    # Trim by token count: drop lowest-priority entries until under limit
    while count_tokens(text) > max_tok and any(by_prio.values()):
        for p in (P3, P2, P1):
            if by_prio[p]:
                by_prio[p].pop()
                break
        parts = _build(by_prio)
        text = _HEADER + ("\n".join(parts) if parts else "")
    return text


# ── Public mutation API ─────────────────────────────────────────────────

def disk_append(content: str, priority: str | None = None) -> dict[str, Any]:
    text = (content or "").strip()
    if not text:
        return {"ok": False, "error": "content is empty"}
    prio = priority and _valid_priority(priority)
    if priority is not None and priority.strip() and prio is None:
        return {"ok": False, "error": "priority must be P1, P2, or P3"}
    prio = prio or P3
    now = _now_iso()
    with _LOCK, _db() as conn:
        row = {"id": _next_id(conn), "updated_at": now, "content": text, "priority": prio}
        cr = _canonical_row(row)
        conn.execute("INSERT INTO core_memory_entries (id, updated_at, content, priority) VALUES (?,?,?,?)",
                     (cr["id"], cr["updated_at"], cr["content"], cr["priority"]))
    return {"ok": True, "entry": row}

def disk_update(entry_id: str, content: str, priority: str | None = None) -> dict[str, Any]:
    eid, text = str(entry_id or "").strip(), (content or "").strip()
    if not eid:
        return {"ok": False, "error": "missing id"}
    if not text:
        return {"ok": False, "error": "content is empty"}
    now = _now_iso()
    new_prio = priority and str(priority).strip() and _valid_priority(priority)
    if priority is not None and str(priority).strip() and not new_prio:
        return {"ok": False, "error": "priority must be P1, P2, or P3"}
    with _LOCK, _db() as conn:
        if not conn.execute("SELECT id FROM core_memory_entries WHERE id = ?", (eid,)).fetchone():
            return {"ok": False, "error": f"unknown id {eid!r}"}
        if new_prio:
            conn.execute("UPDATE core_memory_entries SET content=?, updated_at=?, priority=? WHERE id=?",
                         (text, now, new_prio, eid))
        else:
            conn.execute("UPDATE core_memory_entries SET content=?, updated_at=? WHERE id=?",
                         (text, now, eid))
    out: dict[str, Any] = {"ok": True, "id": eid, "updated_at": now}
    if new_prio:
        out["priority"] = new_prio
    return out

def disk_delete(entry_id: str) -> dict[str, Any]:
    eid = str(entry_id or "").strip()
    if not eid:
        return {"ok": False, "error": "missing id"}
    with _LOCK, _db() as conn:
        cur = conn.execute("DELETE FROM core_memory_entries WHERE id = ?", (eid,))
        if cur.rowcount == 0:
            return {"ok": False, "error": f"unknown id {eid!r}"}
    return {"ok": True, "id": eid}


# ── Snapshot management ─────────────────────────────────────────────────

def user_message_dict() -> dict[str, str]:
    return {"role": "user", "content": render_for_prompt()}

def _write_snapshot(conn: sqlite3.Connection, content: str, passive_dropped: int) -> None:
    conn.execute("""
        INSERT INTO core_memory_snapshot (singleton, content, synced_at, passive_dropped)
        VALUES (1, ?, ?, ?)
        ON CONFLICT(singleton) DO UPDATE SET
          content=excluded.content, synced_at=excluded.synced_at, passive_dropped=excluded.passive_dropped
    """, (content, _now_iso(), passive_dropped))

def sync_snapshot_from_core_memory_conn(conn: sqlite3.Connection) -> None:
    """Sync source entries into snapshot using the caller's SQLite transaction."""
    with _LOCK:
        entries = _load_entries(conn)
        pruned, dropped_ids = _passive_prune(entries)
        if dropped_ids:
            conn.execute(f"DELETE FROM core_memory_entries WHERE id IN ({','.join('?' for _ in dropped_ids)})",
                         tuple(dropped_ids))
        _write_snapshot(conn, _render_entries_to_markdown(pruned), len(dropped_ids))

def sync_snapshot_from_core_memory() -> None:
    with _LOCK, _db() as conn:
        sync_snapshot_from_core_memory_conn(conn)

def render_for_prompt() -> str:
    with _LOCK, _db() as conn:
        row = conn.execute("SELECT content FROM core_memory_snapshot WHERE singleton = 1").fetchone()
        return row[0] if row and isinstance(row[0], str) and row[0].strip() else _HEADER
