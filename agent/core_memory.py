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
from datetime import datetime, timedelta, timezone
from typing import Any

from agent.config import Config
from agent.tokenizer import count_tokens

_LOCK = threading.RLock()

P1 = "P1"
P2 = "P2"
P3 = "P3"
_TTL_P2 = timedelta(days=7)
_TTL_P3 = timedelta(hours=24)

_HEADER = "## Core Memory\n\n---\n\n"


def _db_path() -> str:
    return os.path.abspath(os.path.expanduser(Config.get().agent_db_file))


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _connect() -> sqlite3.Connection:
    path = _db_path()
    _ensure_parent(path)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS core_memory_entries (
          id TEXT PRIMARY KEY,
          updated_at TEXT NOT NULL,
          content TEXT NOT NULL,
          priority TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS core_memory_snapshot (
          singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
          content TEXT NOT NULL,
          synced_at TEXT NOT NULL,
          passive_dropped INTEGER NOT NULL DEFAULT 0
        );

        INSERT OR IGNORE INTO core_memory_snapshot (singleton, content, synced_at, passive_dropped)
        VALUES (1, '', '', 0);
        """
    )


def _next_id_conn(conn: sqlite3.Connection) -> str:
    best = 0
    cur = conn.execute("SELECT id FROM core_memory_entries")
    for (sid,) in cur.fetchall():
        m = re.match(r"^cm_(\d+)$", str(sid))
        if m:
            best = max(best, int(m.group(1)))
    return f"cm_{best + 1}"


def _normalize_priority(raw: Any) -> str:
    """Legacy / missing ``priority`` → P1 (never passively deleted)."""
    p = str(raw or "").strip().upper()
    if p in (P1, P2, P3):
        return p
    return P1


def _parse_priority_arg(raw: Any, *, default: str) -> str | None:
    """For tools: explicit invalid value → None (error). Absent/empty → ``default``."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    p = str(raw).strip().upper()
    if p in (P1, P2, P3):
        return p
    return None


def _parse_updated_dt(ts: str) -> datetime | None:
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


def _parse_sort_key(updated_at: str) -> float:
    s = (updated_at or "").strip()
    if not s:
        return 0.0
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _canonical_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(item["id"]).strip(),
        "updated_at": str(item.get("updated_at") or "").strip(),
        "content": str(item.get("content") or ""),
        "priority": _normalize_priority(item.get("priority")),
    }


def _passive_keep(row: dict[str, Any], *, now_utc: datetime) -> bool:
    prio = _normalize_priority(row.get("priority"))
    if prio == P1:
        return True
    dt = _parse_updated_dt(str(row.get("updated_at") or ""))
    if dt is None:
        return True
    age = now_utc - dt
    if prio == P2:
        return age <= _TTL_P2
    if prio == P3:
        return age <= _TTL_P3
    return True


def _passive_prune(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Split expired P2/P3 from kept rows; return kept rows and expired ids."""
    now_utc = datetime.now(timezone.utc)
    kept: list[dict[str, Any]] = []
    dropped_ids: list[str] = []
    for row in rows:
        if _passive_keep(row, now_utc=now_utc):
            kept.append(row)
            continue
        eid = str(row.get("id") or "").strip()
        if eid:
            dropped_ids.append(eid)
    canonical = [_canonical_row(r) for r in kept]
    canonical.sort(key=lambda r: _parse_sort_key(str(r.get("updated_at") or "")))
    return canonical, dropped_ids


def _load_primary_entries_conn(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    cur = conn.execute(
        "SELECT id, updated_at, content, priority FROM core_memory_entries"
    )
    cleaned: list[dict[str, Any]] = []
    for eid, ua, content, prio in cur.fetchall():
        eid_s = str(eid or "").strip()
        if not eid_s or not isinstance(content, str):
            continue
        cleaned.append(
            _canonical_row(
                {"id": eid_s, "updated_at": ua, "content": content, "priority": prio}
            )
        )
    cleaned.sort(key=lambda r: _parse_sort_key(str(r.get("updated_at") or "")))
    return cleaned


def _render_entries_to_markdown(entries: list[dict[str, Any]]) -> str:
    cfg = Config.get()
    max_tok = max(128, int(cfg.core_memory_max_tokens))
    rows_by_priority = {
        prio: sorted(
            [r for r in entries if str(r.get("priority") or P1) == prio],
            key=lambda r: _parse_sort_key(str(r.get("updated_at") or "")),
        )
        for prio in (P1, P2, P3)
    }

    def _short_ts(raw: str) -> str:
        s = (raw or "").strip()
        if not s:
            return "(no time)"
        return s[:16] if len(s) >= 16 else s

    def _build_parts() -> list[str]:
        rendered: list[str] = []
        for prio in (P1, P2, P3):
            rows = rows_by_priority[prio]
            if not rows:
                continue
            rendered.append(f"### {prio}")
            for row in rows:
                body = str(row.get("content") or "").strip()
                if not body:
                    continue
                ts = _short_ts(str(row.get("updated_at") or ""))
                rendered.append(f"- **{ts}** — {body}")
        return rendered

    parts = _build_parts()
    text = _HEADER + ("\n".join(parts) if parts else "")
    while count_tokens(text) > max_tok and any(rows_by_priority.values()):
        for prio in (P3, P2, P1):
            if rows_by_priority[prio]:
                rows_by_priority[prio].pop()
                break
        parts = _build_parts()
        text = _HEADER + ("\n".join(parts) if parts else "")
    return text


def disk_append(content: str, priority: str | None = None) -> dict[str, Any]:
    """Append one entry (default priority P3)."""
    text = (content or "").strip()
    if not text:
        return {"ok": False, "error": "content is empty"}
    prio = _parse_priority_arg(priority, default=P3)
    if prio is None:
        return {"ok": False, "error": "priority must be P1, P2, or P3"}
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    with _LOCK:
        conn = _connect()
        try:
            _init_schema(conn)
            row = {"id": _next_id_conn(conn), "updated_at": now, "content": text, "priority": prio}
            cr = _canonical_row(row)
            conn.execute(
                "INSERT INTO core_memory_entries (id, updated_at, content, priority) VALUES (?,?,?,?)",
                (cr["id"], cr["updated_at"], cr["content"], cr["priority"]),
            )
            conn.commit()
        finally:
            conn.close()
    return {"ok": True, "entry": row}


def disk_update(
    entry_id: str, content: str, priority: str | None = None
) -> dict[str, Any]:
    """Update content; optional ``priority`` to change tier."""
    eid = str(entry_id or "").strip()
    text = (content or "").strip()
    if not eid:
        return {"ok": False, "error": "missing id"}
    if not text:
        return {"ok": False, "error": "content is empty"}
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    new_prio: str | None = None
    if priority is not None and str(priority).strip():
        new_prio = _parse_priority_arg(priority, default=P1)
        if new_prio is None:
            return {"ok": False, "error": "priority must be P1, P2, or P3"}
    with _LOCK:
        conn = _connect()
        try:
            _init_schema(conn)
            cur = conn.execute(
                "SELECT id FROM core_memory_entries WHERE id = ?", (eid,)
            ).fetchone()
            if not cur:
                return {"ok": False, "error": f"unknown id {eid!r}"}
            if new_prio is not None:
                conn.execute(
                    "UPDATE core_memory_entries SET content = ?, updated_at = ?, priority = ? WHERE id = ?",
                    (text, now, new_prio, eid),
                )
            else:
                conn.execute(
                    "UPDATE core_memory_entries SET content = ?, updated_at = ? WHERE id = ?",
                    (text, now, eid),
                )
            conn.commit()
        finally:
            conn.close()
    out: dict[str, Any] = {"ok": True, "id": eid, "updated_at": now}
    if new_prio is not None:
        out["priority"] = new_prio
    return out


def disk_delete(entry_id: str) -> dict[str, Any]:
    """Delete one entry by id (host/admin only; agents use ``update`` + P3 instead)."""
    eid = str(entry_id or "").strip()
    if not eid:
        return {"ok": False, "error": "missing id"}
    with _LOCK:
        conn = _connect()
        try:
            _init_schema(conn)
            cur = conn.execute("DELETE FROM core_memory_entries WHERE id = ?", (eid,))
            if cur.rowcount == 0:
                return {"ok": False, "error": f"unknown id {eid!r}"}
            conn.commit()
        finally:
            conn.close()
    return {"ok": True, "id": eid}


def user_message_dict() -> dict[str, str]:
    """Chat ``user`` message rendered directly from core memory entries."""
    return {"role": "user", "content": render_for_prompt()}


def _delete_expired_conn(conn: sqlite3.Connection, entry_ids: list[str]) -> None:
    if not entry_ids:
        return
    placeholders = ",".join("?" for _ in entry_ids)
    conn.execute(
        f"DELETE FROM core_memory_entries WHERE id IN ({placeholders})",
        tuple(entry_ids),
    )


def _write_snapshot_conn(conn: sqlite3.Connection, content: str, passive_dropped: int) -> None:
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    conn.execute(
        """
        INSERT INTO core_memory_snapshot (singleton, content, synced_at, passive_dropped)
        VALUES (1, ?, ?, ?)
        ON CONFLICT(singleton) DO UPDATE SET
          content = excluded.content,
          synced_at = excluded.synced_at,
          passive_dropped = excluded.passive_dropped
        """,
        (content, now, passive_dropped),
    )


def sync_snapshot_from_core_memory_conn(conn: sqlite3.Connection) -> None:
    """Sync source entries into snapshot using the caller's SQLite transaction."""
    with _LOCK:
        _init_schema(conn)
        raw = _load_primary_entries_conn(conn)
        pruned, dropped_ids = _passive_prune(raw)
        if dropped_ids:
            _delete_expired_conn(conn, dropped_ids)
        _write_snapshot_conn(conn, _render_entries_to_markdown(pruned), len(dropped_ids))


def sync_snapshot_from_core_memory() -> None:
    """Host/admin helper: sync snapshot on a fresh connection and commit."""
    with _LOCK:
        conn = _connect()
        try:
            sync_snapshot_from_core_memory_conn(conn)
            conn.commit()
        finally:
            conn.close()


def render_for_prompt() -> str:
    """Markdown body read from the stable snapshot."""
    with _LOCK:
        conn = _connect()
        try:
            _init_schema(conn)
            row = conn.execute(
                "SELECT content FROM core_memory_snapshot WHERE singleton = 1"
            ).fetchone()
            if not row or not isinstance(row[0], str) or not row[0].strip():
                return _HEADER
            return row[0]
        finally:
            conn.close()
