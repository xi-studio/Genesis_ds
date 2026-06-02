"""
System prompt — loaded from ``prompt.md`` beside this module.

``infer_system_content()`` fills paths and limits from :class:`Config`.
"""
from __future__ import annotations

import os
from pathlib import Path

from agent.config import Config


def _prompt_rel_path(path: str) -> str:
    if not path:
        return path
    try:
        return os.path.relpath(path, os.getcwd())
    except ValueError:
        return path


def _prompt_template() -> str:
    p = Path(__file__).resolve().parent / "prompt.md"
    return p.read_text(encoding="utf-8")


def infer_system_content() -> str:
    cfg = Config.get()
    cap = max(1, int(cfg.max_exec_source_chars))
    mtr = max(1, int(cfg.max_tool_rounds))
    gap = cfg.self_continue_gap_sec
    sd = cfg.sleep_default_sec
    sm = cfg.sleep_max_sec

    cm = _prompt_rel_path(cfg.agent_db_file)

    return _prompt_template().format(
        workspace_rel=cfg.workspace_rel,
        cm=cm,
        gap=float(gap),
        sd=float(sd),
        sm=float(sm),
        cap=cap,
        mtr=mtr,
    )
