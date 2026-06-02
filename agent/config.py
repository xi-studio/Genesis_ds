"""
Configuration loader and runtime settings for MolAgent core loop.

All module-level state lives here. Other modules import ``config`` and read
attributes from the singleton ``Config.get()``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_path() -> str:
    env = os.environ.get("CORE_LOOP_CONFIG", "").strip()
    if env:
        return env
    agent_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(agent_dir), "config.json")


def _resolve_api_key(cfg: dict) -> str:
    for env_name in ("OPENAI_API_KEY", "DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY"):
        v = (os.environ.get(env_name) or "").strip()
        if v:
            return v
    return str(cfg.get("api_key") or "").strip()


def _openai_base_url(url: str) -> str:
    ep = url.strip().rstrip("/")
    if ep.endswith("/chat/completions"):
        return ep[: -len("/chat/completions")]
    return ep


def _parse_host_port(raw: str | None) -> tuple[str, int]:
    if not raw or not str(raw).strip():
        return "127.0.0.1", 0
    s = str(raw).strip()
    if ":" in s:
        host, _, port_s = s.rpartition(":")
        host = host.strip() or "127.0.0.1"
        try:
            p = int(port_s)
            return (host, p) if p > 0 else ("127.0.0.1", 0)
        except ValueError:
            return "127.0.0.1", 0
    try:
        p = int(s)
        return ("127.0.0.1", p) if p > 0 else ("127.0.0.1", 0)
    except ValueError:
        return "127.0.0.1", 0


def _first(cfg: dict, *keys: str) -> Any:
    """Return value of first key present in cfg, or None."""
    for k in keys:
        v = cfg.get(k)
        if v is not None:
            return v
    return None


def _env_str(key: str) -> str | None:
    v = os.environ.get(key, "").strip()
    return v or None


def _env_num(key: str) -> float | None:
    v = _env_str(key)
    if v is not None:
        try:
            return float(v)
        except ValueError:
            pass
    return None


def _env_bool(key: str) -> bool | None:
    v = _env_str(key)
    if v is None:
        return None
    vl = v.lower()
    if vl in ("1", "true", "yes", "on"):
        return True
    if vl in ("0", "false", "no", "off"):
        return False
    return None


def _bool(v: Any) -> bool:
    return bool(v)


def _num(v: Any, lo: float | None = None, hi: float | None = None) -> float:
    n = float(v)
    if lo is not None:
        n = max(lo, n)
    if hi is not None:
        n = min(hi, n)
    return n


def _int(v: Any, lo: int | None = None, hi: int | None = None) -> int:
    n = int(float(v))
    if lo is not None:
        n = max(lo, n)
    if hi is not None:
        n = min(hi, n)
    return n


def _resolve(cfg: dict, config_keys: tuple[str, ...], env_var: str | None,
             cast, default, lo=None, hi=None):
    """Unified resolver: config keys → env var → default, then cast + clamp."""
    v = _first(cfg, *config_keys)
    if v is None and env_var:
        v = _env_str(env_var)
    if v is None:
        v = default
    if cast is bool:
        return _bool(v)
    if lo is not None or hi is not None:
        if cast is int:
            return _int(v, lo=lo, hi=hi)
        return _num(v, lo=lo, hi=hi)
    return cast(v) if cast else v


# ---------------------------------------------------------------------------
# Config field mapping: (attr, config_keys, env_var, cast, default, lo, hi)
# ---------------------------------------------------------------------------
_CFG_FIELDS: list[tuple] = [
    # LLM
    ("model",           ("model",),           None,            str,   "gpt-4o-mini"),
    ("temperature",     ("temperature",),      None,            float, 0.7,         0.0, 2.0),
    ("max_tokens",      ("max_tokens", "max_output_tokens"), None, int, 8192, 1),
    ("stream",          ("stream",),           None,            bool,  True),
    ("stream_flush_chars", ("stream_flush_chars",), "CORE_LOOP_STREAM_FLUSH_CHARS", int, 0, 0),
    # Tokenizer
    ("tokenizer_reference_model", ("tokenizer_reference_model",), None, str, "gpt-4o-mini"),
    # Agent DB
    ("context_window_max_tokens", ("context_window_max_tokens",), None, int, 500_000, 4096),
    ("context_window_tail_tokens", ("context_window_tail_tokens",), None, int, 100_000, 1024),
    # Tools
    ("max_tool_rounds",  ("max_tool_rounds",),   None,            int,  20,         1),
    # Core memory
    ("core_memory_max_tokens", ("core_memory_max_tokens",), None, int, 20_000, 128),
    # exec
    ("exec_stdout_max_chars", ("exec_stdout_max_chars",), None, int, 32_000, 1024),
    ("max_exec_source_chars", ("max_exec_source_chars",), None, int, 12_000, 1),
    ("exec_batch_interrupt_on_human", ("exec_batch_interrupt_on_human",),
     "CORE_LOOP_EXEC_BATCH_INTERRUPT_ON_HUMAN", bool, False),
    # Chain gaps
    ("self_continue_gap_sec", ("self_continue_gap_sec",
                               "self_continue_gap_sec_autonomous",
                               "self_continue_gap_sec_interactive",
                               "self_continue_gap_sec_fast"),
     None, float, 15.0, 0.0),
    # Wait caps
    ("sleep_default_sec",  ("sleep_default_sec", "idle_watchdog_sec"),
     "CORE_LOOP_SLEEP_DEFAULT_SEC", float, 1800.0, 0.0),
    ("sleep_max_sec",      ("sleep_max_sec",),    "CORE_LOOP_SLEEP_MAX_SEC", float, 86_400.0, 1.0),
    # I/O
    ("terminal_quiet",     ("terminal_quiet",),   "CORE_LOOP_TERMINAL_QUIET", bool, False),
    ("stdin_input",        ("stdin_input",),       "CORE_LOOP_STDIN_INPUT",   bool, True),
]

# Env-only overrides (no config key, checked in apply)
_ENV_OVERRIDES: list[tuple] = [
    ("self_continue_gap_sec",  "CORE_LOOP_SELF_CONTINUE_GAP_SEC", float, 15.0, 0.0),
    ("self_continue_gap_sec",  "CORE_LOOP_SELF_CONTINUE_GAP_AUTONOMOUS", float, None, 0.0),
    ("self_continue_gap_sec",  "CORE_LOOP_SELF_CONTINUE_GAP_INTERACTIVE", float, None, 0.0),
    ("self_continue_gap_sec",  "CORE_LOOP_SELF_CONTINUE_GAP_FAST", float, None, 0.0),
    ("sleep_default_sec",      "CORE_LOOP_SLEEP_DEFAULT_SEC", float, None, 0.0),
    ("sleep_default_sec",      "CORE_LOOP_IDLE_WATCHDOG_SEC", float, None, 0.0),
    ("sleep_max_sec",          "CORE_LOOP_SLEEP_MAX_SEC", float, None, 1.0),
]


# ---------------------------------------------------------------------------
# Config singleton dataclass
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Runtime configuration — singleton via ``Config.get()``."""

    # --- LLM ---
    api_base_url: str = "https://api.openai.com/v1/chat/completions"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 8192
    stream: bool = True
    stream_flush_chars: int = 0
    http_timeout: float = 600.0
    openai_extra_body: dict[str, Any] = field(default_factory=dict)

    # --- Tokenizer ---
    tokenizer_reference_model: str = "gpt-4o-mini"

    # --- Agent SQLite ---
    agent_db_file: str = "workspace/agent.db"
    context_window_max_tokens: int = 500_000
    context_window_tail_tokens: int = 100_000

    # --- Function calling ---
    tool_definitions: list[dict[str, Any]] = field(default_factory=list)
    max_tool_rounds: int = 20

    # --- Core memory ---
    core_memory_max_tokens: int = 20_000

    # --- exec tool ---
    exec_stdout_max_chars: int = 32_000
    max_exec_source_chars: int = 12_000
    exec_batch_interrupt_on_human: bool = False

    # --- Chain gaps ---
    self_continue_gap_sec: float = 15.0

    # --- Wait caps ---
    sleep_default_sec: float = 1800.0
    sleep_max_sec: float = 86_400.0

    # --- I/O ---
    log_file: str = ""
    terminal_quiet: bool = False
    stdin_input: bool = True

    # --- Network ---
    control_host: str = "127.0.0.1"
    control_port: int = 0
    ui_host: str = "127.0.0.1"
    ui_port: int = 0
    web_host: str = "127.0.0.1"
    web_port: int = 0

    # --- Workspace ---
    workspace_rel: str = "workspace"

    # --- Runtime ---
    openai_client: Any = field(default=None, repr=False)
    _instance: Config | None = field(default=None, init=False, repr=False)

    @classmethod
    def get(cls) -> Config:
        if cls._instance is None:
            cls._instance = Config()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def load(self) -> dict:
        path = _config_path()
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing config: {path}\n"
                f"Copy config.json.example to config.json and set base_url, api_key, model, etc."
            )
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def apply(self, cfg: dict) -> None:
        """Apply a loaded config dict to this Config instance."""
        from openai import AsyncOpenAI

        # --- Table-driven fields ---
        for attr, keys, env_var, cast, default, *bounds in _CFG_FIELDS:
            lo = bounds[0] if len(bounds) >= 1 else None
            hi = bounds[1] if len(bounds) >= 2 else None
            v = _first(cfg, *keys)
            if v is None and env_var:
                v = _env_str(env_var)
            if v is not None:
                if cast is bool:
                    setattr(self, attr, _bool(v))
                elif cast is int:
                    setattr(self, attr, _int(v, lo=lo, hi=hi))
                elif cast is float:
                    setattr(self, attr, _num(v, lo=lo, hi=hi))
                else:
                    setattr(self, attr, cast(v))

        # --- Env-only overrides (only if field still at default) ---
        for attr, env_var, cast, default, lo in _ENV_OVERRIDES:
            v = _env_str(env_var)
            if v is None or default is None:
                continue
            current = getattr(self, attr)
            if current == default:
                n = _num(v, lo=lo) if cast is float else v
                setattr(self, attr, n)

        # --- context_window_tail_tokens cap ---
        if self.context_window_tail_tokens >= self.context_window_max_tokens:
            self.context_window_tail_tokens = max(1024, self.context_window_max_tokens // 2)

        # --- LLM specials ---
        ep = str(_first(cfg, "base_url", "endpoint", "api_url") or self.api_base_url).strip()
        self.api_base_url = ep or self.api_base_url
        self.api_key = _resolve_api_key(cfg)

        eb = _first(cfg, "llm_extra_body", "openai_extra_body")
        self.openai_extra_body = dict(eb) if isinstance(eb, dict) and eb else {}

        # --- Tools ---
        raw_tools = cfg.get("tools")
        if isinstance(raw_tools, list) and any(isinstance(t, dict) for t in raw_tools):
            self.tool_definitions = [t for t in raw_tools if isinstance(t, dict)]
        else:
            from agent.tools import get_tool_definitions
            self.tool_definitions = list(get_tool_definitions())

        # --- Workspace + DB ---
        wr = str(_first(cfg, "workspace_rel", "workspace") or "").strip()
        self.workspace_rel = os.path.expanduser(wr) if wr else self.workspace_rel

        default_agent_db = os.path.join(self.workspace_rel, "agent.db")
        adb = str(_first(cfg, "agent_db_file", "agent_sqlite_file", "agent_sqlite",
                          "consciousness_db_file", "core_memory_db_file") or default_agent_db).strip()
        self.agent_db_file = os.path.abspath(os.path.expanduser(adb))

        # --- Core memory legacy ---
        if cfg.get("core_memory_max_kb") is not None:
            self.core_memory_max_tokens = max(128, int(float(cfg["core_memory_max_kb"]) * 1024 / 4))
        if cfg.get("core_memory_max_bytes") is not None:
            self.core_memory_max_tokens = max(128, int(int(cfg["core_memory_max_bytes"]) / 4))

        # --- I/O ---
        self.log_file = str(cfg.get("log_file") or "").strip()
        env_log = _env_str("CORE_LOOP_LOG_FILE")
        if env_log:
            self.log_file = env_log

        # --- Network ---
        ctl = _env_str("CORE_LOOP_CONTROL_LISTEN") or _first(cfg, "control_listen")
        self.control_host, self.control_port = _parse_host_port(ctl)

        ui_raw = _env_str("CORE_LOOP_UI_LISTEN") or _first(cfg, "ui_listen")
        self.ui_host, self.ui_port = _parse_host_port(ui_raw)

        web_raw = _env_str("CORE_LOOP_WEB_LISTEN") or _first(cfg, "web_listen")
        self.web_host, self.web_port = _parse_host_port(web_raw)

        # --- HTTP timeout ---
        rt = cfg.get("request_timeout")
        if isinstance(rt, dict):
            self.http_timeout = float(rt.get("total") or rt.get("sock_read") or rt.get("connect") or 600.0)
        elif cfg.get("timeout") is not None:
            self.http_timeout = float(cfg["timeout"])

        # --- OpenAI client ---
        base = _openai_base_url(self.api_base_url)
        kwargs = {"api_key": self.api_key, "timeout": self.http_timeout}
        if base:
            kwargs["base_url"] = base
        self.openai_client = AsyncOpenAI(**kwargs)

    def load_and_apply(self) -> dict:
        cfg = self.load()
        self.apply(cfg)
        return cfg

    @property
    def config_path(self) -> str:
        return _config_path()
