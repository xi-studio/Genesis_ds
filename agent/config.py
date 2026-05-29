"""
Configuration loader and runtime settings for MolAgent core loop.

All module-level state lives here. Other modules import ``config`` and read
attributes from the singleton ``Config.get()`` — no more scattered ``global``
declarations.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


def _config_path() -> str:
    env = os.environ.get("CORE_LOOP_CONFIG", "").strip()
    if env:
        return env
    # config.json lives in the parent of agent/ (i.e. core/)
    agent_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(agent_dir)
    return os.path.join(parent_dir, "config.json")


def _resolve_api_key(cfg: dict) -> str:
    for env_name in ("OPENAI_API_KEY", "DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY"):
        v = (os.environ.get(env_name) or "").strip()
        if v:
            return v
    return str(cfg.get("api_key") or "").strip()


def _openai_base_url(url: str) -> str:
    """Strip /chat/completions if present so AsyncOpenAI gets a …/v1 style base."""
    ep = url.strip().rstrip("/")
    if ep.endswith("/chat/completions"):
        return ep[: -len("/chat/completions")]
    return ep


def _parse_host_port(raw: str | None) -> tuple[str, int]:
    """Parse ``host:port`` or bare ``port`` string. Port 0 = disabled."""
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


def _timeout_from_config(cfg: dict) -> float:
    rt = cfg.get("request_timeout")
    if isinstance(rt, dict):
        for key in ("total", "sock_read", "connect"):
            if rt.get(key) is not None:
                return float(rt[key])
    if cfg.get("timeout") is not None:
        return float(cfg["timeout"])
    return 600.0


def _env_bool(key: str, default: bool | None = None) -> bool | None:
    v = (os.environ.get(key) or "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


def _env_float(key: str, default: float | None = None) -> float | None:
    v = (os.environ.get(key) or "").strip()
    if v:
        try:
            return float(v)
        except ValueError:
            pass
    return default


def _env_int(key: str, default: int | None = None) -> int | None:
    v = (os.environ.get(key) or "").strip()
    if v and v.isdigit():
        return int(v)
    return default


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
    # Extra JSON fields for OpenAI-compatible providers (e.g. DeepSeek thinking / reasoning_effort)
    openai_extra_body: dict[str, Any] = field(default_factory=dict)

    # --- Tokenizer ---
    tokenizer_reference_model: str = "gpt-4o-mini"

    # --- Agent SQLite: consciousness + infer window + core memory (multiple tables, one file) ---
    agent_db_file: str = "workspace/agent.db"
    context_window_max_tokens: int = 500_000
    context_window_tail_tokens: int = 100_000

    # --- Function calling: default = built-in tool schemas (see apply); model may ignore them ---
    tool_definitions: list[dict[str, Any]] = field(default_factory=list)
    max_tool_rounds: int = 20

    # --- Core memory (tables inside agent DB; see agent.core_memory) ---
    core_memory_max_tokens: int = 20_000

    # --- exec tool / host Python (stdout mirrors to terminal + UI; not duplicated into consciousness) ---
    exec_stdout_max_chars: int = 32_000
    max_exec_source_chars: int = 12_000
    exec_batch_interrupt_on_human: bool = False

    # --- Chain gaps (interruptible; triggers wake early) ---
    self_continue_gap_sec: float = 15.0  # ``/next`` default and boot chain
    self_explore_gap_sec: float = 120.0

    # --- Wait phase caps (``/call_for_human``, ``/sleep``, or implicit wait tail) ---
    call_for_human_wait_sec: float = 60.0
    sleep_default_sec: float = 1800.0  # ``/sleep`` without a number; generic wait fallback
    sleep_max_sec: float = 86_400.0  # hard cap for any wait (e.g. 24h)

    # --- I/O ---
    log_file: str = ""
    terminal_quiet: bool = False
    split_input_prompt: bool = False
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

    # --- Runtime (not in config.json) ---
    openai_client: Any = field(default=None, repr=False)

    _instance: Config | None = field(default=None, init=False, repr=False)

    @classmethod
    def get(cls) -> Config:
        if cls._instance is None:
            cls._instance = Config()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """For testing — clear the singleton."""
        cls._instance = None

    def load(self) -> dict:
        """Load config.json and return raw dict (does NOT apply)."""
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

        # LLM
        ep = str(
            cfg.get("base_url") or cfg.get("endpoint") or cfg.get("api_url") or self.api_base_url
        ).strip()
        self.api_base_url = ep or self.api_base_url
        self.api_key = _resolve_api_key(cfg)
        self.model = str(cfg.get("model") or self.model).strip()
        self.temperature = float(cfg.get("temperature", self.temperature))
        self.max_tokens = int(
            cfg.get("max_tokens") or cfg.get("max_output_tokens") or self.max_tokens
        )
        self.stream = cfg.get("stream", self.stream) is not False

        eb = cfg.get("llm_extra_body") or cfg.get("openai_extra_body")
        if isinstance(eb, dict) and eb:
            self.openai_extra_body = dict(eb)
        else:
            self.openai_extra_body = {}

        # Tools: non-empty list of dicts replaces defaults; anything else → built-in set.
        raw_tools = cfg.get("tools")
        if isinstance(raw_tools, list) and any(isinstance(t, dict) for t in raw_tools):
            self.tool_definitions = [t for t in raw_tools if isinstance(t, dict)]
        else:
            from agent.tools import get_tool_definitions

            self.tool_definitions = list(get_tool_definitions())

        mtr = cfg.get("max_tool_rounds")
        if mtr is not None:
            self.max_tool_rounds = max(1, int(mtr))

        sfc = cfg.get("stream_flush_chars")
        if sfc is not None:
            self.stream_flush_chars = max(0, int(sfc))
        else:
            env_sfc = _env_int("CORE_LOOP_STREAM_FLUSH_CHARS")
            if env_sfc is not None:
                self.stream_flush_chars = max(0, env_sfc)

        # Tokenizer
        tr_cfg = cfg.get("tokenizer_reference_model")
        if tr_cfg is not None:
            self.tokenizer_reference_model = str(tr_cfg).strip() or "gpt-4o-mini"

        wr = (cfg.get("workspace_rel") or cfg.get("workspace") or "").strip()
        if wr:
            self.workspace_rel = os.path.expanduser(wr)

        default_agent_db = os.path.join(self.workspace_rel, "agent.db")
        adb = (
            str(
                cfg.get("agent_db_file")
                or cfg.get("agent_sqlite_file")
                or cfg.get("agent_sqlite")
                or cfg.get("consciousness_db_file")
                or cfg.get("core_memory_db_file")
                or ""
            ).strip()
            or default_agent_db
        )
        self.agent_db_file = os.path.abspath(os.path.expanduser(adb))

        cwmax = cfg.get("context_window_max_tokens")
        if cwmax is not None:
            self.context_window_max_tokens = max(4096, int(cwmax))
        cwtail = cfg.get("context_window_tail_tokens")
        if cwtail is not None:
            self.context_window_tail_tokens = max(1024, int(cwtail))
        if self.context_window_tail_tokens >= self.context_window_max_tokens:
            self.context_window_tail_tokens = max(1024, self.context_window_max_tokens // 2)

        # Core memory (max injected markdown tokens)
        cmt = cfg.get("core_memory_max_tokens")
        if cmt is not None:
            self.core_memory_max_tokens = max(128, int(cmt))
        elif cfg.get("core_memory_max_kb") is not None:
            self.core_memory_max_tokens = max(
                128, int(float(cfg["core_memory_max_kb"]) * 1024 / 4)
            )
        elif cfg.get("core_memory_max_bytes") is not None:
            self.core_memory_max_tokens = max(128, int(int(cfg["core_memory_max_bytes"]) / 4))

        # exec tool limits
        if cfg.get("exec_stdout_max_chars") is not None:
            self.exec_stdout_max_chars = max(1024, int(cfg["exec_stdout_max_chars"]))

        ei = cfg.get("exec_batch_interrupt_on_human")
        env_ei = _env_bool("CORE_LOOP_EXEC_BATCH_INTERRUPT_ON_HUMAN")
        if ei is not None:
            self.exec_batch_interrupt_on_human = bool(ei)
        elif env_ei is not None:
            self.exec_batch_interrupt_on_human = env_ei

        # /self_continue chain gap (and legacy alias keys → single field)
        sc = cfg.get("self_continue_gap_sec")
        env_sc = _env_float("CORE_LOOP_SELF_CONTINUE_GAP_SEC")
        ga = cfg.get("self_continue_gap_sec_autonomous")
        gi = cfg.get("self_continue_gap_sec_interactive")
        gf = cfg.get("self_continue_gap_sec_fast")
        env_a = _env_float("CORE_LOOP_SELF_CONTINUE_GAP_AUTONOMOUS")
        env_i = _env_float("CORE_LOOP_SELF_CONTINUE_GAP_INTERACTIVE")
        env_f = _env_float("CORE_LOOP_SELF_CONTINUE_GAP_FAST")

        if sc is not None:
            self.self_continue_gap_sec = max(0.0, float(sc))
        elif env_sc is not None:
            self.self_continue_gap_sec = max(0.0, env_sc)
        elif ga is not None:
            self.self_continue_gap_sec = max(0.0, float(ga))
        elif env_a is not None:
            self.self_continue_gap_sec = max(0.0, env_a)
        elif gi is not None:
            self.self_continue_gap_sec = max(0.0, float(gi))
        elif env_i is not None:
            self.self_continue_gap_sec = max(0.0, env_i)
        elif gf is not None:
            self.self_continue_gap_sec = max(0.0, float(gf))
        elif env_f is not None:
            self.self_continue_gap_sec = max(0.0, env_f)
        else:
            self.self_continue_gap_sec = 15.0

        seg = cfg.get("self_explore_gap_sec")
        env_seg = _env_float("CORE_LOOP_SELF_EXPLORE_GAP_SEC")
        if seg is not None:
            self.self_explore_gap_sec = max(0.0, float(seg))
        elif env_seg is not None:
            self.self_explore_gap_sec = max(0.0, env_seg)
        else:
            self.self_explore_gap_sec = 120.0

        # ``/call_for_human`` wait — legacy key ``call_for_human_watchdog_sec`` still accepted
        cfw = cfg.get("call_for_human_wait_sec")
        cfh_legacy = cfg.get("call_for_human_watchdog_sec")
        env_cfw = _env_float("CORE_LOOP_CALL_FOR_HUMAN_WAIT_SEC")
        env_cfh = _env_float("CORE_LOOP_CALL_FOR_HUMAN_WATCHDOG_SEC")
        if cfw is not None:
            self.call_for_human_wait_sec = max(0.0, float(cfw))
        elif cfh_legacy is not None:
            self.call_for_human_wait_sec = max(0.0, float(cfh_legacy))
        elif env_cfw is not None:
            self.call_for_human_wait_sec = max(0.0, env_cfw)
        elif env_cfh is not None:
            self.call_for_human_wait_sec = max(0.0, env_cfh)
        else:
            self.call_for_human_wait_sec = 60.0

        sdef = cfg.get("sleep_default_sec")
        iwd_legacy = cfg.get("idle_watchdog_sec")
        env_sd = _env_float("CORE_LOOP_SLEEP_DEFAULT_SEC")
        env_iwd = _env_float("CORE_LOOP_IDLE_WATCHDOG_SEC")
        if sdef is not None:
            self.sleep_default_sec = max(0.0, float(sdef))
        elif iwd_legacy is not None:
            self.sleep_default_sec = max(0.0, float(iwd_legacy))
        elif env_sd is not None:
            self.sleep_default_sec = max(0.0, env_sd)
        elif env_iwd is not None:
            self.sleep_default_sec = max(0.0, env_iwd)
        else:
            self.sleep_default_sec = 1800.0

        smax = cfg.get("sleep_max_sec")
        env_smax = _env_float("CORE_LOOP_SLEEP_MAX_SEC")
        if smax is not None:
            self.sleep_max_sec = max(1.0, float(smax))
        elif env_smax is not None:
            self.sleep_max_sec = max(1.0, env_smax)
        else:
            self.sleep_max_sec = 86_400.0

        # I/O
        self.log_file = str(cfg.get("log_file") or "").strip()
        env_log = (os.environ.get("CORE_LOOP_LOG_FILE") or "").strip()
        if env_log:
            self.log_file = env_log

        tq = cfg.get("terminal_quiet")
        env_tq = _env_bool("CORE_LOOP_TERMINAL_QUIET")
        if tq is not None:
            self.terminal_quiet = bool(tq)
        elif env_tq is not None:
            self.terminal_quiet = env_tq
        else:
            self.terminal_quiet = False

        sip = cfg.get("split_input_prompt")
        env_sip = _env_bool("CORE_LOOP_SPLIT_INPUT")
        if sip is not None:
            self.split_input_prompt = bool(sip)
        elif env_sip is not None:
            self.split_input_prompt = env_sip

        if cfg.get("stdin_input") is not None:
            self.stdin_input = bool(cfg["stdin_input"])
        else:
            env_si = _env_bool("CORE_LOOP_STDIN_INPUT")
            if env_si is not None:
                self.stdin_input = env_si

        # Network
        ctl = (os.environ.get("CORE_LOOP_CONTROL_LISTEN") or "").strip() or cfg.get(
            "control_listen"
        )
        self.control_host, self.control_port = _parse_host_port(ctl if ctl else None)

        ui_raw = (os.environ.get("CORE_LOOP_UI_LISTEN") or "").strip() or cfg.get("ui_listen")
        self.ui_host, self.ui_port = _parse_host_port(ui_raw if ui_raw else None)

        web_raw = (os.environ.get("CORE_LOOP_WEB_LISTEN") or "").strip() or cfg.get("web_listen")
        self.web_host, self.web_port = _parse_host_port(web_raw if web_raw else None)

        # HTTP timeout
        self.http_timeout = _timeout_from_config(cfg)

        # OpenAI client
        base = _openai_base_url(self.api_base_url)
        kwargs = {"api_key": self.api_key, "timeout": self.http_timeout}
        if base:
            kwargs["base_url"] = base
        self.openai_client = AsyncOpenAI(**kwargs)

    def load_and_apply(self) -> dict:
        """Convenience: load config.json and apply. Returns raw dict."""
        cfg = self.load()
        self.apply(cfg)
        return cfg

    @property
    def config_path(self) -> str:
        return _config_path()
