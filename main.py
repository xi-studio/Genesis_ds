"""Core Loop Server — Web UI + WebSocket + Control + Stdin.

This module handles ALL server/UI concerns. core_loop.py remains pure kernel.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading

import core_loop  # noqa: E402 — same-directory entry point
from agent.config import Config
from agent.output import say
from agent.host_primitives import trigger, _cancel_infer_if_running
from agent.timestamp import now_local
import agent.ui_stub as _us


# --- Override ui_stub emit_ui_event with real implementation ---
async def _emit_ui_event_real(ev: dict) -> None:
    """Push one JSON event per client: NDJSON + newline on TCP TUI; raw JSON text on WebSocket."""
    cfg = Config.get()
    if cfg.ui_port <= 0 and cfg.web_port <= 0:
        return
    lk = _us.ui_broadcast_lock
    if lk is None:
        return
    raw = json.dumps(ev, ensure_ascii=False, default=str, allow_nan=False)
    n_tcp = len(_us.ui_clients)
    n_ws = len(_us.web_ws_clients)
    if (
        ev.get("event") == "infer_end"
        and ev.get("ok") is True
        and cfg.web_port > 0
        and n_ws == 0
        and n_tcp == 0
    ):
        say(
            "  [ui] infer_end：无已连接的 WebSocket/NDJSON 客户端，"
            "网页收不到事件。请用 config 里 web_listen 的地址打开 http 页面（勿用本地 file://）。",
            flush=True,
        )
    line = raw + "\n"
    b = line.encode("utf-8")
    async with lk:
        dead_tcp: list[asyncio.StreamWriter] = []
        for w in list(_us.ui_clients):
            try:
                w.write(b)
                await w.drain()
            except (ConnectionResetError, BrokenPipeError, OSError, RuntimeError):
                dead_tcp.append(w)
        for w in dead_tcp:
            _us.ui_clients.discard(w)

        dead_ws: list[object] = []
        for ws in list(_us.web_ws_clients):
            try:
                await ws.send_str(raw)  # type: ignore[union-attr]
            except (ConnectionResetError, BrokenPipeError, OSError, RuntimeError, TypeError):
                dead_ws.append(ws)
        for ws in dead_ws:
            _us.web_ws_clients.discard(ws)


# Patch the stub
_us.emit_ui_event = _emit_ui_event_real


# --- Web Server (aiohttp) ---
async def _websocket_handler(request: "object") -> "object":
    """Handle browser WebSocket connections (/ws)."""
    from aiohttp import web

    ws = web.WebSocketResponse()
    await ws.prepare(request)
    lk = _us.ui_broadcast_lock
    if lk is not None:
        async with lk:
            _us.web_ws_clients.add(ws)
    try:
        cfg = Config.get()
        await _us.emit_ui_event({"event": "hello", "model": cfg.model, "server_time": now_local()})
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                await _dispatch_ui_message(data)
            elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
                break
    finally:
        if lk is not None:
            async with lk:
                _us.web_ws_clients.discard(ws)
    return ws


async def _start_web_server() -> None:
    """Start aiohttp web server for browser UI."""
    from aiohttp import web

    web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

    async def index(_request: web.Request) -> web.StreamResponse:
        index_path = os.path.join(web_dir, "index.html")
        if not os.path.isfile(index_path):
            return web.Response(
                text="Missing web/index.html next to core_loop.py.",
                status=500,
            )
        resp = web.FileResponse(index_path)
        resp.headers["Cache-Control"] = "no-store, max-age=0"
        return resp

    async def favicon(_request: web.Request) -> web.StreamResponse:
        svg = (
            "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
            "<circle cx='50' cy='50' r='40' fill='none' stroke='#00d4aa' stroke-width='4'/>"
            "<circle cx='50' cy='50' r='8' fill='#00d4aa'/></svg>"
        )
        return web.Response(
            text=svg,
            content_type="image/svg+xml",
            charset="utf-8",
        )

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/favicon.ico", favicon)
    app.router.add_get("/ws", _websocket_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    cfg = Config.get()
    site = web.TCPSite(runner, cfg.web_host, cfg.web_port)
    await site.start()


# --- TUI Session (TCP NDJSON) ---
async def _ui_session(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """NDJSON from TUI: {"cmd":"chat","text":"..."}, {"cmd":"cancel"}, {"cmd":"ping"}."""
    lk = _us.ui_broadcast_lock
    if lk is not None:
        async with lk:
            _us.ui_clients.add(writer)
    try:
        cfg = Config.get()
        await _us.emit_ui_event({"event": "hello", "model": cfg.model, "server_time": now_local()})
        while True:
            raw = await reader.readline()
            if not raw:
                break
            try:
                msg = json.loads(raw.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            await _dispatch_ui_message(msg)
    except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
        pass
    finally:
        if lk is not None:
            async with lk:
                _us.ui_clients.discard(writer)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


# --- Control Session (TCP line protocol) ---
async def _control_session(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """Line protocol: chat → trigger; /cancel /stop /ping /help."""
    try:
        while True:
            raw = await reader.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line in ("/cancel", "/stop"):
                ok = _cancel_infer_if_running()
                reply = "cancel: scheduled\n" if ok else "cancel: idle\n"
            elif line == "/ping":
                reply = "pong\n"
            elif line.startswith("/help"):
                reply = "text → [Human] trigger; /cancel /stop /ping /help\n"
            else:
                say(f"  [control] {line[:200]}")
                trigger(f"[Human] {line}")
                reply = "ok\n"
            writer.write(reply.encode("utf-8"))
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
        pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


# --- Shared UI message dispatcher ---
async def _dispatch_ui_message(msg: dict) -> None:
    """Shared by TCP ui_listen and browser /ws."""
    cmd = msg.get("cmd")
    if cmd == "chat":
        text = (msg.get("text") or "").strip()
        if text:
            await _us.emit_ui_event({"event": "user", "text": text, "server_time": now_local()})
            trigger(f"[Human] {text}")
    elif cmd in ("cancel", "stop"):
        _cancel_infer_if_running()
    elif cmd == "ping":
        await _us.emit_ui_event({"event": "pong"})


# --- Stdin Listener ---
def _start_stdin_listener() -> None:
    """Start a background thread that reads lines from stdin and triggers the loop."""

    def _reader() -> None:
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                if line in ("/cancel", "/stop"):
                    _cancel_infer_if_running()
                elif line == "/ping":
                    pass
                elif line.startswith("/help"):
                    pass
                else:
                    say(f"  [stdin] {line[:200]}")
                    trigger(f"[Human] {line}")
        except (EOFError, KeyboardInterrupt):
            pass
        except Exception as e:
            say(f"  [stdin error] {e}", flush=True)

    t = threading.Thread(target=_reader, daemon=True, name="stdin-reader")
    t.start()


# --- Server Main ---
def main() -> int:
    """Start all servers and run the core loop."""
    cfg = Config.get()
    try:
        cfg.load_and_apply()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    # Initialize UI broadcast lock
    import agent.ui_stub as _us
    _us.ui_broadcast_lock = asyncio.Lock()

    _tty = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()

    if cfg.stdin_input and not _tty:
        say(
            "[core_loop] stdin is not a TTY — line input may not work in this environment. "
            "Use control_listen (e.g. `nc host port`), or a real terminal.",
            flush=True,
        )
    if cfg.stdin_input:
        _start_stdin_listener()
    elif cfg.control_port <= 0:
        say(
            "[core_loop] stdin_input is false but no control_listen — inject via trigger only.",
            file=sys.stderr,
            flush=True,
        )

    async def _main_with_servers():
        """Run core loop with all servers started."""
        # Start UI server
        if cfg.ui_port > 0:
            try:
                ui_srv = await asyncio.start_server(_ui_session, cfg.ui_host, cfg.ui_port)
                asyncio.create_task(ui_srv.serve_forever())
            except OSError as e:
                msg = f"  ui: bind {cfg.ui_host}:{cfg.ui_port} failed: {e}"
                say(msg, file=sys.stderr, flush=True)

        # Start Web server
        if cfg.web_port > 0:
            try:
                await _start_web_server()
                web_url = f"http://{cfg.web_host}:{cfg.web_port}/"
                print(f"  web: {web_url}", file=sys.stderr, flush=True)
                say(f"  web: {web_url}")
            except OSError as e:
                msg = f"  web: bind {cfg.web_host}:{cfg.web_port} failed: {e}"
                say(msg, file=sys.stderr, flush=True)
            except ImportError as e:
                msg = f"  web: aiohttp required — uv add aiohttp ({e})"
                say(msg, file=sys.stderr, flush=True)

        # Start Control server
        if cfg.control_port > 0:
            try:
                srv = await asyncio.start_server(
                    _control_session, cfg.control_host, cfg.control_port
                )
                asyncio.create_task(srv.serve_forever())
            except OSError as e:
                msg = f"  control: bind {cfg.control_host}:{cfg.control_port} failed: {e}"
                say(msg, file=sys.stderr, flush=True)

        # Run core loop
        await core_loop.main()

    try:
        asyncio.run(_main_with_servers())
    except KeyboardInterrupt:
        say("\n[core_loop] KeyboardInterrupt — exit", flush=True)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
