"""Local HTTP servers for the generic multi-website admin demo."""

from __future__ import annotations

import html
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlsplit

from generic_models.site_catalog import PageSpec, WebsiteSpec, get_websites, root_path
from generic_models.visual_websites import render_visual_page


class RunningSiteServer(NamedTuple):
    spec: WebsiteSpec
    server: ThreadingHTTPServer
    thread: threading.Thread
    url: str
    log_path: Path


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_site_server(spec: WebsiteSpec, *, host: str = "127.0.0.1", log_dir: str | Path = "generic_models/live_logs") -> RunningSiteServer:
    """Start one generic website server in a daemon thread."""

    log_root = Path(log_dir)
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{spec.site_id}.jsonl"
    handler = _make_handler(spec=spec, log_path=log_path)
    server = ReusableThreadingHTTPServer((host, spec.port), handler)
    thread = threading.Thread(target=server.serve_forever, name=f"{spec.site_id}-server", daemon=True)
    thread.start()
    return RunningSiteServer(spec=spec, server=server, thread=thread, url=f"http://{host}:{spec.port}/", log_path=log_path)


def start_all_site_servers(*, host: str = "127.0.0.1", log_dir: str | Path = "generic_models/live_logs") -> list[RunningSiteServer]:
    """Start all configured generic websites."""

    return [start_site_server(spec, host=host, log_dir=log_dir) for spec in get_websites().values()]


def stop_site_servers(servers: list[RunningSiteServer]) -> None:
    """Stop all running generic website servers."""

    for running in servers:
        running.server.shutdown()
        running.server.server_close()
    for running in servers:
        running.thread.join(timeout=2)


def _make_handler(*, spec: WebsiteSpec, log_path: Path) -> type[BaseHTTPRequestHandler]:
    page_map = {page.path: page for page in spec.pages}
    canonical_root = root_path(spec.site_id)
    write_lock = threading.Lock()

    class GenericSiteHandler(BaseHTTPRequestHandler):
        server_version = "GenericWSDLab/1.0"

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            requested_path = _canonical_path(self.path, canonical_root=canonical_root)
            if requested_path in {"/favicon.ico", "/robots.txt"}:
                self._send_not_found(requested_path)
                return

            page = page_map.get(requested_path)
            if page is None:
                self._send_not_found(requested_path)
                return

            body = _render_page(spec, page)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body.encode("utf-8"))))
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
            self._log_event(path=requested_path, status_code=200)

        def log_message(self, _format: str, *_args: object) -> None:
            return

        def _send_not_found(self, requested_path: str) -> None:
            body = _render_not_found(spec, requested_path, canonical_root)
            self.send_response(404)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body.encode("utf-8"))))
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
            self._log_event(path=requested_path, status_code=404)

        def _log_event(self, *, path: str, status_code: int) -> None:
            event = {
                "timestamp": time.time(),
                "site_id": spec.site_id,
                "ip": self.client_address[0],
                "method": self.command,
                "path": path,
                "status_code": status_code,
                "referrer": self.headers.get("Referer", ""),
                "user_agent": self.headers.get("User-Agent", ""),
            }
            with write_lock:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=True) + "\n")

    return GenericSiteHandler


def _canonical_path(raw_path: str, *, canonical_root: str) -> str:
    path = urlsplit(raw_path).path or "/"
    if path == "/":
        return canonical_root
    return path.rstrip("/") if path != canonical_root else canonical_root


def _render_page(spec: WebsiteSpec, page: PageSpec) -> str:
    return render_visual_page(spec, page)


def _render_not_found(spec: WebsiteSpec, path: str, canonical_root: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8" /><title>Not found</title></head>
<body style="font-family: sans-serif; padding: 40px;">
  <h1>{html.escape(spec.name)} could not find this path</h1>
  <p>{html.escape(path)}</p>
  <p><a href="{html.escape(canonical_root)}">Return to site root</a></p>
</body>
</html>"""
