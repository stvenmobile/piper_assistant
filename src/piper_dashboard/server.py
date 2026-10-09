"""
Piper's dashboard: a small read-only web service over her memory (piper_memory). Standard library
only. It opens the database READ-ONLY, so it runs happily beside a research session.

    python3 src/piper_dashboard/server.py                 (start_piper.sh starts it)
    python3 src/piper_dashboard/server.py --db path/to/piper_memory.db --port 8080

Then open http://<jetson>:8080/ on the home network. It binds to dashboard.host (0.0.0.0 = every
interface): keep it on the LAN - never forward the port to the internet (the topics are private).
"""
import argparse
import json
import re
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from piper_brain.config import CONFIG, ROOT_DIR      # noqa: E402
from piper_dashboard import api                       # noqa: E402
from piper_memory import MemoryStore                  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"


class Dashboard:
    def __init__(self, db_path: Path):
        self.path = db_path
        self.mem = None
        self.lock = threading.Lock()

    def store(self):
        if self.mem is None:
            if not self.path.exists():
                raise FileNotFoundError(f"no memory database at {self.path} yet")
            self.mem = MemoryStore(self.path, None, readonly=True)
        return self.mem

    def handle(self, route: str, qs: dict):
        one = lambda k, d="": qs.get(k, [d])[0]
        with self.lock:
            mem = self.store()
            if route == "/api/status":
                return api.status(mem)
            if route == "/api/events":
                return api.events(mem, since=int(one("since", "0") or 0))
            if route == "/api/topics":
                return api.topics(mem)
            if route == "/api/sessions":
                return api.sessions(mem)
            if route == "/api/notable":
                return api.notable(mem, kind=one("kind"), show=one("show", "open"))
            if route == "/api/graph":
                return api.graph(mem, topic=int(one("topic", "0") or 0) or None, limit=min(300, int(one("limit", "120"))),
                                 vague=one("vague") == "1")
            if route == "/api/concept":
                return api.concept(mem, int(one("id", "0")))
            if route == "/api/findings":
                return api.findings(mem, topic=int(one("topic", "0") or 0) or None, stance=one("stance"),
                                    relevance=one("relevance"), source=one("source"), q=one("q").strip(),
                                    kind=one("kind", "claims"), judged=one("judged"),
                                    limit=min(200, int(one("limit", "50"))), offset=int(one("offset", "0")))
        return None

    def post(self, route: str):
        m = re.fullmatch(r"/api/notable/(\d+)/reviewed", route)
        if m:
            return {"ok": api.mark_reviewed(self.path, int(m.group(1)))}
        return None


def make_handler(dash: Dashboard):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):                   # quiet: no line per request
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path.startswith("/api/"):
                try:
                    data = dash.handle(u.path, parse_qs(u.query))
                except FileNotFoundError as e:
                    return self._send(503, json.dumps({"error": str(e)}).encode(), "application/json")
                except Exception as e:
                    traceback.print_exc()
                    return self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json")
                if data is None:
                    return self._send(404, b'{"error": "unknown endpoint"}', "application/json")
                return self._send(200, json.dumps(data, default=str).encode(), "application/json")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            try:
                data = dash.post(urlparse(self.path).path)
            except Exception as e:
                traceback.print_exc()
                return self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json")
            if data is None:
                return self._send(404, b'{"error": "unknown endpoint"}', "application/json")
            return self._send(200, json.dumps(data).encode(), "application/json")

    return Handler


def main():
    cfg = CONFIG["dashboard"]
    ap = argparse.ArgumentParser(description="Piper's dashboard")
    ap.add_argument("--db", help="memory database (default: memory.path)")
    ap.add_argument("--host", default=cfg["host"])
    ap.add_argument("--port", type=int, default=cfg["port"])
    args = ap.parse_args()
    db = Path(args.db) if args.db else ROOT_DIR / CONFIG["memory"]["path"]
    server = ThreadingHTTPServer((args.host, args.port), make_handler(Dashboard(db)))
    print(f"[Dashboard] http://{args.host}:{args.port}/  (memory {db}, read-only)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
