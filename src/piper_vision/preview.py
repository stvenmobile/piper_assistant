"""
A live preview of what Piper sees: http://<jetson>:<port>/ shows the camera with the detected
faces, the attended target and the frame rate. For tuning and debugging - it serves whatever
JPEG the vision loop last handed it.
"""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = b"""<!doctype html><meta name=viewport content="width=device-width">
<title>Piper vision</title><body style="margin:0;background:#111;color:#ccc;font-family:sans-serif">
<img src="/stream" style="width:100%;max-width:1280px;display:block;margin:auto">
<p style="text-align:center">Piper vision - live</p></body>"""


class Preview:
    def __init__(self, port: int):
        self.port = port
        self.jpeg = None
        self.cond = threading.Condition()

    def publish(self, jpeg: bytes):
        with self.cond:
            self.jpeg = jpeg
            self.cond.notify_all()

    def start(self):
        preview = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        while True:
                            with preview.cond:
                                preview.cond.wait(timeout=2)
                                jpeg = preview.jpeg
                            if jpeg is None:
                                continue
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                            time.sleep(0.05)               # ~20 fps is plenty for a preview
                    except (BrokenPipeError, ConnectionResetError):
                        return
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(PAGE)

        srv = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
