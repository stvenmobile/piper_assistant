"""
Client for the vision service - how the assistant knows who is in front of it.

A background thread keeps a connection to the vision service (retrying every couple of
seconds), tracks the latest PRESENCE, and hands other replies (ENROLLED, ...) to whoever is
waiting for them. Never raises; `connected` says whether vision is actually there.
"""
import queue
import socket
import threading
import time

from piper_brain.config import CONFIG
from piper_head import protocol


class VisionClient:
    RETRY_S = 2.0

    def __init__(self, host: str = "127.0.0.1", port: int | None = None, enabled: bool | None = None):
        cfg = CONFIG["vision"]
        self.addr = (host, port or cfg["events_port"])
        self.enabled = cfg["enabled"] if enabled is None else enabled
        self.sock = None
        self.connected = False
        self.presence = {"present": False, "track": None, "who": None, "recognition": False}
        self.replies: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        if self.enabled:
            threading.Thread(target=self._run, daemon=True).start()

    # --- what the assistant asks -------------------------------------------------------------
    def who(self) -> dict:
        """{present, track, who, recognition} - who: a name, "" = a stranger, None = not
        recognised yet. recognition is False when the vision service can't recognise faces."""
        with self.lock:
            return dict(self.presence)

    def still_here(self, track) -> bool:
        p = self.who()
        return p["present"] and p["track"] == track

    def wait_for_verdict(self, timeout: float) -> dict:
        """Wait (briefly) until whoever is present has been recognised one way or the other."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            p = self.who()
            if not p["present"] or p["who"] is not None:
                return p
            time.sleep(0.1)
        return self.who()

    def enroll(self, name: str, track, timeout: float = 15.0) -> dict:
        """Learn the face of tracked person `track` as `name`. Returns the ENROLLED or
        ENROLL_FAILED message (reason "timeout" / "no vision" if nothing came back)."""
        self._drain()
        if not self._send({"t": "ENROLL", "name": name, "track": track}):
            return {"t": "ENROLL_FAILED", "name": name, "reason": "no vision"}
        return self._wait(("ENROLLED", "ENROLL_FAILED"), timeout) or \
            {"t": "ENROLL_FAILED", "name": name, "reason": "timeout"}

    def forget(self, name: str, timeout: float = 3.0) -> bool:
        self._drain()
        if not self._send({"t": "FORGET", "name": name}):
            return False
        msg = self._wait(("FORGOTTEN",), timeout)
        return bool(msg and msg.get("ok"))

    # --- plumbing ------------------------------------------------------------------------------
    def _drain(self):
        while not self.replies.empty():
            self.replies.get_nowait()

    def _wait(self, types, timeout):
        end = time.monotonic() + timeout
        while (left := end - time.monotonic()) > 0:
            try:
                msg = self.replies.get(timeout=left)
            except queue.Empty:
                break
            if msg.get("t") in types:
                return msg
        return None

    def _send(self, msg) -> bool:
        sock = self.sock
        if sock is None:
            return False
        try:
            sock.sendall(protocol.encode(msg))
            return True
        except OSError:
            return False

    def _run(self):
        while True:
            try:
                sock = socket.create_connection(self.addr, timeout=2)
                sock.settimeout(None)
            except OSError:
                time.sleep(self.RETRY_S)
                continue
            self.sock, self.connected = sock, True
            buf = b""
            try:
                while True:
                    data = sock.recv(4096)
                    if not data:
                        break
                    buf += data
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        msg = protocol.decode(line)
                        if not msg:
                            continue
                        if msg["t"] == "PRESENCE":
                            with self.lock:
                                self.presence = {k: msg.get(k) for k in ("present", "track", "who", "recognition")}
                        else:
                            self.replies.put(msg)
            except OSError:
                pass
            self.sock, self.connected = None, False
            with self.lock:
                self.presence = {"present": False, "track": None, "who": None, "recognition": False}
            try:
                sock.close()
            except OSError:
                pass
            time.sleep(self.RETRY_S)
