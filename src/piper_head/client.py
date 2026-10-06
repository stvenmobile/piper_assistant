"""
Client for the head link service - what the assistant uses to drive the light ring.

Never blocks the conversation and never raises: if the link service isn't running, messages
are dropped and it quietly tries to reconnect (at most every couple of seconds).
"""
import socket
import threading
import time

from piper_brain.config import CONFIG
from piper_head import protocol


class HeadClient:
    RETRY_S = 2.0

    def __init__(self, host: str | None = None, port: int | None = None, enabled: bool | None = None):
        cfg = CONFIG["head"]
        self.host = host or cfg["host"]
        self.port = port or cfg["port"]
        self.enabled = cfg["enabled"] if enabled is None else enabled
        self.sock = None
        self.next_try = 0.0

    def _connect(self) -> bool:
        if self.sock is not None:
            return True
        if not self.enabled or time.monotonic() < self.next_try:
            return False
        try:
            self.sock = socket.create_connection((self.host, self.port), timeout=0.3)
            self.sock.settimeout(0.3)
            threading.Thread(target=self._drain, args=(self.sock,), daemon=True).start()
            return True
        except OSError:
            self.sock = None
            self.next_try = time.monotonic() + self.RETRY_S
            return False

    @staticmethod
    def _drain(sock):
        """Read and discard what the link sends (STATUS twice a second): a client that never
        reads fills its socket buffer, and the link then drops it."""
        while True:
            try:
                if not sock.recv(4096):
                    return
            except socket.timeout:
                continue
            except OSError:
                return

    def send(self, msg: dict) -> bool:
        if not self._connect():
            return False
        try:
            self.sock.sendall(protocol.encode(msg))
            return True
        except OSError:
            self.close()
            self.next_try = time.monotonic() + self.RETRY_S
            return False

    def face(self, state: str, mood: str | None = None, **kw) -> bool:
        """Show a state on the ring; mood and attention= are optional."""
        return self.send(protocol.face(state, mood, **kw))

    def attention(self, deg: float | None) -> bool:
        """Point the ring's attention arc (0 = top, clockwise from the front), or None to clear."""
        return self.send(protocol.attention(deg))

    def assistant_state(self, status: str, mood: str | None = None) -> bool:
        """The assistant's own state names (IDLE / ENGAGED / PROCESSING / SPEAKING)."""
        state = protocol.ASSISTANT_TO_FACE.get(status)
        return self.face(state, mood) if state else False

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None
