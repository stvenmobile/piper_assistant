"""
The vision service's own little server (127.0.0.1:vision.events_port): newline-delimited JSON,
the same framing as the head link (piper_head.protocol). The assistant connects with
piper_vision.client.VisionClient.

    vision -> clients   PRESENCE {present, track, who, recognition}
                            who: a name, "" = a stranger, null = not recognised yet
                            recognition: false if face recognition isn't available (no model)
                        ENROLLED {name, track, looks} | ENROLL_FAILED {name, track, reason}
                        FORGOTTEN {name, ok}
    clients -> vision   ENROLL {name, track}    learn tracked person `track`'s face as `name`
                        FORGET {name}

New clients get the current PRESENCE straight away. Commands are queued for the vision loop
(the only thread that touches the camera and the face library).
"""
import queue
import socket
import threading

from piper_head import protocol


class Hub:
    def __init__(self, port: int, host: str = "127.0.0.1", recognition: bool = True):
        self.addr = (host, port)
        self.recognition = recognition
        self.commands: queue.Queue = queue.Queue()
        self.clients: list[socket.socket] = []
        self.lock = threading.Lock()
        self.presence = {"t": "PRESENCE", "present": False, "track": None, "who": None,
                         "recognition": recognition}

    def start(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(self.addr)
        srv.listen(4)
        threading.Thread(target=self._accept, args=(srv,), daemon=True).start()

    def _accept(self, srv):
        while True:
            conn, _ = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self.lock:
                self.clients.append(conn)
                self._send(conn, self.presence)
            threading.Thread(target=self._read, args=(conn,), daemon=True).start()

    def _read(self, conn):
        buf = b""
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    msg = protocol.decode(line)
                    if msg:
                        self.commands.put(msg)
        except OSError:
            pass
        self._drop(conn)

    @staticmethod
    def _send(conn, msg) -> bool:
        try:
            conn.sendall(protocol.encode(msg))
            return True
        except OSError:
            return False

    def _drop(self, conn):
        with self.lock:
            if conn in self.clients:
                self.clients.remove(conn)
        try:
            conn.close()
        except OSError:
            pass

    def broadcast(self, msg: dict):
        with self.lock:
            if msg.get("t") == "PRESENCE":
                self.presence = msg
            dead = [c for c in self.clients if not self._send(c, msg)]
        for c in dead:
            self._drop(c)

    def set_presence(self, present: bool, track=None, who=None):
        msg = {"t": "PRESENCE", "present": present, "track": track, "who": who,
               "recognition": self.recognition}
        if msg != self.presence:
            self.broadcast(msg)
