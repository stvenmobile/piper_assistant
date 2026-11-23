"""The head link's message framing, its client, and which USB device is the servo adapter."""
import socket
import threading
import time

import pytest

from piper_head import protocol
from piper_head.client import HeadClient


# ---- protocol -------------------------------------------------------------------------------
def test_encode_decode_round_trip():
    msg = protocol.mouth(level=0.25, mode="talk", symbol="thinking")
    line = protocol.encode(msg)
    assert line.endswith(b"\n") and b" " not in line
    assert protocol.decode(line) == {"t": "MOUTH", "level": 0.25, "mode": "talk", "symbol": "thinking"}


@pytest.mark.parametrize("line", [b"", b"not json", b"[1, 2]", b'{"no_type": 1}'])
def test_decode_rejects_junk(line):
    assert protocol.decode(line) is None


def test_mouth_level_is_clamped():
    assert protocol.mouth(1.7) == {"t": "MOUTH", "level": 1.0}


# ---- the client -------------------------------------------------------------------------------
def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_client_without_a_link_never_blocks_or_raises():
    c = HeadClient(port=free_port(), enabled=True)
    t0 = time.monotonic()
    for _ in range(20):
        assert c.mouth(mode="smile") is False
    assert time.monotonic() - t0 < 1.0                 # one quick failed connect, then backs off


def test_disabled_client_sends_nothing():
    assert HeadClient(port=free_port(), enabled=False).mouth(mode="smile") is False


def test_client_reaches_a_running_link():
    port = free_port()
    got = []
    srv = socket.socket()
    srv.bind(("127.0.0.1", port))
    srv.listen()

    def serve():
        conn, _ = srv.accept()
        buf = b""
        while b"\n" not in buf:
            buf += conn.recv(1024)
        got.append(protocol.decode(buf.split(b"\n")[0]))
        conn.close()

    th = threading.Thread(target=serve, daemon=True)
    th.start()
    c = HeadClient(port=port, enabled=True)
    assert c.mouth(mode="frown", symbol="thinking") is True
    th.join(2)
    srv.close()
    assert got == [{"t": "MOUTH", "mode": "frown", "symbol": "thinking"}]


# ---- the servo adapter (CH343) is found by its USB id -----------------------------------------------
def _ports(monkeypatch, ports):
    from serial.tools import list_ports
    from types import SimpleNamespace as P
    monkeypatch.setattr(list_ports, "comports", lambda: [P(device=d, vid=v, pid=p) for d, v, p in ports])


def test_servos_find_the_adapter(monkeypatch):
    from piper_head.servos import find_port
    _ports(monkeypatch, [("/dev/ttyUSB0", 0x1A86, 0x7523), ("/dev/ttyACM0", 0x1A86, 0x55D3)])
    assert find_port() == "/dev/ttyACM0"
    _ports(monkeypatch, [("/dev/ttyUSB0", 0x1A86, 0x7523)])
    assert find_port() is None
