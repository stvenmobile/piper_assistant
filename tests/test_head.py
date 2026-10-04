import socket
import threading
import time

import pytest

from piper_head import protocol
from piper_head.client import HeadClient
from piper_head.link import HeadLink


# ---- protocol -------------------------------------------------------------------------------
def test_encode_decode_round_trip():
    msg = protocol.face("thinking", "curious", attention=-20)
    line = protocol.encode(msg)
    assert line.endswith(b"\n") and b" " not in line
    assert protocol.decode(line) == {"t": "FACE", "state": "thinking", "mood": "curious", "attention": -20.0}


@pytest.mark.parametrize("line", [b"", b"not json", b"[1, 2]", b'{"no_type": 1}'])
def test_decode_rejects_junk(line):
    assert protocol.decode(line) is None


def test_face_attention_three_ways():
    assert "attention" not in protocol.face("idle")                  # leave it as it is
    assert protocol.face("idle", attention=None)["attention"] is None   # clear it
    assert protocol.face("idle", attention=15)["attention"] == 15.0


def test_face_rejects_unknown_names():
    with pytest.raises(ValueError):
        protocol.face("dancing")
    with pytest.raises(ValueError):
        protocol.face("idle", "grumpy")


def test_every_assistant_state_has_a_face():
    for status in ("IDLE", "ENGAGED", "PROCESSING", "SPEAKING"):
        assert protocol.ASSISTANT_TO_FACE[status] in protocol.STATES


# ---- link logic -----------------------------------------------------------------------------
def test_heartbeats_count_up():
    link = HeadLink()
    assert [link.heartbeat()["seq"] for _ in range(3)] == [1, 2, 3]


def test_client_face_is_forwarded_and_remembered():
    link = HeadLink()
    out = link.from_client(protocol.face("listening", "warm"))
    assert out == [protocol.face("listening", "warm")]
    link.from_client(protocol.face("thinking"))                       # mood not repeated...
    assert link.last["FACE"] == {"t": "FACE", "state": "thinking", "mood": "warm"}   # ...but kept


def test_unknown_client_messages_are_not_forwarded():
    assert HeadLink().from_client({"t": "SELF_DESTRUCT"}) == []


def test_esp32_reboot_restores_the_face():
    link = HeadLink()
    link.from_client(protocol.config(max_brightness=40))
    link.from_client(protocol.face("speaking"))
    back, to_clients = link.from_device({"t": "EVENT", "what": "boot"})
    assert back == [{"t": "CONFIG", "max_brightness": 40}, {"t": "FACE", "state": "speaking"}]
    assert to_clients == [{"t": "EVENT", "what": "boot"}]


def test_status_is_kept_and_passed_on():
    link = HeadLink()
    back, to_clients = link.from_device({"t": "STATUS", "state": "idle"})
    assert back == [] and link.status == {"t": "STATUS", "state": "idle"}
    assert to_clients == [{"t": "STATUS", "state": "idle"}]


def test_connect_sends_heartbeat_then_state():
    link = HeadLink()
    link.from_client(protocol.face("idle"))
    msgs = link.on_connect()
    assert msgs[0]["t"] == "HEARTBEAT" and msgs[-1] == {"t": "FACE", "state": "idle"}


# ---- client -----------------------------------------------------------------------------------
def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_client_without_a_link_never_blocks_or_raises():
    c = HeadClient(port=free_port(), enabled=True)
    t0 = time.monotonic()
    for _ in range(20):
        assert c.face("thinking") is False
    assert time.monotonic() - t0 < 1.0                 # one quick failed connect, then backs off


def test_disabled_client_sends_nothing():
    assert HeadClient(port=free_port(), enabled=False).face("idle") is False


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
    assert c.assistant_state("PROCESSING", "curious") is True
    th.join(2)
    srv.close()
    assert got == [{"t": "FACE", "state": "thinking", "mood": "curious"}]
