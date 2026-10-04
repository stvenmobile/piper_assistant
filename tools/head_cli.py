"""
Drive the head by hand, through the head link service (which must be running).

    python3 tools/head_cli.py face thinking            # show a state
    python3 tools/head_cli.py face listening --mood warm --attention -30
    python3 tools/head_cli.py config --max-brightness 40
    python3 tools/head_cli.py watch                    # print STATUS / EVENT lines from the ESP32
    python3 tools/head_cli.py demo                     # cycle through every state and mood
"""
import argparse
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from piper_brain.config import CONFIG          # noqa: E402
from piper_head import protocol                # noqa: E402


def connect():
    cfg = CONFIG["head"]
    try:
        return socket.create_connection((cfg["host"], cfg["port"]), timeout=2)
    except OSError as e:
        sys.exit(f"Can't reach the head link on {cfg['host']}:{cfg['port']} ({e}). "
                 f"Start it with: python3 src/piper_head/link.py")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("face")
    f.add_argument("state", choices=protocol.STATES)
    f.add_argument("--mood", choices=protocol.MOODS)
    f.add_argument("--attention", type=float)
    f.add_argument("--clear-attention", action="store_true")
    c = sub.add_parser("config")
    c.add_argument("--max-brightness", type=int)
    c.add_argument("--ring-offset", type=float)
    sub.add_parser("watch")
    sub.add_parser("demo")
    a = ap.parse_args()

    s = connect()
    if a.cmd == "face":
        kw = {}
        if a.attention is not None:
            kw["attention"] = a.attention
        elif a.clear_attention:
            kw["attention"] = None
        s.sendall(protocol.encode(protocol.face(a.state, a.mood, **kw)))
    elif a.cmd == "config":
        fields = {}
        if a.max_brightness is not None:
            fields["max_brightness"] = a.max_brightness
        if a.ring_offset is not None:
            fields["ring_offset"] = a.ring_offset
        s.sendall(protocol.encode(protocol.config(**fields)))
    elif a.cmd == "watch":
        s.settimeout(None)
        buf = b""
        while True:
            chunk = s.recv(1024)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                print(line.decode(errors="replace"))
    elif a.cmd == "demo":
        for state in ("idle", "listening", "thinking", "speaking", "sleeping", "error"):
            print(state)
            s.sendall(protocol.encode(protocol.face(state, "neutral", attention=None)))
            time.sleep(4)
        for mood in protocol.MOODS:
            print(f"listening, {mood}")
            s.sendall(protocol.encode(protocol.face("listening", mood)))
            time.sleep(4)
        print("attention sweep")
        for deg in list(range(-90, 91, 10)) + list(range(90, -91, -10)):
            s.sendall(protocol.encode(protocol.face("idle", "neutral", attention=deg)))
            time.sleep(0.15)
        s.sendall(protocol.encode(protocol.face("idle", "neutral", attention=None)))
    s.close()


if __name__ == "__main__":
    main()
