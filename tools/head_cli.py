"""
Drive the mouth OLED by hand, through the head link service (which must be running).

    python3 tools/head_cli.py mouth smile              # neutral talk happytalk smile frown surprise
                                                       #   sleep happysleep research blank
    python3 tools/head_cli.py mouth talk --level 180   # with speech levels (0-255) for --seconds
    python3 tools/head_cli.py mouth demo               # its demo loop; "mouth demo-off" stops it
    python3 tools/head_cli.py status listening         # idle listening thinking confused
"""
import argparse
import socket
import sys
import threading
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


def drain(s):
    """Read and discard anything the link sends, so it never backs up."""
    def run():
        try:
            while s.recv(4096):
                pass
        except OSError:
            pass
    threading.Thread(target=run, daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mouth", help="the mouth OLED's expression, or its demo loop")
    m.add_argument("mode", choices=protocol.MOUTH_MODES + ("demo", "demo-off"))
    m.add_argument("--level", type=int, help="also send speech levels 0-255 (talk modes)")
    m.add_argument("--seconds", type=float, default=3.0, help="how long to send --level (default 3)")
    st = sub.add_parser("status", help="the status symbol in the mouth's top-right corner")
    st.add_argument("symbol", choices=protocol.MOUTH_SYMBOLS)
    a = ap.parse_args()

    s = connect()
    if a.cmd == "mouth":
        drain(s)
        if a.mode in ("demo", "demo-off"):
            s.sendall(protocol.encode(protocol.mouth(demo=a.mode == "demo")))
        else:
            s.sendall(protocol.encode(protocol.mouth(mode=a.mode)))
            if a.level is not None:                    # ~30 levels a second, a little uneven like speech
                import random
                end = time.monotonic() + a.seconds
                while time.monotonic() < end:
                    lvl = max(0, min(255, a.level + random.randint(-60, 60))) / 255
                    s.sendall(protocol.encode(protocol.mouth(level=lvl)))
                    time.sleep(0.033)
        time.sleep(0.2)                                # let the link pass it on before closing
    elif a.cmd == "status":
        s.sendall(protocol.encode(protocol.mouth(symbol=a.symbol)))
        time.sleep(0.2)
    s.close()


if __name__ == "__main__":
    main()
