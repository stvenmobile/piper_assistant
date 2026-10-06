"""
Drive the head by hand, through the head link service (which must be running).

    python3 tools/head_cli.py face thinking            # show a state
    python3 tools/head_cli.py face listening --mood warm --attention -30
    python3 tools/head_cli.py config --max-brightness 40
    python3 tools/head_cli.py watch                    # print STATUS / EVENT lines from the ESP32
    python3 tools/head_cli.py demo                     # step through every state and mood:
                                                       #   n = next, p = previous, + / - = brightness,
                                                       #   q = quit
"""
import argparse
import os
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


class Keys:
    """Single key presses without Enter (Linux terminal or Windows console)."""

    def __enter__(self):
        if os.name == "nt":
            import msvcrt
            self.msvcrt = msvcrt
        else:
            import termios
            import tty
            self.termios = termios
            self.fd = sys.stdin.fileno()
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        if os.name != "nt":
            self.termios.tcsetattr(self.fd, self.termios.TCSADRAIN, self.saved)

    def get(self, timeout: float | None = None) -> str | None:
        """The next key, or None if none was pressed within `timeout` seconds."""
        if os.name == "nt":
            end = None if timeout is None else time.monotonic() + timeout
            while end is None or time.monotonic() < end:
                if self.msvcrt.kbhit():
                    return self.msvcrt.getwch()
                time.sleep(0.02)
            return None
        import select
        if select.select([sys.stdin], [], [], timeout)[0]:
            return sys.stdin.read(1)
        return None


def demo_steps():
    """(label, message) for each step; message None = the attention sweep."""
    steps = [(state, protocol.face(state, "neutral", attention=None))
             for state in ("idle", "listening", "thinking", "speaking", "sleeping", "error")]
    steps += [(f"listening, {mood}", protocol.face("listening", mood, attention=None))
              for mood in protocol.MOODS]
    steps.append(("attention sweep (idle, arc moving left and right)", None))
    return steps


def demo(s, brightness: int):
    """Step through the states and moods by hand."""
    def send(msg):
        s.sendall(protocol.encode(msg))

    steps = demo_steps()
    sweep = list(range(-90, 91, 10)) + list(range(90, -91, -10))
    print("n = next   p = previous   + / - = brightness   q = quit\n")
    send(protocol.config(max_brightness=brightness))
    i, shown = 0, None
    with Keys() as keys:
        while True:
            if shown != i:
                label, msg = steps[i]
                print(f"[{i + 1}/{len(steps)}] {label}   (brightness {brightness})")
                if msg is not None:
                    send(msg)
                shown, k = i, 0
            if steps[i][1] is None:              # the sweep animates until a key is pressed
                send(protocol.face("idle", "neutral", attention=sweep[k % len(sweep)]))
                k += 1
                key = keys.get(0.15)
            else:
                key = keys.get()
            if key is None:
                continue
            key = key.lower()
            if key == "q":
                break
            if key == "n":
                i = (i + 1) % len(steps)
            elif key == "p":
                i = (i - 1) % len(steps)
            elif key in "+=-_":
                brightness = max(0, min(255, brightness + (5 if key in "+=" else -5)))
                send(protocol.config(max_brightness=brightness))
                print(f"      brightness {brightness}")
    send(protocol.face("idle", "neutral", attention=None))
    print("left the ring idle")


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
    d = sub.add_parser("demo")
    d.add_argument("--brightness", type=int, default=64, help="starting max_brightness (0-255)")
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
        demo(s, a.brightness)
    s.close()


if __name__ == "__main__":
    main()
