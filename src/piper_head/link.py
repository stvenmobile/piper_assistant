"""
Head link service: the one program that talks to piper-watch's mouth (its XIAO ESP32-S3).

    python3 -m piper_head.link          (from src/; start_piper.sh starts it)
    python3 src/piper_head/link.py -v   verbose: print every message to and from the mouth

* Finds the mouth on USB - the Espressif device (VID 303A) that answers "?" with "mouth: ..." - or
  uses head.mouth_port, and reconnects if it is unplugged or resets.
* Listens on 127.0.0.1:head.port for local clients - the assistant, tools - and passes their MOUTH
  messages to the mouth as its text commands (protocol.mouth_lines).
* Sets the mouth's style (head.mouth_style) and restores its last mode and status symbol whenever it
  (re)connects.
(The servos have their own adapter and are driven by vision directly, not through the link.)

Version 1.1.0 - created before 2026-11-24 (see git history), last modified 2026-11-24
History:
    1.1.0  2026-11-24  listening also stops the new excitedtalk mouth (-> neutral).
    1.0.0  -           earlier work, unversioned (git history)
"""
import argparse
import re
import socket
import sys
import threading
import time
from pathlib import Path

if __package__ in (None, ""):                        # allow `python3 src/piper_head/link.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from piper_brain.config import CONFIG
from piper_head import protocol

ESPRESSIF_VID = 0x303A


class HeadLink:
    """The link's logic, without any I/O: feed it lines, it returns what to send where."""

    def __init__(self, mouth_style: str | None = None):
        self.mouth_style = mouth_style  # "lips" / "spectrum", sent first whenever the mouth connects
        self.mouth_last = {}            # the mouth's last mode / symbol

    def to_mouth(self, msg: dict) -> list[str]:
        """A client's message -> text commands for the mouth XIAO (remembering mode and symbol)."""
        if msg.get("t") != "MOUTH":
            return []
        if msg.get("demo") is False:                    # the demo ends in neutral
            self.mouth_last["mode"] = "neutral"
        if msg.get("mode") in protocol.MOUTH_MODES:
            self.mouth_last["mode"] = msg["mode"]
        if msg.get("symbol") in protocol.MOUTH_SYMBOLS:
            self.mouth_last["symbol"] = msg["symbol"]
            # the mouth stops talking when it starts listening (its firmware does the same)
            if msg["symbol"] == "listening" and "mode" not in msg:
                talking = {"talk": "neutral", "happytalk": "smile", "excitedtalk": "neutral"}
                if self.mouth_last.get("mode") in talking:
                    self.mouth_last["mode"] = talking[self.mouth_last["mode"]]
        return protocol.mouth_lines(msg)

    def mouth_on_connect(self) -> list[str]:
        """Mouth (re)connected: set its style, restore its mode and symbol."""
        style = [f"S{self.mouth_style}"] if self.mouth_style in ("lips", "spectrum") else []
        return style + protocol.mouth_lines({"t": "MOUTH", **self.mouth_last})

def find_mouth_port(setting: str, kind_of, exclude=()) -> str | None:
    """The mouth's XIAO: an Espressif device that answers "?" with "mouth: ..."."""
    if not setting or setting == "off":
        return None
    if setting != "auto":
        return setting
    from serial.tools import list_ports
    for p in list_ports.comports():
        if p.vid == ESPRESSIF_VID and p.device not in exclude and kind_of(p.device) == "mouth":
            return p.device
    return None


def open_serial(port: str, baud: int):
    """Open an ESP32-S3 native USB port. DTR on / RTS off BEFORE opening: on the S3's native USB
    those lines drive reset/boot, and with the defaults the board stopped accepting writes."""
    import serial
    ser = serial.Serial()
    ser.port, ser.baudrate = port, baud
    ser.timeout, ser.write_timeout = 0.2, 1
    ser.dtr, ser.rts = True, False
    ser.open()
    return ser


_REPORT = re.compile(r"^mouth: (LIVE|DEMO), mode ")


def is_mouth_report(line) -> bool:
    """The mouth's full state line ("mouth: LIVE, mode 0 neutral, ..."), not "mouth: LIVE" or "mouth: unknown ..."."""
    if isinstance(line, bytes):
        line = line.decode(errors="replace")
    return bool(_REPORT.match(line.strip()))


def mouth_style_of(line) -> str | None:
    """The style in a mouth report ("mouth: LIVE, mode 0 neutral, ..., style spectrum, contrast 160"), None if the
    firmware has no styles (older than the spectrum mouth) or it isn't a report."""
    if isinstance(line, bytes):
        line = line.decode(errors="replace")
    m = re.search(r", style (\w+)", line) if is_mouth_report(line) else None
    return m.group(1) if m else None


def probe_is_mouth(port: str, baud: int, wait_s: float = 1.5) -> bool:
    """Ask a board "?" (twice, in case it was just starting): the mouth answers "mouth: ..."."""
    try:
        ser = open_serial(port, baud)
    except Exception:
        return False
    data = b""
    try:
        end = time.monotonic() + wait_s
        asked = 0
        while time.monotonic() < end and b"mouth:" not in data:
            if asked < 2 and time.monotonic() > end - wait_s + asked * 0.7:
                ser.write(b"?\n")
                asked += 1
            data += ser.read(256)
    except Exception:
        pass
    finally:
        try:
            ser.close()
        except Exception:
            pass
    return b"mouth:" in data


def describe_ports() -> str:
    """The serial ports that ARE there - for the 'no ESP32 found' message."""
    from serial.tools import list_ports
    ports = [f"{p.device} ({p.vid or 0:04X}:{p.pid or 0:04X} {p.description})" for p in list_ports.comports()]
    return ", ".join(ports) or "none"


class Service:
    def __init__(self, cfg: dict, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
        self.core = HeadLink(CONFIG["head"].get("mouth_style"))
        self.mouth = None                     # the mouth XIAO's serial port
        self.mouth_port = None
        self.mouth_lock = threading.Lock()
        self.kinds: dict[str, str] = {}       # Espressif port -> "mouth" / "other" (asked once)

    # ---- which board is which ----------------------------------------------------------------
    def kind_of(self, device: str) -> str:
        if device not in self.kinds:
            mouth = probe_is_mouth(device, self.cfg["baud"])
            self.kinds[device] = "mouth" if mouth else "other"
            if self.verbose or mouth:
                print(f"[HeadLink] {device} is {'the mouth' if mouth else 'not the mouth'}")
        return self.kinds[device]

    def discover(self) -> str | None:
        """The mouth's port."""
        from serial.tools import list_ports
        present = {p.device for p in list_ports.comports()}
        for d in [d for d in self.kinds if d not in present]:   # unplugged: ask again next time
            del self.kinds[d]
        return find_mouth_port(self.cfg.get("mouth_port", "auto"), self.kind_of)

    # ---- the mouth XIAO ------------------------------------------------------------------------
    def send_mouth(self, lines):
        with self.mouth_lock:
            ser = self.mouth
            if ser is None:
                return                                  # optional: dropped quietly until it's there
            try:
                for line in lines:
                    ser.write((line + "\n").encode("ascii"))
                    if self.verbose and not line.startswith("L"):
                        print(f"[HeadLink] -> mouth  {line}")
            except Exception as e:
                print(f"[HeadLink] Mouth write failed: {e}")
                self.mouth = None
                try:
                    ser.close()
                except Exception:
                    pass

    def await_mouth(self, ser, tries: int = 4, wait_s: float = 1.0):
        """Send "?" until the mouth answers (it may be booting); returns its report line, or None."""
        for _ in range(tries):
            try:
                ser.reset_input_buffer()
                ser.write(b"?\n")
            except Exception:
                return None
            buf, until = b"", time.monotonic() + wait_s
            while time.monotonic() < until:
                try:
                    buf += ser.read(256)
                except Exception:
                    return None
                for line in buf.split(b"\n"):
                    if is_mouth_report(line):
                        return line.strip()
        return None

    def log_mouth_style(self, line: bytes):
        want, have = self.core.mouth_style, mouth_style_of(line)
        print(f"[HeadLink] Mouth says: {line.decode(errors='replace')}")
        if have is None:
            print(f"[HeadLink] Mouth firmware has no styles - flash the current piper-watch mouth firmware "
                  f"(wanted {want or 'its default'})")
        elif want in ("lips", "spectrum") and have != want:
            print(f"[HeadLink] Mouth reports style {have}, but head.mouth_style asks for {want} - it didn't take")
        else:
            print(f"[HeadLink] Mouth style: {have}" + ("" if want else " (head.mouth_style not set: left as is)"))

    def mouth_loop(self):
        if self.cfg.get("mouth_port", "auto") in ("", "off"):
            return
        last_warn = None
        while True:
            port = self.discover()
            if not port:
                if last_warn is None or (self.verbose and time.monotonic() - last_warn > 15):
                    last_warn = time.monotonic()      # once (every 15 s with --verbose)
                    print(f"[HeadLink] No mouth XIAO found yet - its style is set when it connects. "
                          f"Ports seen: {describe_ports()}")
                time.sleep(3)
                continue
            try:
                ser = open_serial(port, self.cfg["baud"])
            except Exception as e:
                print(f"[HeadLink] Can't open the mouth on {port}: {e}")
                time.sleep(3)
                continue
            print(f"[HeadLink] Mouth connected on {port}")
            # It may have just restarted when the port opened: wait until it answers before sending anything,
            # or the style / mode lines are lost while it boots.
            report = self.await_mouth(ser)
            if report is None:
                print("[HeadLink] Mouth didn't answer '?' yet - sending its settings anyway")
            with self.mouth_lock:
                self.mouth, self.mouth_port = ser, port
            self.send_mouth(self.core.mouth_on_connect() + ["?"])
            check_until = time.monotonic() + 2          # then check the style in its last report
            last_report = None
            buf = b""
            while True:
                with self.mouth_lock:
                    if self.mouth is not ser:
                        break
                if check_until and time.monotonic() > check_until:
                    check_until = 0
                    if last_report is None:
                        print("[HeadLink] Mouth didn't report its state after connecting")
                    else:
                        self.log_mouth_style(last_report)
                try:
                    buf += ser.read(256)
                except Exception as e:
                    print(f"[HeadLink] Mouth disconnected: {e}")
                    with self.mouth_lock:
                        self.mouth = None
                    try:
                        ser.close()
                    except Exception:
                        pass
                    break
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if check_until and is_mouth_report(line):
                        last_report = line
                    if self.verbose and line:
                        print(f"[HeadLink] <- mouth  {line.decode(errors='replace')}")
            with self.mouth_lock:
                self.mouth_port = None
            self.kinds.pop(port, None)
            time.sleep(1)

    def client_loop(self, conn: socket.socket):
        # a client that stops reading must not stall the link: give up on a send after 1 s
        conn.settimeout(1.0)
        peer = conn.getpeername()
        print(f"[HeadLink] Client connected {peer[0]}:{peer[1]}")
        conn.settimeout(None)
        buf = b""
        try:
            while True:
                chunk = conn.recv(1024)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    msg = protocol.decode(line)
                    if msg is None:
                        print(f"[HeadLink] Client sent something that isn't a message: {line[:120]!r}")
                        continue
                    if self.verbose:
                        print(f"[HeadLink] <- client {msg}")
                    self.send_mouth(self.core.to_mouth(msg))
        except OSError:
            pass
        finally:
            conn.close()
            print(f"[HeadLink] Client disconnected {peer[0]}:{peer[1]}")

    def serve(self):
        style = self.core.mouth_style
        print(f"[HeadLink] Mouth style (head.mouth_style in config.yaml): "
              + (style if style in ("lips", "spectrum") else f"{style!r} - not lips/spectrum, left as the mouth has it"
                 if style else "not set - left as the mouth has it"))
        threading.Thread(target=self.mouth_loop, daemon=True).start()
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.cfg["host"], self.cfg["port"]))
        srv.listen()
        print(f"[HeadLink] Listening for clients on {self.cfg['host']}:{self.cfg['port']}")
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=self.client_loop, args=(conn,), daemon=True).start()


def main():
    ap = argparse.ArgumentParser(description="piper-watch head link service")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every message to and from the mouth")
    args = ap.parse_args()
    try:
        Service(CONFIG["head"], verbose=args.verbose).serve()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
