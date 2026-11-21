"""
Head link service: the one program that talks to the piper-watch ESP32.

    python3 -m piper_head.link          (from src/; start_piper.sh starts it)
    python3 src/piper_head/link.py -v   verbose: print every message to and from the ESP32

* Finds the head's controller on USB - the face (the 2.4" CYD's CH340 USB-serial chip) or an
  ESP32-S3 (Espressif's own USB) - or uses head.serial_port, and reconnects if it is unplugged
  or resets. (The servo adapter's CH343 is from the same maker but a different chip: not matched.)
* Sends a HEARTBEAT every head.heartbeat_s (the ESP32 shows "offline" and, later, stops the
  motor if they stop).
* Listens on 127.0.0.1:head.port for local clients - the assistant, vision, tools - and forwards
  their FACE / CONFIG / LOOK / MOUTH messages to the ESP32. ESP32 STATUS / EVENT lines go to every client.
* Remembers the last FACE and CONFIG and re-sends them whenever the ESP32 (re)connects or
  reboots, so the ring is never stuck showing a stale state.
* The mouth OLED's own XIAO ESP32-S3 (head.mouth_port) is a second Espressif USB device. Each
  Espressif port is asked "?" once: the mouth answers "mouth: ...", the head controller ignores
  it. MOUTH messages go to the mouth as its text commands (protocol.mouth_lines); its last mode
  and status symbol are re-sent whenever it (re)connects. Speech levels also go to the head.
"""
import argparse
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
CH340 = (0x1A86, 0x7523)          # the CYD face's USB-serial chip (the servo adapter is a CH343, 1A86:55D3)
FORWARDED = {"FACE", "CONFIG", "LOOK", "MOUTH"}


class HeadLink:
    """The link's logic, without any I/O: feed it lines, it returns what to send where."""

    def __init__(self):
        self.seq = 0
        self.last = {}                  # last FACE / CONFIG sent, by type
        self.status = None              # last STATUS from the ESP32
        self.mouth_last = {}            # the mouth's last mode / symbol

    def heartbeat(self) -> dict:
        self.seq += 1
        return protocol.heartbeat(self.seq)

    def from_client(self, msg: dict) -> list[dict]:
        """A client's message -> messages for the ESP32."""
        if msg.get("t") in FORWARDED:
            if msg["t"] in ("FACE", "CONFIG"):
                merged = dict(self.last.get(msg["t"], {}))
                merged.update(msg)
                self.last[msg["t"]] = merged
            if msg["t"] == "MOUTH":                     # the head controller only gets speech levels
                return [{"t": "MOUTH", "level": msg["level"]}] if "level" in msg else []
            return [msg]
        return []

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
        return protocol.mouth_lines(msg)

    def mouth_on_connect(self) -> list[str]:
        """Mouth (re)connected: restore its mode and symbol."""
        return protocol.mouth_lines({"t": "MOUTH", **self.mouth_last})

    def from_device(self, msg: dict) -> tuple[list[dict], list[dict]]:
        """An ESP32 message -> (messages back to the ESP32, messages for the clients)."""
        to_device = []
        if msg.get("t") == "STATUS":
            self.status = msg
        if msg.get("t") == "EVENT" and msg.get("what") in ("boot", "link"):
            # it has just started or come back: restore what it should be showing
            to_device = [self.last[k] for k in ("CONFIG", "FACE") if k in self.last]
        return to_device, [msg]

    def on_connect(self) -> list[dict]:
        """Serial port (re)opened: a heartbeat first, then the remembered state."""
        return [self.heartbeat()] + [self.last[k] for k in ("CONFIG", "FACE") if k in self.last]


def find_port(setting: str, kind_of=None, exclude=()) -> str | None:
    """The head controller: the CH340 face, else an Espressif device that isn't the mouth.
    kind_of(device) -> "mouth" / "other" (asks the board; None = don't ask)."""
    if setting and setting != "auto":
        return setting
    from serial.tools import list_ports
    ports = [p for p in list_ports.comports() if p.device not in exclude]
    for p in ports:
        if (p.vid, p.pid) == CH340:
            return p.device
    for p in ports:
        if p.vid == ESPRESSIF_VID and (kind_of is None or kind_of(p.device) != "mouth"):
            return p.device
    return None


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
        self.drop_warned = False              # "not sent" said once per disconnection, not per message
        self.core = HeadLink()
        self.lock = threading.Lock()
        self.write_lock = threading.Lock()    # heartbeats and clients write from different threads
        self.serial = None
        self.clients: list[socket.socket] = []
        self.head_port = None                 # the ports the two loops have open
        self.mouth = None                     # the mouth XIAO's serial port
        self.mouth_port = None
        self.mouth_lock = threading.Lock()
        self.kinds: dict[str, str] = {}       # Espressif port -> "mouth" / "other" (asked once)
        self.discover_lock = threading.Lock() # one loop at a time looks for (and asks) boards

    # ---- which board is which ----------------------------------------------------------------
    def kind_of(self, device: str) -> str:
        if device not in self.kinds:
            mouth = probe_is_mouth(device, self.cfg["baud"])
            self.kinds[device] = "mouth" if mouth else "other"
            if self.verbose or mouth:
                print(f"[HeadLink] {device} is {'the mouth' if mouth else 'not the mouth'}")
        return self.kinds[device]

    def discover(self, which: str) -> str | None:
        """The port for "head" or "mouth", skipping the one the other loop has open."""
        from serial.tools import list_ports
        with self.discover_lock:
            present = {p.device for p in list_ports.comports()}
            for d in [d for d in self.kinds if d not in present]:   # unplugged: ask again next time
                del self.kinds[d]
            if which == "head":
                return find_port(self.cfg["serial_port"], self.kind_of, exclude={self.mouth_port} - {None})
            return find_mouth_port(self.cfg.get("mouth_port", "auto"), self.kind_of,
                                   exclude={self.head_port} - {None})

    # ---- serial ------------------------------------------------------------------------------
    def send_device(self, msgs):
        with self.lock:
            ser = self.serial
        for m in msgs:
            if ser is None:
                # say so once per disconnection (every message with --verbose), not on every
                # message: the head is optional, research and conversation carry on without it
                if m.get("t") != "HEARTBEAT" and (self.verbose or not self.drop_warned):
                    self.drop_warned = True
                    print(f"[HeadLink] No ESP32 connected - head messages are dropped until it is"
                          + (f": {m}" if self.verbose else ""))
                return
            try:
                with self.write_lock:
                    ser.write(protocol.encode(m))
                if self.verbose and m.get("t") != "HEARTBEAT":
                    print(f"[HeadLink] -> ESP32  {m}")
            except Exception as e:
                print(f"[HeadLink] Write failed: {e}")
                self.drop_serial()
                return

    def drop_serial(self):
        with self.lock:
            ser, self.serial = self.serial, None
        if ser:
            try:
                ser.close()
            except Exception:
                pass

    def serial_loop(self):
        last_warn = None
        while True:
            port = self.discover("head")
            if not port:
                # once, then a reminder every 10 minutes (every 15 s with --verbose)
                if last_warn is None or time.monotonic() - last_warn > (15 if self.verbose else 600):
                    last_warn = time.monotonic()
                    print(f"[HeadLink] Waiting for the ESP32 (an Espressif USB device, VID {ESPRESSIF_VID:04X}"
                          f" - the S3's native USB port). Ports seen: {describe_ports()}")
                time.sleep(2)
                continue
            try:
                ser = open_serial(port, self.cfg["baud"])
            except Exception as e:
                print(f"[HeadLink] Can't open {port}: {e}")
                time.sleep(2)
                continue
            print(f"[HeadLink] ESP32 connected on {port}")
            self.drop_warned, last_warn = False, None
            last_status = 0.0
            with self.lock:
                self.serial, self.head_port = ser, port
            self.send_device(self.core.on_connect())
            buf = b""
            while True:
                with self.lock:
                    if self.serial is not ser:
                        break
                try:
                    chunk = ser.read(256)
                except Exception as e:
                    print(f"[HeadLink] ESP32 disconnected: {e}")
                    self.drop_serial()
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    msg = protocol.decode(line)
                    if msg is None:
                        if line.startswith(b"mouth:"):       # it was the mouth after all (missed the "?")
                            print(f"[HeadLink] {port} is the mouth, not the head controller - switching")
                            self.kinds[port] = "mouth"
                            self.drop_serial()
                            break
                        if self.verbose and line.strip():
                            print(f"[HeadLink] <- ESP32  (not JSON) {line[:120]!r}")
                        continue
                    if msg.get("t") == "EVENT":
                        print(f"[HeadLink] ESP32 event: {msg.get('what')}")
                    elif self.verbose and msg.get("t") == "STATUS" and time.monotonic() - last_status > 5:
                        last_status = time.monotonic()          # STATUS comes twice a second
                        print(f"[HeadLink] <- ESP32  {msg}")
                    back, out = self.core.from_device(msg)
                    self.send_device(back)
                    self.broadcast(out)
            with self.lock:
                self.head_port = None
            time.sleep(1)

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

    def mouth_loop(self):
        if self.cfg.get("mouth_port", "auto") in ("", "off"):
            return
        last_warn = None
        while True:
            port = self.discover("mouth")
            if not port:
                if self.verbose and (last_warn is None or time.monotonic() - last_warn > 15):
                    last_warn = time.monotonic()
                    print(f"[HeadLink] No mouth XIAO yet. Ports seen: {describe_ports()}")
                time.sleep(3)
                continue
            try:
                ser = open_serial(port, self.cfg["baud"])
            except Exception as e:
                print(f"[HeadLink] Can't open the mouth on {port}: {e}")
                time.sleep(3)
                continue
            print(f"[HeadLink] Mouth connected on {port}")
            with self.mouth_lock:
                self.mouth, self.mouth_port = ser, port
            self.send_mouth(self.core.mouth_on_connect())
            buf = b""
            while True:
                with self.mouth_lock:
                    if self.mouth is not ser:
                        break
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
                    if self.verbose and line.strip():
                        print(f"[HeadLink] <- mouth  {line.strip().decode(errors='replace')}")
            with self.mouth_lock:
                self.mouth_port = None
            self.kinds.pop(port, None)
            time.sleep(1)

    def heartbeat_loop(self):
        while True:
            self.send_device([self.core.heartbeat()])
            time.sleep(self.cfg["heartbeat_s"])

    # ---- local clients -----------------------------------------------------------------------
    def broadcast(self, msgs):
        data = b"".join(protocol.encode(m) for m in msgs)
        for c in list(self.clients):
            try:
                # non-blocking: a client that isn't reading (buffer full) is dropped, not waited on
                c.setblocking(False)
                c.sendall(data)
            except Exception:
                if c in self.clients:
                    self.clients.remove(c)
                print("[HeadLink] Dropped a client that stopped reading")
                try:
                    c.close()
                except OSError:
                    pass
            finally:
                try:
                    c.setblocking(True)
                except OSError:
                    pass

    def client_loop(self, conn: socket.socket):
        # a client that stops reading must not stall the link: give up on a send after 1 s
        conn.settimeout(1.0)
        peer = conn.getpeername()
        print(f"[HeadLink] Client connected {peer[0]}:{peer[1]}")
        # tell the newcomer the current settings (e.g. the brightness someone set earlier)
        if "CONFIG" in self.core.last:
            try:
                conn.sendall(protocol.encode(self.core.last["CONFIG"]))
            except OSError:
                pass
        self.clients.append(conn)
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
                    self.send_device(self.core.from_client(msg))
                    self.send_mouth(self.core.to_mouth(msg))
        except OSError:
            pass
        finally:
            if conn in self.clients:
                self.clients.remove(conn)
            conn.close()
            print(f"[HeadLink] Client disconnected {peer[0]}:{peer[1]}")

    def serve(self):
        threading.Thread(target=self.serial_loop, daemon=True).start()
        threading.Thread(target=self.mouth_loop, daemon=True).start()
        threading.Thread(target=self.heartbeat_loop, daemon=True).start()
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
    ap.add_argument("-v", "--verbose", action="store_true", help="print every message to and from the ESP32")
    args = ap.parse_args()
    try:
        Service(CONFIG["head"], verbose=args.verbose).serve()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
