"""
Head link service: the one program that talks to the piper-watch ESP32.

    python3 -m piper_head.link          (from src/; start_piper.sh starts it)
    python3 src/piper_head/link.py -v   verbose: print every message to and from the ESP32

* Finds the ESP32-S3 on USB (Espressif's vendor ID) or uses head.serial_port, and reconnects
  if it is unplugged or resets.
* Sends a HEARTBEAT every head.heartbeat_s (the ESP32 shows "offline" and, later, stops the
  motor if they stop).
* Listens on 127.0.0.1:head.port for local clients - the assistant, vision, tools - and forwards
  their FACE / CONFIG / LOOK messages to the ESP32. ESP32 STATUS / EVENT lines go to every client.
* Remembers the last FACE and CONFIG and re-sends them whenever the ESP32 (re)connects or
  reboots, so the ring is never stuck showing a stale state.
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
FORWARDED = {"FACE", "CONFIG", "LOOK"}


class HeadLink:
    """The link's logic, without any I/O: feed it lines, it returns what to send where."""

    def __init__(self):
        self.seq = 0
        self.last = {}                  # last FACE / CONFIG sent, by type
        self.status = None              # last STATUS from the ESP32

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
            return [msg]
        return []

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


def find_port(setting: str) -> str | None:
    if setting and setting != "auto":
        return setting
    from serial.tools import list_ports
    for p in list_ports.comports():
        if p.vid == ESPRESSIF_VID:
            return p.device
    return None


def describe_ports() -> str:
    """The serial ports that ARE there - for the 'no ESP32 found' message."""
    from serial.tools import list_ports
    ports = [f"{p.device} ({p.vid or 0:04X}:{p.pid or 0:04X} {p.description})" for p in list_ports.comports()]
    return ", ".join(ports) or "none"


class Service:
    def __init__(self, cfg: dict, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
        self.last_drop_warn = 0.0
        self.core = HeadLink()
        self.lock = threading.Lock()
        self.write_lock = threading.Lock()    # heartbeats and clients write from different threads
        self.serial = None
        self.clients: list[socket.socket] = []

    # ---- serial ------------------------------------------------------------------------------
    def send_device(self, msgs):
        with self.lock:
            ser = self.serial
        for m in msgs:
            if ser is None:
                # say so (at most every 5 s) instead of silently dropping what clients send
                if m.get("t") != "HEARTBEAT" and time.monotonic() - self.last_drop_warn > 5:
                    self.last_drop_warn = time.monotonic()
                    print(f"[HeadLink] No ESP32 connected - not sent: {m}")
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
        import serial
        last_warn = 0.0
        while True:
            port = find_port(self.cfg["serial_port"])
            if not port:
                if time.monotonic() - last_warn > 15:
                    last_warn = time.monotonic()
                    print(f"[HeadLink] Waiting for the ESP32 (an Espressif USB device, VID {ESPRESSIF_VID:04X}"
                          f" - the S3's native USB port). Ports seen: {describe_ports()}")
                time.sleep(2)
                continue
            try:
                # DTR on / RTS off BEFORE opening: on the S3's native USB those lines drive
                # reset/boot, and with the defaults the board stopped accepting writes (bench test)
                ser = serial.Serial()
                ser.port, ser.baudrate = port, self.cfg["baud"]
                ser.timeout, ser.write_timeout = 0.2, 1
                ser.dtr, ser.rts = True, False
                ser.open()
            except Exception as e:
                print(f"[HeadLink] Can't open {port}: {e}")
                time.sleep(2)
                continue
            print(f"[HeadLink] ESP32 connected on {port}")
            last_status = 0.0
            with self.lock:
                self.serial = ser
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
                c.sendall(data)
            except Exception:
                self.clients.remove(c)

    def client_loop(self, conn: socket.socket):
        self.clients.append(conn)
        peer = conn.getpeername()
        print(f"[HeadLink] Client connected {peer[0]}:{peer[1]}")
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
        except OSError:
            pass
        finally:
            if conn in self.clients:
                self.clients.remove(conn)
            conn.close()
            print(f"[HeadLink] Client disconnected {peer[0]}:{peer[1]}")

    def serve(self):
        threading.Thread(target=self.serial_loop, daemon=True).start()
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
