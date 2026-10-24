"""
The pan/tilt servos: two Feetech STS3215s on a Waveshare bus servo adapter (USB, CH343 chip).

    python3 src/piper_head/servos.py            check: find the adapter, read both servos

Angles are SERVO degrees in the robot convention (pan + = left, tilt + = the head up); each
servo's centre and direction were set on the test stand (piper-watch tools/servo_test), and the
tilt's angle limits are stored in the servo itself. Never raises once constructed: if the adapter
is missing or a servo stops answering, moves are dropped and `ok` goes False.
"""
import glob
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ADDR_TORQUE, ADDR_ACC, ADDR_GOAL, ADDR_POS = 40, 41, 42, 56
STEPS_PER_DEG = 4096 / 360
CENTRE = 2048
CH343 = (0x1A86, 0x55D3)        # the Waveshare adapter's USB-serial chip - NOT the face's CH340 (1A86:7523)


def find_port() -> str | None:
    """The Waveshare adapter's serial port (its CH343's exact USB ID), or None."""
    try:
        from serial.tools import list_ports
        for p in sorted(list_ports.comports(), key=lambda p: p.device):
            if (p.vid, p.pid) == CH343:
                return p.device
    except ImportError:
        pass
    hits = sorted(glob.glob("/dev/serial/by-id/*1a86_USB_Single_Serial*"))   # the CH343's by-id name
    return hits[0] if hits else None


class PanTilt:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.ids = {"pan": cfg["pan_id"], "tilt": cfg["tilt_id"]}
        self.dir = {"pan": cfg["pan_dir"], "tilt": cfg["tilt_dir"]}
        self.ok = False
        self.port = self.pk = None
        try:
            from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler
            self._success = COMM_SUCCESS
            name = cfg["port"] if cfg["port"] != "auto" else find_port()
            if name is None:
                print("[Servos] No servo adapter found (CH343 USB)")
                return
            self.port, self.pk = PortHandler(name), PacketHandler(0)
            if not (self.port.openPort() and self.port.setBaudRate(cfg["baud"])):
                print(f"[Servos] Can't open {name}")
                return
            missing = [k for k, sid in self.ids.items() if self.pk.ping(self.port, sid)[1] != COMM_SUCCESS]
            if missing:
                print(f"[Servos] On {name}, but no answer from: {', '.join(missing)}")
                return
            self.ok = True
            print(f"[Servos] Pan (ID {self.ids['pan']}) and tilt (ID {self.ids['tilt']}) on {name}")
        except ImportError:
            print("[Servos] scservo_sdk missing: pip install feetech-servo-sdk")
        except Exception as e:                                   # never take the caller down
            print(f"[Servos] Disabled: {e}")

    def _deg_to_pos(self, axis, deg):
        return max(0, min(4095, round(CENTRE + self.dir[axis] * deg * STEPS_PER_DEG)))

    def position(self) -> tuple[float, float] | None:
        """(pan, tilt) where the servos are now, or None."""
        if not self.ok:
            return None
        out = []
        for axis in ("pan", "tilt"):
            v, res, _ = self.pk.read2ByteTxRx(self.port, self.ids[axis], ADDR_POS)
            if res != self._success:
                return None
            out.append(self.dir[axis] * (v - CENTRE) / STEPS_PER_DEG)
        return out[0], out[1]

    def move(self, pan: float, tilt: float, speed: int | None = None, acc: int | None = None):
        """Start both servos toward (pan, tilt) - returns at once; they get there on their own."""
        if not self.ok:
            return
        speed = speed or self.cfg["speed"]
        acc = acc or self.cfg["acc"]
        try:
            for axis, deg in (("pan", pan), ("tilt", tilt)):
                sid, pos = self.ids[axis], self._deg_to_pos(axis, deg)
                self.pk.write1ByteTxRx(self.port, sid, ADDR_ACC, acc)
                self.pk.writeTxRx(self.port, sid, ADDR_GOAL, 6, [pos & 0xFF, pos >> 8, 0, 0, speed & 0xFF, speed >> 8])
        except Exception as e:
            print(f"[Servos] Move failed: {e}")

    def torque(self, on: bool):
        if not self.ok:
            return
        for sid in self.ids.values():
            self.pk.write1ByteTxRx(self.port, sid, ADDR_TORQUE, 1 if on else 0)

    def close(self):
        if self.port is not None:
            self.port.closePort()


if __name__ == "__main__":
    from piper_brain.config import CONFIG
    pt = PanTilt(CONFIG["vision"]["follow"])
    print("position (pan, tilt):", pt.position())
    pt.close()
