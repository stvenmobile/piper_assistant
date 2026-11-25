"""
Low-level readings for Piper's health checks - the machine and its processes, straight from /proc
and /sys (Linux only, no extra packages), so they work the same on the Orin and on piper-mind.

    cpu_times()            whole-machine CPU counters (busy, total) in clock ticks
    memory()               {"total_mb", "available_mb"}
    load_average()         the 1-minute load average
    temperatures()         {sensor name: degrees C} from the thermal zones (Jetson: cpu, gpu, tj, ...)
    gpu_load()             Jetson GPU load 0..100, or None where there is no such file
    process(pid)           {"pid", "ticks", "rss_mb", "name"} for one process, None if it has gone
    find_processes(pattern) pids whose command line contains the pattern
    machine_info()         CPU model, cores, memory, kernel - for comparing two machines

Every reader returns None (or an empty result) instead of raising when something is missing.

Version 1.0.0 - created 2026-11-24, last modified 2026-11-24
History:
    1.0.0  2026-11-24  First version (for the piper-mind -> Orin consolidation assessment).
"""
import os
import platform
from pathlib import Path

PROC = Path("/proc")
TICKS = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
GPU_LOAD_FILES = ("/sys/devices/platform/gpu.0/load", "/sys/devices/gpu.0/load",
                  "/sys/devices/platform/17000000.gpu/load")


def _read(path) -> str | None:
    try:
        return Path(path).read_text()
    except OSError:
        return None


def cpu_times() -> tuple[int, int] | None:
    """(busy ticks, total ticks) for the whole machine, all cores."""
    text = _read(PROC / "stat")
    if not text:
        return None
    vals = [int(v) for v in text.splitlines()[0].split()[1:]]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)          # idle + iowait
    total = sum(vals[:8])                                        # without guest (already in user)
    return total - idle, total


def cores() -> int:
    return os.cpu_count() or 1


def memory() -> dict | None:
    text = _read(PROC / "meminfo")
    if not text:
        return None
    kb = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if rest.strip():
            kb[key] = int(rest.split()[0])
    return {"total_mb": kb.get("MemTotal", 0) / 1024, "available_mb": kb.get("MemAvailable", 0) / 1024}


def load_average() -> float | None:
    text = _read(PROC / "loadavg")
    return float(text.split()[0]) if text else None


def temperatures() -> dict:
    """{zone type: degrees C}; duplicate types get a number added."""
    out = {}
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        name = (_read(zone / "type") or zone.name).strip()
        raw = _read(zone / "temp")
        if raw is None:
            continue
        try:
            c = int(raw.strip()) / 1000
        except ValueError:
            continue
        if c <= -40 or c > 150:                                  # unused zones report nonsense
            continue
        key, n = name, 2
        while key in out:
            key, n = f"{name}{n}", n + 1
        out[key] = round(c, 1)
    return out


def gpu_load() -> float | None:
    """Jetson GPU load in % (the kernel reports per mille), None elsewhere."""
    for f in GPU_LOAD_FILES:
        raw = _read(f)
        if raw and raw.strip().isdigit():
            return int(raw.strip()) / 10
    return None


def process(pid: int) -> dict | None:
    """CPU ticks used so far (user + system; children are counted separately, see children()) and
    resident memory."""
    stat = _read(PROC / str(pid) / "stat")
    if not stat:
        return None
    name = stat[stat.find("(") + 1:stat.rfind(")")]
    f = stat[stat.rfind(")") + 2:].split()
    ticks = int(f[11]) + int(f[12])                              # utime stime
    rss_mb = int(f[21]) * os.sysconf("SC_PAGE_SIZE") / 1048576
    return {"pid": pid, "ticks": ticks, "rss_mb": rss_mb, "name": name}


def cmdline(pid: int) -> str:
    raw = _read(PROC / str(pid) / "cmdline")
    return raw.replace("\0", " ").strip() if raw else ""


def children(pid: int) -> list[int]:
    """Direct and indirect child processes."""
    kids, out = {}, []
    for p in PROC.iterdir():
        if p.name.isdigit():
            stat = _read(p / "stat")
            if stat:
                ppid = int(stat[stat.rfind(")") + 2:].split()[1])
                kids.setdefault(ppid, []).append(int(p.name))
    todo = list(kids.get(pid, []))
    while todo:
        c = todo.pop()
        out.append(c)
        todo += kids.get(c, [])
    return out


def find_processes(pattern: str, exclude: set | None = None) -> list[int]:
    """Pids whose command line contains `pattern` (not this process, not its parents' shells)."""
    me = os.getpid()
    skip = (exclude or set()) | {me, os.getppid()}
    hits = []
    for p in PROC.iterdir():
        if p.name.isdigit() and int(p.name) not in skip and pattern in cmdline(int(p.name)):
            hits.append(int(p.name))
    return sorted(hits)


def machine_info() -> dict:
    model = ""
    for line in (_read(PROC / "cpuinfo") or "").splitlines():
        if line.lower().startswith(("model name", "hardware", "cpu part")) and not model:
            model = line.split(":", 1)[1].strip()
    jetson = (_read("/proc/device-tree/model") or "").strip("\0 \n")
    mem = memory() or {}
    return {"host": platform.node(), "cpu": jetson or model or platform.processor() or platform.machine(),
            "cores": cores(), "memory_gb": round(mem.get("total_mb", 0) / 1024, 1),
            "kernel": platform.release(), "arch": platform.machine()}
