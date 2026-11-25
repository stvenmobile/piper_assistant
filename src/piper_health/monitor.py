"""
Piper's health monitor: how hard each of her services is working, and what is left of the machine.

    python3 src/piper_health/monitor.py --seconds 600
        watch Piper's services (health.services) for 10 minutes, then report
    python3 src/piper_health/monitor.py --run -- python3 src/piper_research/service.py --now --cycles 5
        run a workload, watch it (and its child processes) beside the services, then report;
        durations printed by the workload as "[123 s]" are collected too (--mark to change the pattern)

While it runs it sends a live snapshot to the dashboard every health.post_every_s (60 s), and the
finished report at the end (health.post_url, by default research.mind_url = piper-mind's dashboard;
--no-post to skip). Sending never holds the monitor up: if the dashboard can't be reached the report is
still saved here.

The report (printed, and saved to health.history_dir as <host>_<date>_<time>.txt + .json):
    machine     CPU, cores, memory - for comparing two machines
    processes   per service: average / 95th percentile / peak CPU (100% = one full core), CPU seconds
                used, peak memory
    system      whole-machine CPU (100% = every core busy), lowest free memory, load average,
                hottest reading per temperature sensor, GPU load (Jetson)
    marks       the workload's own durations (e.g. research cycle times) and CPU seconds per mark

As a library - the same Monitor can run inside Piper later, so she can notice when a new task
crowds out the others:
    m = Monitor(services=CONFIG["health"]["services"]).start()
    ...  m.latest()   the newest sample
    ...  m.stop(); m.summary()

Linux only (reads /proc and /sys - see probe.py); needs no extra packages.

Version 1.1.0 - created 2026-11-24, last modified 2026-11-25
History:
    1.1.0  2026-11-25  Reports to the dashboard: a live snapshot every minute and the finished report
                       (series thinned to <= 400 points); --post URL, --no-post.
    1.0.0  2026-11-24  First version: services, a watched workload, system load, temperatures, GPU.
"""
import argparse
import json
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from piper_health import probe

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERVICES = {                      # name -> a part of the command line that identifies it
    "assistant": "src/main.py",
    "vision": "piper_vision/service.py",
    "head_link": "piper_head/link.py",
    "research": "piper_research/service.py",
    "dashboard": "piper_dashboard/server.py",
}
REDISCOVER_S = 10.0                       # services can restart: look for their pids again this often
MAX_POINTS = 400                          # the series sent to the dashboard is thinned to this many points


def _config() -> dict:
    try:
        from piper_brain.config import CONFIG
        return CONFIG.get("health", {})
    except Exception:
        return {}


class Monitor:
    """Samples the machine and the watched processes every `interval` seconds in a thread."""

    def __init__(self, services: dict | None = None, pids: dict | None = None, interval: float = 1.0,
                 keep: int = 86400):
        self.services = dict(DEFAULT_SERVICES if services is None else services)
        self.fixed = dict(pids or {})                 # name -> pid (with its children), e.g. a workload
        self.interval = interval
        self.samples = deque(maxlen=keep)
        self.cpu_seconds = {}                         # name -> CPU seconds used while watched
        self.peak_rss = {}
        self._pids = {}                               # name -> [pid, ...]
        self._found_at = 0.0
        self._prev_proc = {}                          # pid -> ticks
        self._prev_cpu = None
        self._prev_t = None
        self._stop = threading.Event()
        self._thread = None
        self.started = self.ended = None

    # ---- sampling ------------------------------------------------------------------------------
    def _discover(self, now: float):
        if now - self._found_at < REDISCOVER_S and self._pids:
            return
        self._found_at = now
        fixed_pids = set()
        for name, pid in self.fixed.items():
            fixed_pids |= {pid, *probe.children(pid)}
        pids = {name: [p for p in probe.find_processes(pat) if p not in fixed_pids]
                for name, pat in self.services.items()}
        for name, pid in self.fixed.items():
            pids[name] = [pid] + probe.children(pid)
        self._pids = {k: v for k, v in pids.items() if v}

    def sample(self) -> dict | None:
        """Take one sample (the first call only sets the baseline and returns None)."""
        now = time.monotonic()
        if self.fixed:                                # a workload's children come and go: always refresh
            self._found_at = 0.0
        self._discover(now)
        cpu = probe.cpu_times()
        procs = {}
        for name, pids in self._pids.items():
            ticks, rss, alive = 0, 0.0, 0
            for pid in pids:
                p = probe.process(pid)
                if not p:
                    continue
                alive += 1
                prev = self._prev_proc.get(pid)
                if prev is not None and p["ticks"] >= prev:
                    ticks += p["ticks"] - prev
                self._prev_proc[pid] = p["ticks"]
                rss += p["rss_mb"]
            if alive:
                procs[name] = {"ticks": ticks, "rss_mb": rss}
        first = self._prev_t is None
        dt = None if first else now - self._prev_t
        out = None
        if not first and dt > 0:
            sys_cpu = None
            if cpu and self._prev_cpu:
                busy, total = cpu[0] - self._prev_cpu[0], cpu[1] - self._prev_cpu[1]
                sys_cpu = 100.0 * busy / total if total > 0 else None
            mem = probe.memory() or {}
            out = {"t": round(now - self.started_mono, 2), "system_cpu": sys_cpu,
                   "mem_available_mb": round(mem.get("available_mb", 0)), "load": probe.load_average(),
                   "temps": probe.temperatures(), "gpu": probe.gpu_load(), "procs": {}}
            for name, v in procs.items():
                secs = v["ticks"] / probe.TICKS
                out["procs"][name] = {"cpu": round(100.0 * secs / dt, 1), "rss_mb": round(v["rss_mb"], 1)}
                self.cpu_seconds[name] = self.cpu_seconds.get(name, 0.0) + secs
                self.peak_rss[name] = max(self.peak_rss.get(name, 0.0), v["rss_mb"])
            self.samples.append(out)
        self._prev_cpu, self._prev_t = cpu, now
        return out

    def start(self):
        self.started, self.started_mono = datetime.now(), time.monotonic()
        self.sample()

        def run():
            while not self._stop.wait(self.interval):
                try:
                    self.sample()
                except Exception as e:                # a health check must never take Piper down
                    print(f"[Health] {e}")
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval * 2 + 1)
        self.sample()                                 # a last sample up to now
        self.ended = datetime.now()
        return self

    def latest(self) -> dict | None:
        return self.samples[-1] if self.samples else None

    # ---- summary -------------------------------------------------------------------------------
    def summary(self) -> dict:
        s = list(self.samples)
        names = sorted({n for x in s for n in x["procs"]} | set(self.cpu_seconds))
        procs = {}
        for n in names:
            vals = [x["procs"][n]["cpu"] for x in s if n in x["procs"]]
            procs[n] = {**stats(vals), "cpu_seconds": round(self.cpu_seconds.get(n, 0.0), 1),
                        "peak_rss_mb": round(self.peak_rss.get(n, 0.0), 1), "seen_s": len(vals) * self.interval}
        temps = {}
        for x in s:
            for k, v in (x.get("temps") or {}).items():
                temps[k] = max(temps.get(k, v), v)
        mem = [x["mem_available_mb"] for x in s if x.get("mem_available_mb")]
        gpu = [x["gpu"] for x in s if x.get("gpu") is not None]
        load = [x["load"] for x in s if x.get("load") is not None]
        return {
            "machine": probe.machine_info(),
            "started": self.started.isoformat(timespec="seconds") if self.started else None,
            "seconds": round(s[-1]["t"], 1) if s else 0,
            "samples": len(s),
            "processes": procs,
            "system": {"cpu": stats([x["system_cpu"] for x in s if x.get("system_cpu") is not None]),
                       "mem_available_min_mb": min(mem) if mem else None,
                       "load_max": max(load) if load else None,
                       "temps_max": temps, "gpu": stats(gpu) if gpu else None},
        }


def stats(vals: list) -> dict:
    """avg / p95 / peak of a list of numbers (all None when it's empty)."""
    if not vals:
        return {"avg": None, "p95": None, "peak": None}
    v = sorted(vals)
    p95 = v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))]
    return {"avg": round(statistics.mean(v), 1), "p95": round(p95, 1), "peak": round(v[-1], 1)}


def thin(series: list, n: int = MAX_POINTS) -> list:
    """At most n points: consecutive samples averaged (CPU, memory, GPU) or maxed (temperatures)."""
    if len(series) <= n:
        return series
    out, size = [], len(series) / n
    for i in range(n):
        chunk = series[int(i * size):int((i + 1) * size)] or [series[min(len(series) - 1, int(i * size))]]
        avg = lambda vals: round(sum(vals) / len(vals), 1) if vals else None
        temps = {}
        for x in chunk:
            for k, v in (x.get("temps") or {}).items():
                temps[k] = max(temps.get(k, v), v)
        procs = {}
        for name in {n2 for x in chunk for n2 in x["procs"]}:
            vals = [x["procs"][name] for x in chunk if name in x["procs"]]
            procs[name] = {"cpu": avg([v["cpu"] for v in vals]), "rss_mb": max(v["rss_mb"] for v in vals)}
        out.append({"t": chunk[-1]["t"],
                    "system_cpu": avg([x["system_cpu"] for x in chunk if x.get("system_cpu") is not None]),
                    "mem_available_mb": min((x["mem_available_mb"] for x in chunk if x.get("mem_available_mb")),
                                            default=None),
                    "load": max((x["load"] for x in chunk if x.get("load") is not None), default=None),
                    "temps": temps, "gpu": avg([x["gpu"] for x in chunk if x.get("gpu") is not None]),
                    "procs": procs})
    return out


def post(url: str, path: str, body: dict, timeout: float = 3.0) -> bool:
    """POST JSON to the dashboard; False (never an exception) if it can't be reached."""
    try:
        req = urllib.request.Request(url.rstrip("/") + path, json.dumps(body, default=str).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


class LivePoster:
    """Sends a snapshot of the last minute to the dashboard every `every` seconds, in a thread."""

    def __init__(self, mon: Monitor, url: str, meta: dict, every: float = 60.0):
        self.mon, self.url, self.meta, self.every = mon, url, meta, every
        self._stop = threading.Event()
        self.warned = False

    def snapshot(self) -> dict | None:
        s = list(self.mon.samples)
        if not s:
            return None
        recent = [x for x in s if x["t"] >= s[-1]["t"] - self.every] or s[-1:]
        avg = lambda vals: round(sum(vals) / len(vals), 1) if vals else None
        procs = {}
        for name in {n for x in recent for n in x["procs"]}:
            vals = [x["procs"][name] for x in recent if name in x["procs"]]
            procs[name] = {"cpu": avg([v["cpu"] for v in vals]), "rss_mb": vals[-1]["rss_mb"]}
        temps = s[-1].get("temps") or {}
        return {**self.meta, "machine": probe.machine_info(), "elapsed_s": s[-1]["t"], "window_s": self.every,
                "system_cpu": avg([x["system_cpu"] for x in recent if x.get("system_cpu") is not None]),
                "mem_available_mb": s[-1].get("mem_available_mb"), "load": s[-1].get("load"),
                "gpu": avg([x["gpu"] for x in recent if x.get("gpu") is not None]),
                "temp_max": max(temps.values()) if temps else None, "temps": temps, "procs": procs}

    def send(self, final: bool = False):
        snap = self.snapshot()
        if snap is None:
            return
        snap["running"] = not final
        if not post(self.url, "/api/health/live", snap) and not self.warned:
            self.warned = True
            print(f"[Health] Can't reach the dashboard at {self.url} - reports are still saved here")

    def start(self):
        def run():
            while not self._stop.wait(self.every):
                self.send()
        threading.Thread(target=run, daemon=True).start()
        return self

    def stop(self):
        self._stop.set()
        self.send(final=True)


def post_url(arg: str | None, cfg: dict) -> str:
    """Where to send: --post, else health.post_url, else research.mind_url; "" = nowhere."""
    if arg is not None:
        return arg
    if cfg.get("post_url") is not None:
        return cfg["post_url"]
    try:
        from piper_brain.config import CONFIG
        return CONFIG.get("research", {}).get("mind_url") or ""
    except Exception:
        return ""


def report(summary: dict, marks: list | None = None, title: str = "") -> str:
    m, sysd = summary["machine"], summary["system"]
    f = lambda x: "-" if x is None else (f"{x:.1f}" if isinstance(x, float) else str(x))
    lines = [f"Piper health report{(' - ' + title) if title else ''}",
             f"{m['host']}: {m['cpu']}, {m['cores']} cores, {m['memory_gb']} GB ({m['arch']}, kernel {m['kernel']})",
             f"started {summary['started']}, {summary['seconds']:.0f} s, {summary['samples']} samples", "",
             f"{'process':<14}{'cpu avg %':>10}{'p95 %':>8}{'peak %':>8}{'cpu s':>9}{'peak MB':>9}   (100% = one core)",
             "-" * 72]
    for n, p in summary["processes"].items():
        lines.append(f"{n:<14}{f(p['avg']):>10}{f(p['p95']):>8}{f(p['peak']):>8}{f(p['cpu_seconds']):>9}"
                     f"{f(p['peak_rss_mb']):>9}")
    if not summary["processes"]:
        lines.append("(none of the watched processes was running)")
    c = sysd["cpu"]
    lines += ["", f"system CPU (100% = all {m['cores']} cores): avg {f(c['avg'])}%, p95 {f(c['p95'])}%,"
                  f" peak {f(c['peak'])}%",
              f"lowest free memory: {f(sysd['mem_available_min_mb'])} MB; highest load average: {f(sysd['load_max'])}"]
    if sysd["gpu"]:
        g = sysd["gpu"]
        lines.append(f"GPU: avg {f(g['avg'])}%, peak {f(g['peak'])}%")
    if sysd["temps_max"]:
        lines.append("hottest: " + ", ".join(f"{k} {v:.0f} C" for k, v in sysd["temps_max"].items()))
    if marks:
        work = summary["processes"].get("workload", {})
        lines += ["", f"workload marks: {len(marks)}, avg {statistics.mean(marks):.1f} s,"
                      f" min {min(marks):.1f} s, max {max(marks):.1f} s"]
        if work.get("cpu_seconds") is not None:
            lines.append(f"workload CPU per mark: {work['cpu_seconds'] / len(marks):.1f} CPU seconds")
    return "\n".join(lines)


def run_workload(cmd: list, mon: Monitor, mark: re.Pattern, log_path: Path) -> tuple[int, list]:
    """Run the workload with the monitor watching it; echo and save its output, collect marks."""
    marks = []
    import resource
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        mon.fixed["workload"] = proc.pid
        mon._found_at = 0.0
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
                for mm in mark.finditer(line):
                    marks.append(float(mm.group(1)))
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait(timeout=10)
            print("\n(workload stopped)")
    after = resource.getrusage(resource.RUSAGE_CHILDREN)   # exact: sampling misses the last moments
    mon.cpu_seconds["workload"] = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    mon.peak_rss["workload"] = max(mon.peak_rss.get("workload", 0.0), after.ru_maxrss / 1024)
    return proc.returncode, marks


def main(argv=None):
    cfg = _config()
    ap = argparse.ArgumentParser(description="Piper's health monitor (see the module docstring)")
    ap.add_argument("--seconds", type=float, help="watch for this long (without --run; Ctrl-C ends early)")
    ap.add_argument("--interval", type=float, default=cfg.get("interval_s", 1.0))
    ap.add_argument("--watch", action="append", default=[], metavar="NAME=PATTERN",
                    help="watch another process (part of its command line); repeatable")
    ap.add_argument("--only", action="store_true", help="watch only --watch / --run, not Piper's services")
    ap.add_argument("--mark", default=r"\[(\d+(?:\.\d+)?) s\]", help="regex for durations in the workload's output")
    ap.add_argument("--title", default="", help="a note for the report, e.g. 'piper-mind baseline'")
    ap.add_argument("--out", default=cfg.get("history_dir", "data/health"), help="where reports are saved")
    ap.add_argument("--post", default=None, metavar="URL",
                    help="the dashboard to report to (default health.post_url, else research.mind_url)")
    ap.add_argument("--no-post", action="store_true", help="don't send anything to the dashboard")
    ap.add_argument("--run", action="store_true", help="the workload: everything after '--run --'")
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = None
    if "--run" in argv:                                  # everything after --run (and an optional --) is the workload
        i = argv.index("--run")
        cmd = argv[i + 1:]
        if cmd[:1] == ["--"]:
            cmd = cmd[1:]
        argv = argv[:i]
    a = ap.parse_args(argv)

    services = {} if a.only else dict(cfg.get("services") or DEFAULT_SERVICES)
    for w in a.watch:
        name, _, pat = w.partition("=")
        services[name] = pat or name
    out_dir = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{probe.machine_info()['host']}_{datetime.now():%Y%m%d_%H%M%S}"

    mon = Monitor(services=services, interval=a.interval).start()
    url = "" if a.no_post else post_url(a.post, cfg)
    meta = {"host": probe.machine_info()["host"], "name": stamp, "title": a.title,
            "started": mon.started.isoformat(timespec="seconds"), "interval_s": a.interval,
            "workload": " ".join(cmd) if cmd else None, "until_s": None if cmd else (a.seconds or 60.0)}
    live = LivePoster(mon, url, meta, float(cfg.get("post_every_s", 60))).start() if url else None
    if url:
        print(f"[Health] reporting to the dashboard at {url}")
    marks, rc = [], None
    try:
        if cmd:
            print(f"[Health] running: {' '.join(cmd)}")
            rc, marks = run_workload(cmd, mon, re.compile(a.mark), out_dir / f"{stamp}_workload.log")
        else:
            seconds = a.seconds or 60.0
            print(f"[Health] watching for {seconds:.0f} s (Ctrl-C ends early) ...")
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n(stopped)")
    mon.stop()
    if live:
        live.stop()
    summ = mon.summary()
    summ.update({"title": a.title, "workload": " ".join(cmd) if cmd else None, "returncode": rc,
                 "marks": marks})
    text = report(summ, marks, a.title)
    print("\n" + text)
    (out_dir / f"{stamp}.txt").write_text(text + "\n", encoding="utf-8")
    (out_dir / f"{stamp}.json").write_text(json.dumps({**summ, "series": list(mon.samples)}, indent=1), encoding="utf-8")
    print(f"\nsaved to {out_dir / stamp}.txt (+ .json)")
    if url:
        sent = post(url, "/api/health/report", {**summ, "name": stamp, "text": text, "series": thin(list(mon.samples))},
                    timeout=10.0)
        print("sent to the dashboard" if sent else f"(couldn't send the report to {url})")


if __name__ == "__main__":
    main()
