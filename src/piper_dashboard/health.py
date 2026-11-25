"""
The dashboard's Health tab: what piper_health's monitor sends in, and the reports on disk.

    receive_live(body)     POST /api/health/live    a snapshot of the last minute, from a running monitor
    receive_report(body)   POST /api/health/report  a finished report (series already thinned to ~400 points)
    overview()             GET  /api/health         the latest snapshot per machine + the report list
    report(name)           GET  /api/health/report?name=...   one report, with its series

Snapshots are kept in memory only (the newest per machine). Reports are saved as JSON in
health.history_dir - the same folder the monitor writes to, so reports made on this machine show up
too. Nothing here touches the research memory, so it works before the memory database exists.

Version 1.0.0 - created 2026-11-25, last modified 2026-11-25
History:
    1.0.0  2026-11-25  First version.
"""
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from piper_brain.config import CONFIG, ROOT_DIR

MAX_POINTS = 400
LIVE: dict[str, dict] = {}                 # host -> its newest snapshot
_lock = threading.Lock()
_cache: dict[str, tuple[float, dict]] = {}  # report name -> (file mtime, its list entry)
_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def _dir() -> Path:
    d = Path(CONFIG["health"]["history_dir"])
    d = d if d.is_absolute() else ROOT_DIR / d
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe(name) -> str:
    return _SAFE.sub("_", str(name or ""))[:80]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def receive_live(body: dict) -> dict:
    host = _safe(body.get("host"))
    if not host:
        return {"ok": False, "error": "no host"}
    body["received_at"] = _now().isoformat(timespec="seconds")
    with _lock:
        LIVE[host] = body
    return {"ok": True}


def receive_report(body: dict) -> dict:
    name = _safe(body.get("name"))
    if not name or not isinstance(body.get("machine"), dict):
        return {"ok": False, "error": "not a health report"}
    series = body.get("series") or []
    if len(series) > MAX_POINTS * 2:                     # an old monitor that didn't thin it
        step = len(series) / MAX_POINTS
        body["series"] = [series[int(i * step)] for i in range(MAX_POINTS)]
    (_dir() / f"{name}.json").write_text(json.dumps(body), encoding="utf-8")
    if body.get("text"):
        (_dir() / f"{name}.txt").write_text(str(body["text"]) + "\n", encoding="utf-8")
    return {"ok": True, "name": name}


def _entry(path: Path) -> dict | None:
    """The report-list line for one report file (cached until the file changes)."""
    mtime = path.stat().st_mtime
    hit = _cache.get(path.stem)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or "machine" not in d:
        return None
    sysd = d.get("system") or {}
    temps = sysd.get("temps_max") or {}
    procs = d.get("processes") or {}
    top = max(procs.items(), key=lambda kv: kv[1].get("avg") or 0, default=(None, {}))
    e = {"name": path.stem, "host": d["machine"].get("host"), "cores": d["machine"].get("cores"),
         "title": d.get("title") or "", "started": d.get("started"), "seconds": d.get("seconds"),
         "cpu_avg": (sysd.get("cpu") or {}).get("avg"), "cpu_peak": (sysd.get("cpu") or {}).get("peak"),
         "mem_min_mb": sysd.get("mem_available_min_mb"), "temp_max": max(temps.values()) if temps else None,
         "gpu_avg": (sysd.get("gpu") or {}).get("avg"), "top": top[0], "top_cpu": top[1].get("avg"),
         "workload": d.get("workload"), "marks": len(d.get("marks") or [])}
    _cache[path.stem] = (mtime, e)
    return e


def overview(limit: int = 60) -> dict:
    now = _now()
    with _lock:
        live = [dict(v) for v in LIVE.values()]
    for v in live:
        v["age_s"] = (now - datetime.fromisoformat(v["received_at"])).total_seconds()
    files = sorted((p for p in _dir().glob("*.json") if not p.stem.endswith("_workload")),
                   key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    reports = [e for e in (_entry(p) for p in files) if e]
    return {"live": sorted(live, key=lambda v: v.get("host", "")), "reports": reports,
            "post_every_s": CONFIG["health"].get("post_every_s", 60)}


def report(name: str) -> dict | None:
    path = _dir() / f"{_safe(name)}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    series = d.get("series") or []
    if len(series) > MAX_POINTS:                         # a long local report: thin it for the page
        from piper_health.monitor import thin
        d["series"] = thin(series, MAX_POINTS)
    txt = path.with_suffix(".txt")
    if not d.get("text") and txt.exists():
        d["text"] = txt.read_text(encoding="utf-8")
    return d
