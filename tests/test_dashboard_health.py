"""
The dashboard's Health tab (piper_dashboard/health.py) and the monitor's report thinning.

Version 1.0.0 - created 2026-11-25, last modified 2026-11-25
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from piper_brain.config import CONFIG              # noqa: E402
from piper_dashboard import health                 # noqa: E402
from piper_health.monitor import thin              # noqa: E402


def sample(t, cpu=10.0, vision=150.0, temp=60.0):
    return {"t": t, "system_cpu": cpu, "mem_available_mb": 9000, "load": 1.0, "temps": {"tj": temp},
            "gpu": None, "procs": {"vision": {"cpu": vision, "rss_mb": 200.0}}}


def report_body(name="orin-nx_20261125_120000", n=10):
    return {"name": name, "machine": {"host": "orin-nx", "cores": 8}, "title": "test", "started": "2026-11-25T12:00:00",
            "seconds": n, "processes": {"vision": {"avg": 150.0, "p95": 160.0, "peak": 170.0}},
            "system": {"cpu": {"avg": 30.0, "peak": 50.0}, "mem_available_min_mb": 9000, "temps_max": {"tj": 70.0},
                       "gpu": None}, "marks": [], "text": "report text", "series": [sample(i) for i in range(n)]}


def test_thin_keeps_short_series_and_shrinks_long_ones():
    short = [sample(i) for i in range(50)]
    assert thin(short, 400) is short
    long = [sample(i, temp=60 + (i == 777) * 20) for i in range(2000)]
    out = thin(long, 400)
    assert len(out) == 400
    assert max(x["temps"]["tj"] for x in out) == 80          # a brief hot spike survives thinning
    assert out[0]["procs"]["vision"]["cpu"] == 150.0


def test_receive_report_and_overview(tmp_path, monkeypatch):
    monkeypatch.setitem(CONFIG["health"], "history_dir", str(tmp_path))
    assert health.receive_report(report_body())["ok"]
    assert (tmp_path / "orin-nx_20261125_120000.txt").read_text().startswith("report text")
    ov = health.overview()
    (r,) = ov["reports"]
    assert r["host"] == "orin-nx" and r["top"] == "vision" and r["temp_max"] == 70.0
    d = health.report("orin-nx_20261125_120000")
    assert len(d["series"]) == 10


def test_bad_reports_and_names_are_refused(tmp_path, monkeypatch):
    monkeypatch.setitem(CONFIG["health"], "history_dir", str(tmp_path))
    assert not health.receive_report({"name": "x"})["ok"]                 # no machine: not a report
    health.receive_report(report_body(name="../../etc/evil"))
    assert all(p.parent == tmp_path for p in tmp_path.rglob("*"))         # stayed in the folder
    assert health.report("../nothing") is None


def test_live_snapshot_keeps_newest_per_host():
    health.LIVE.clear()
    health.receive_live({"host": "orin-nx", "system_cpu": 10})
    health.receive_live({"host": "orin-nx", "system_cpu": 20})
    assert not health.receive_live({"system_cpu": 5})["ok"]
    live = health.overview()["live"]
    assert len(live) == 1 and live[0]["system_cpu"] == 20 and live[0]["age_s"] < 5
