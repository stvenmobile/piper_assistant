"""piper_health: the /proc readers, the monitor's sampling and summary, and the command line.

Version 1.0.0 - created 2026-11-24, last modified 2026-11-24
"""
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if not Path("/proc/stat").exists():
    pytest.skip("piper_health reads Linux /proc", allow_module_level=True)

from piper_health import monitor, probe  # noqa: E402

BUSY = "import time\nend = time.time() + {s}\nwhile time.time() < end: pass\n"


def test_probe_readers():
    busy, total = probe.cpu_times()
    assert 0 <= busy <= total
    mem = probe.memory()
    assert mem["total_mb"] > mem["available_mb"] > 0
    me = probe.process(__import__("os").getpid())
    assert me["ticks"] >= 0 and me["rss_mb"] > 1
    assert probe.process(999999999) is None
    info = probe.machine_info()
    assert info["cores"] >= 1 and info["memory_gb"] > 0
    assert isinstance(probe.temperatures(), dict)


def test_stats():
    assert monitor.stats([]) == {"avg": None, "p95": None, "peak": None}
    s = monitor.stats(list(range(1, 101)))
    assert s["avg"] == 50.5 and s["peak"] == 100 and 94 <= s["p95"] <= 96


def test_monitor_sees_a_busy_process_by_its_command_line():
    p = subprocess.Popen([sys.executable, "-c", BUSY.format(s=2.5) + "# health-test-marker"])
    try:
        m = monitor.Monitor(services={"busy": "health-test-marker"}, interval=0.25).start()
        time.sleep(1.5)
        m.stop()
    finally:
        p.wait()
    s = m.summary()
    assert "busy" in s["processes"]
    assert s["processes"]["busy"]["avg"] > 50            # one core, flat out
    assert s["processes"]["busy"]["cpu_seconds"] > 0.5
    assert s["system"]["cpu"]["avg"] is not None


def test_monitor_counts_a_workloads_children():
    parent = ("import subprocess, sys\n"
              f"subprocess.run([sys.executable, '-c', {BUSY.format(s=1.5)!r}])\n")
    p = subprocess.Popen([sys.executable, "-c", parent])
    try:
        m = monitor.Monitor(services={}, pids={"workload": p.pid}, interval=0.25).start()
        p.wait()
        m.stop()
    finally:
        p.wait()
    assert m.summary()["processes"]["workload"]["cpu_seconds"] > 0.7


def test_command_line_run_reports_marks(tmp_path):
    work = "import time\nfor _ in range(2):\n    time.sleep(0.3)\n    print('cycle done  [12 s]', flush=True)\n"
    monitor.main(["--only", "--interval", "0.2", "--out", str(tmp_path), "--title", "test",
                  "--run", "--", sys.executable, "-c", work])
    txt = next(tmp_path.glob("*.txt")).read_text()
    assert "workload marks: 2, avg 12.0 s" in txt
    assert "workload" in txt and list(tmp_path.glob("*.json")) and list(tmp_path.glob("*_workload.log"))
