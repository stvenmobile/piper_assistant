"""
Piper Assistant: dashboard-driven runtime with LangGraph Supervisor.

The Orin NX now runs headless, away from where the operator sits, so the
web dashboard (piper_web/dashboard.py) is the only active input/output
surface - voice I/O (piper_audio/listener.py, piper_audio/speaker.py,
piper_audio/kokoro_speaker.py) is intentionally left in the codebase
unused rather than removed, in case it's wanted again later.
"""

import os
import sys
import select
import time
import threading

os.environ["ORT_LOGGING_LEVEL"] = "3"

from werkzeug.serving import make_server

from piper_brain.config import CONFIG
from piper_brain.supervisor import PiperSupervisor, PiperBrainState
from piper_brain.state import AgentState, create_initial_state
from piper_brain.journal import ActivityJournal
from piper_web.dashboard import create_app

STARTUP_GRACE_PERIOD_SECONDS = 25.0

# How long to wait for a research trial that's mid-flight when shutdown is
# requested. idle_introspection_worker's daemon thread does real CUDA
# work (model forward passes, SVD); killing the process while that's in
# progress - which is what happened before, since nothing ever waited on
# this thread - can abort natively ("terminate called without an active
# exception") instead of exiting cleanly. Most shutdowns hit this thread
# between trials, where it returns almost immediately; this timeout only
# matters on the rare shutdown that lands mid-trial.
IDLE_THREAD_JOIN_TIMEOUT_SECONDS = 120.0


def idle_introspection_worker(supervisor: PiperSupervisor, state: AgentState, stop_event: threading.Event, start_time: float):
    """Background worker executing paced research trials whenever the dashboard is idle."""
    dummy_brain_state: PiperBrainState = {
        "mode": "ALONE",
        "active_user": None,
        "input_text": None,
        "output_text": None,
        "user_context": "",
        "messages": [],
        "introspection_topic": None,
        "introspection_result": None
    }

    # Wait out startup grace period before starting any heavy background GPU tasks
    while not stop_event.is_set() and (time.time() - start_time < STARTUP_GRACE_PERIOD_SECONDS):
        time.sleep(0.5)

    while not stop_event.is_set():
        if state["status"] == "IDLE":
            supervisor.autonomous_introspection_node(dummy_brain_state)

        for _ in range(30):
            if stop_event.is_set() or state["status"] != "IDLE":
                break
            time.sleep(0.1)


def keyboard_monitor(stop_event: threading.Event):
    """Background thread watching for 'q'/'quit'/'exit' on stdin.

    An alternative to Ctrl+C, which lands on whatever the main thread
    happens to be doing - previously that was Flask's own blocking
    app.run() loop, which has no clean programmatic shutdown hook in
    current Werkzeug, so Ctrl+C could only ever abort the process rather
    than let it wind down in order. main() now runs its own simple wait
    loop instead, so both this and Ctrl+C converge on the same graceful
    shutdown path (see main()'s stop_event handling below).
    """
    print("\n[Controls] Type 'q' and press [Enter] to exit cleanly.\n")
    while not stop_event.is_set():
        if select.select([sys.stdin], [], [], 0.5)[0]:
            line = sys.stdin.readline().strip().lower()
            if line in ("q", "quit", "exit"):
                print("\n[System] Exit signal received from keyboard.")
                stop_event.set()
                break


def main():
    print("--- Starting Piper Assistant (Dashboard Mode) ---")
    supervisor = PiperSupervisor()
    journal = ActivityJournal()

    state: AgentState = create_initial_state()
    state["status"] = "IDLE"

    stop_event = threading.Event()
    start_time = time.time()

    journal.log("SYSTEM", "Piper assistant runtime initialized in dashboard mode (voice I/O inactive).")

    idle_thread = threading.Thread(
        target=idle_introspection_worker,
        args=(supervisor, state, stop_event, start_time),
        daemon=True
    )
    idle_thread.start()

    kb_thread = threading.Thread(target=keyboard_monitor, args=(stop_event,), daemon=True)
    kb_thread.start()

    dashboard_cfg = CONFIG.get("dashboard", {})
    host = dashboard_cfg.get("host", "0.0.0.0")
    port = dashboard_cfg.get("port", 8080)

    app = create_app(supervisor, state, journal)
    # make_server (not app.run) so the server can be stopped cleanly from
    # another thread via server.shutdown() once shutdown is requested,
    # instead of only being interruptible by whatever landed on it.
    server = make_server(host, port, app, threaded=True)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"[Dashboard] Serving on http://{host}:{port} (Ctrl+C or type 'q' + Enter to stop)")

    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        journal.log("SHUTDOWN", "SIGINT caught.")
        stop_event.set()

    print("[Shutdown] Stopping dashboard server...")
    server.shutdown()
    server_thread.join(timeout=10)

    print("[Shutdown] Waiting for any in-flight research trial to finish...")
    idle_thread.join(timeout=IDLE_THREAD_JOIN_TIMEOUT_SECONDS)
    if idle_thread.is_alive():
        print("[Shutdown] Idle thread did not finish within timeout - exiting anyway.")

    journal.log("SYSTEM", "Piper assistant terminated.")
    print("--- Piper Assistant Terminated ---")


if __name__ == "__main__":
    main()
