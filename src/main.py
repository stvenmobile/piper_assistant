"""
Piper Assistant: dashboard-driven runtime with LangGraph Supervisor.

The Orin NX now runs headless, away from where the operator sits, so the
web dashboard (piper_web/dashboard.py) is the only active input/output
surface - voice I/O (piper_audio/listener.py, piper_audio/speaker.py,
piper_audio/kokoro_speaker.py) is intentionally left in the codebase
unused rather than removed, in case it's wanted again later.
"""

import os
import time
import threading

os.environ["ORT_LOGGING_LEVEL"] = "3"

from piper_brain.config import CONFIG
from piper_brain.supervisor import PiperSupervisor, PiperBrainState
from piper_brain.state import AgentState, create_initial_state
from piper_brain.journal import ActivityJournal
from piper_web.dashboard import create_app

STARTUP_GRACE_PERIOD_SECONDS = 25.0


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

    dashboard_cfg = CONFIG.get("dashboard", {})
    host = dashboard_cfg.get("host", "0.0.0.0")
    port = dashboard_cfg.get("port", 8080)

    app = create_app(supervisor, state, journal)
    print(f"[Dashboard] Serving on http://{host}:{port} (Ctrl+C to stop)")

    try:
        app.run(host=host, port=port, threaded=True)
    except KeyboardInterrupt:
        journal.log("SHUTDOWN", "SIGINT caught.")
    finally:
        stop_event.set()
        journal.log("SYSTEM", "Piper assistant terminated.")
        print("--- Piper Assistant Terminated ---")


if __name__ == "__main__":
    main()
