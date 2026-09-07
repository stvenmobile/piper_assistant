"""
Piper Assistant: State-driven interactive runtime loop with LangGraph Supervisor.
"""

import sys
import os
import time
import select
import threading

os.environ["ORT_LOGGING_LEVEL"] = "3"

from langchain_core.messages import HumanMessage, AIMessage

from piper_audio.listener import PiperListener
from piper_brain.config import CONFIG
from piper_brain.quick_responder import QuickResponder
from piper_brain.supervisor import PiperSupervisor, PiperBrainState
from piper_brain.state import AgentState, create_initial_state, append_and_truncate_message
from piper_brain.journal import ActivityJournal

ENGAGED_TIMEOUT_SECONDS = 20.0
STARTUP_GRACE_PERIOD_SECONDS = 25.0
running = True

# TTS Engine Selection
tts_engine = CONFIG.get("audio", {}).get("tts_engine", "piper").lower()
if tts_engine == "kokoro":
    from piper_audio.kokoro_speaker import KokoroSpeaker
    speaker = KokoroSpeaker()
else:
    from piper_audio.speaker import PiperSpeaker
    speaker = PiperSpeaker()


def keyboard_monitor(journal: ActivityJournal):
    """Background thread watching for 'q' or 'exit' on stdin."""
    global running
    print("\n[Controls] Type 'q' and press [Enter] to exit cleanly.\n")
    while running:
        if select.select([sys.stdin], [], [], 0.5)[0]:
            line = sys.stdin.readline().strip().lower()
            if line in ("q", "quit", "exit"):
                print("\n[System] Exit signal received from keyboard.")
                journal.log("SHUTDOWN", "Operator requested exit via keyboard CLI.")
                running = False
                break


def idle_introspection_worker(supervisor: PiperSupervisor, state: AgentState, stop_event: threading.Event, start_time: float):
    """Background worker executing paced soft-prompt bursts during IDLE mode after startup grace period."""
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
    global running
    print("--- Starting Piper Assistant ---")
    listener = PiperListener()
    quick_responder = QuickResponder()
    supervisor = PiperSupervisor()
    journal = ActivityJournal()

    state: AgentState = create_initial_state()
    # Start in ENGAGED mode so the user can speak immediately without wake word
    state["status"] = "ENGAGED"
    state["last_interaction_time"] = time.time()
    
    stop_event = threading.Event()
    start_time = time.time()

    journal.log("SYSTEM", "Piper assistant runtime initialized.", "Inactivity timeout: 20s | Memory window: 8 turns")

    # Start CLI and background research threads
    kb_thread = threading.Thread(target=keyboard_monitor, args=(journal,), daemon=True)
    kb_thread.start()

    idle_thread = threading.Thread(
        target=idle_introspection_worker,
        args=(supervisor, state, stop_event, start_time),
        daemon=True
    )
    idle_thread.start()

    # Spoken startup announcement
    print("[Speaker] Announcing system online...")
    speaker.speak("Piper assistant is online and ready.")

    while running:
        try:
            # 1. Inactivity timeout check
            now = time.time()
            if state["status"] == "ENGAGED" and (now - state["last_interaction_time"] > ENGAGED_TIMEOUT_SECONDS):
                print(f"\n[State Transition] ENGAGED -> IDLE ({ENGAGED_TIMEOUT_SECONDS}s inactivity reached)")
                journal.log("STATE", "ENGAGED -> IDLE", f"Inactivity window exceeded ({ENGAGED_TIMEOUT_SECONDS}s).")
                state["status"] = "IDLE"

            # 2. State-dependent listening strategy
            if state["status"] == "IDLE":
                raw_text = listener.listen_for_wake_word_and_command(max_command_duration=6.0)
                if not raw_text:
                    continue

                print("\n[State Transition] IDLE -> ENGAGED (Wake word detected)")
                journal.log("STATE", "IDLE -> ENGAGED", f"Wake-word triggered with: '{raw_text}'")
                speaker.play_chime()
                state["status"] = "ENGAGED"
                state["last_interaction_time"] = time.time()

                user_text = listener._strip_wake_word(raw_text)
                if not user_text:
                    user_text = raw_text
            else:
                print("[Listener] In conversation (listening for command or follow-up)...")
                user_text = listener.listen_command_window(max_duration=6.0, silence_timeout=1.0)
                if not user_text:
                    continue
                state["last_interaction_time"] = time.time()

            if not running or not user_text:
                continue

            print(f"\n[User]: {user_text}")
            append_and_truncate_message(state, HumanMessage(content=user_text))

            # 3. Fast-Path Local Intent Check
            quick_reply = quick_responder.match(user_text)
            if quick_reply:
                print(f"[Piper (Local)]: {quick_reply}")
                journal.log("INTENT_LOCAL", f"Matched '{user_text}'", f"Replied: '{quick_reply}'")

                state["status"] = "SPEAKING"
                speaker.speak(quick_reply)
                append_and_truncate_message(state, AIMessage(content=quick_reply))

                if any(k in user_text.lower() for k in ["goodbye", "bye", "see you"]):
                    state["status"] = "IDLE"
                    journal.log("STATE", "ENGAGED -> IDLE", "User issued dismissal.")
                    print("[State Transition] ENGAGED -> IDLE (Dismissed)")
                elif any(k in user_text.lower() for k in ["shut down", "exit"]):
                    journal.log("SHUTDOWN", "Voice shutdown issued.")
                    running = False
                    break
                else:
                    state["status"] = "ENGAGED"
                    state["last_interaction_time"] = time.time()
                continue

            # 4. LangGraph Supervisor (Remote Ollama)
            state["status"] = "PROCESSING"
            print(f"[Piper (Supervisor Processing)]: Escalating '{user_text}'...")
            journal.log("INTENT_LLM", f"Escalated prompt to Ollama: '{user_text}'")

            brain_state: PiperBrainState = {
                "mode": "ENGAGED",
                "active_user": "Steve",
                "input_text": user_text,
                "output_text": None,
                "user_context": "",
                "messages": state["messages"],
                "introspection_topic": None,
                "introspection_result": None
            }

            result = supervisor.process(brain_state)
            reply_text = result.get("output_text") or "I processed your request."
            state["messages"] = result.get("messages", state["messages"])

            print(f"[Piper (LLM Reply)]: {reply_text}")
            journal.log("REPLY_LLM", f"Generated: '{reply_text}'")

            state["status"] = "SPEAKING"
            speaker.speak(reply_text)

            # Check if supervisor shifted to IDLE
            if result.get("mode") == "ALONE":
                state["status"] = "IDLE"
                journal.log("STATE", "ENGAGED -> IDLE", "Supervisor shifted mode to ALONE.")
                print("[State Transition] ENGAGED -> IDLE (Supervisor Shift)")
            else:
                state["status"] = "ENGAGED"
                state["last_interaction_time"] = time.time()

        except KeyboardInterrupt:
            journal.log("SHUTDOWN", "SIGINT caught.")
            running = False
            break

    print("--- Shutting Down Piper Assistant ---")
    stop_event.set()
    journal.log("SYSTEM", "Piper assistant terminated.")
    speaker.speak("Goodbye.")
    sys.exit(0)


if __name__ == "__main__":
    main()