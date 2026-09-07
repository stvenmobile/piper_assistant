"""
Piper Web Dashboard.

Replaces the mic/TTS runtime loop as Piper's only active input/output
surface now that the Orin NX runs headless - voice (piper_audio/) stays in
the codebase, just unused by main.py. This module shares the same
AgentState dict and PiperSupervisor instance main.py's background research
worker uses: typing a message here takes supervisor.research_lock before
calling process(), so it waits for any in-flight research trial to reach
its own natural stopping point instead of racing it for the GPU.
"""

from pathlib import Path
from typing import Optional

from flask import Flask, request, redirect, url_for, render_template_string
from langchain_core.messages import HumanMessage

from piper_brain.tools import get_latest_experiment_summary
from piper_brain.supervisor import PiperSupervisor, PiperBrainState
from piper_brain.state import AgentState
from piper_brain.journal import ActivityJournal, JOURNAL_FILE

PAGE_TEMPLATE = """
<!doctype html>
<title>Piper Dashboard</title>
<meta http-equiv="refresh" content="20">
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; background: #111; color: #eee; }
  h1, h2 { color: #fff; }
  .status { display: inline-block; padding: 2px 10px; border-radius: 10px; background: #2a2; color: #fff; font-size: 0.9em; }
  textarea { width: 100%; background: #222; color: #eee; border: 1px solid #444; border-radius: 4px; padding: 8px; font-size: 1em; }
  button { margin-top: 8px; padding: 6px 18px; background: #36c; color: #fff; border: none; border-radius: 4px; cursor: pointer; }
  .reply { background: #1a2a1a; border-left: 3px solid #2a2; padding: 8px 12px; margin: 1em 0; }
  .messages { list-style: none; padding: 0; }
  .messages li { padding: 4px 0; border-bottom: 1px solid #333; }
  pre { background: #1a1a1a; padding: 10px; overflow-x: auto; font-size: 0.85em; }
</style>

<h1>Piper</h1>
<p>Status: <span class="status">{{ status }}</span></p>
<p>{{ latest_experiment }}</p>

<h2>Engage</h2>
<form method="post" action="{{ url_for('engage') }}">
  <textarea name="text" rows="3" placeholder="Type an instruction or question for Piper..."></textarea><br>
  <button type="submit">Send</button>
</form>

{% if reply %}
<div class="reply"><b>Piper says:</b> {{ reply }}</div>
{% endif %}

<h2>Recent conversation</h2>
<ul class="messages">
{% for m in messages %}
  <li><b>{{ m.role }}:</b> {{ m.content }}</li>
{% else %}
  <li>No conversation yet.</li>
{% endfor %}
</ul>

<h2>Recent journal</h2>
<pre>{{ journal_tail }}</pre>
"""


def _journal_tail(lines: int = 25) -> str:
    if not JOURNAL_FILE.exists():
        return ""
    text = JOURNAL_FILE.read_text(encoding="utf-8")
    return "\n".join(text.splitlines()[-lines:])


def create_app(supervisor: PiperSupervisor, state: AgentState, journal: ActivityJournal) -> Flask:
    app = Flask(__name__)

    def _render(reply: Optional[str] = None):
        readable_messages = []
        for m in state["messages"][-8:]:
            role = "You" if isinstance(m, HumanMessage) else "Piper"
            readable_messages.append({"role": role, "content": m.content})

        return render_template_string(
            PAGE_TEMPLATE,
            status=state["status"],
            latest_experiment=get_latest_experiment_summary(),
            reply=reply,
            messages=readable_messages,
            journal_tail=_journal_tail(),
        )

    @app.route("/")
    def index():
        return _render()

    @app.route("/engage", methods=["POST"])
    def engage():
        text = request.form.get("text", "").strip()
        if not text:
            return redirect(url_for("index"))

        state["status"] = "PROCESSING"
        journal.log("INTENT_WEB", f"Dashboard input: '{text}'")

        brain_state: PiperBrainState = {
            "mode": "ENGAGED",
            "active_user": "Steve",
            "input_text": text,
            "output_text": None,
            "user_context": "",
            "messages": state["messages"],
            "introspection_topic": None,
            "introspection_result": None,
        }

        with supervisor.research_lock:
            result = supervisor.process(brain_state)

        reply_text = result.get("output_text") or "Understood."
        state["messages"] = result.get("messages", state["messages"])
        journal.log("REPLY_WEB", f"Generated: '{reply_text}'")

        # No continuous listening session exists here the way it did with
        # the mic (that had an inactivity timeout tied to listening
        # cadence) - each dashboard submission is a single discrete turn,
        # so drop straight back to IDLE so background research resumes
        # between messages rather than waiting on a timeout that no longer
        # applies.
        state["status"] = "IDLE"

        return _render(reply=reply_text)

    return app
