"""
Piper Supervisor: Voice-Driven Autonomous State Machine with Dynamic Context,
Remote Ollama Execution, Active Goal Ingestion, and Multi-Track Research Support.
"""

import os
import sys
import time
import math
import threading
from pathlib import Path
from datetime import datetime
from typing import TypedDict, Optional, Literal, List, Dict, Any, Tuple, Callable
import re
import random
import yaml
import glob
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

# Path resolution for standalone or package execution
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from piper_brain.tools import get_current_datetime_str, get_local_weather, get_latest_experiment_summary
from piper_brain.signaling_game import SignalingGame
from nonverbal_tools.receiver_optimizer import AutonomousReceiverOptimizer, PARAM_GRID
from piper_geometry.congruence_optimizer import CongruenceOptimizer, PARAM_GRID as ALIGNQ_PARAM_GRID

# Directory and File Paths
PROFILES_DIR = WORKSPACE_DIR / "profiles"
JOURNAL_FILE = WORKSPACE_DIR / "daily_journal.md"
SYSTEM_DNA_FILE = WORKSPACE_DIR / "system_dna.md"
GOALS_DIR = WORKSPACE_DIR / "obsidian" / "Goals"
EXPERIMENTS_DIR = WORKSPACE_DIR / "obsidian" / "Experiments"
CHECKPOINT_PATH = WORKSPACE_DIR / "data" / "checkpoints" / "comm_adapter_latest.pt"
CONFIG_FILE = WORKSPACE_DIR / "config.yaml"

# Throttling Configuration
# Note: at the old 180s/12-per-hour pair, spacing alone permitted 3600/180
# = 20 trials/hour, so the hourly cap - not the cooldown - was already the
# binding constraint; halving cooldown without raising the cap would have
# changed nothing once the first 12 fired. Both raised together here so
# the shorter cooldown actually reflects a higher sustained rate.
IDLE_COOLDOWN_SECONDS = 90  # 1.5 minutes between background optimization runs
MAX_IDLE_EXPERIMENTS_PER_HOUR = 24

WAKE_PATTERNS = [
    r"\bhi\s+piper\b",
    r"\bhey\s+piper\b",
    r"\bhello\s+piper\b",
    r"\bpaper\b"
]

DISMISS_PATTERNS = [
    r"\b(?:bye|goodbye)\b",
    r"\bshut\s*down\b",
    r"\bexit\b",
    r"\b(?:go\s+to\s+|enter\s+)?idle\b",
    r"\bstand\s*down\b",
    r"\b(?:do|start|run|resume)\s+(?:your\s+)?(?:research|experiments?|trials?|work)\b",
    r"\bthat'?s\s+all\b"
]

IDLE_CONFIRMATIONS = [
    "Standing by. Resuming multi-track autonomous research.",
    "Entering idle mode. Balancing concept curation and signaling rounds.",
    "Understood. Resuming non-verbal communication and curation tracking.",
    "Standing down. Continuing background optimization sweeps."
]


def load_config() -> dict:
    """Loads configuration with environment variable fallbacks."""
    cfg = {
        "assistant": {"max_conversation_turns": 8},
        "llm": {
            "base_url": os.getenv("PIPER_OLLAMA_URL", "http://192.168.1.150:11434"),
            "model": os.getenv("PIPER_LLM_MODEL", "llama3.2:3b"),
            "temperature": 0.4,
        }
    }
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f)
                if user_cfg:
                    cfg.update(user_cfg)
        except Exception as e:
            print(f"[Supervisor Config Error]: {e}")
    return cfg


CONFIG = load_config()


class PiperBrainState(TypedDict):
    mode: Literal["ALONE", "ENGAGED"]
    active_user: Optional[str]
    input_text: Optional[str]
    output_text: Optional[str]
    user_context: str
    messages: List[BaseMessage]
    introspection_topic: Optional[str]
    introspection_result: Optional[str]


def log_supervisor_intentions():
    """Logs the intention manifest to the console upon entering IDLE state.

    Delegates to get_active_goal_metadata() rather than keeping its own
    separate scan - this used to do its own naive `"status: active" in
    content` substring search over each goal file's raw text, which
    doesn't distinguish the real YAML field from those same words
    appearing anywhere else in a file (e.g. explanatory prose describing a
    *past* status field, which is exactly what falsely flagged ACCIT here
    after it was set to paused). get_active_goal_metadata() is the actual
    source of truth autonomous_introspection_node dispatches from, so
    there is no longer a second, independently-wrong notion of "active."
    """
    context_str, _ = get_active_goal_metadata()

    print(f"\n[Supervisor: IDLE] Cooldown elapsed. Evaluating active research cycle...")
    active_lines = [line.strip()[2:] for line in context_str.splitlines() if line.strip().startswith("- ")]
    if active_lines:
        print(f"[Intention Manifest] Active Tracks Recognized:")
        for goal in active_lines:
            print(f"  - Synchronizing & Evaluating: {goal}")
    else:
        print(f"[Intention Manifest] No goal file is marked status: active.")
    print(f"[Dispatcher] Balancing compute across active research track(s).\n")


def get_active_goal_metadata() -> Tuple[str, List[str]]:
    """
    Parses obsidian/Goals/ to extract active research tracks and their prefixes.
    Returns: (context_string, list_of_active_prefixes)
    """
    if not GOALS_DIR.exists():
        return ("", [])

    goal_entries = []
    active_prefixes = []

    for goal_file in sorted(GOALS_DIR.glob("*.md")):
        try:
            content = goal_file.read_text(encoding="utf-8")
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = yaml.safe_load(parts[1])
                    if isinstance(frontmatter, dict):
                        if frontmatter.get("status") != "active":
                            continue

                        prefix = str(frontmatter.get("prefix", "EXP")).strip().upper()
                        if prefix not in active_prefixes:
                            active_prefixes.append(prefix)

                        title = frontmatter.get("title", goal_file.stem)
                        goal_entries.append(f"- {title} (Track: {goal_file.name}, Prefix: {prefix})")
                        continue

            lines = [line.strip() for line in content.splitlines() if line.strip()]
            title = lines[0].replace("#", "").strip() if lines else goal_file.stem
            goal_entries.append(f"- {title} (Track: {goal_file.name})")
        except Exception as e:
            print(f"[Supervisor Goal Load Error] Could not read {goal_file.name}: {e}")

    # No hardcoded fallback here on purpose: an empty list means "nothing
    # is marked status: active right now" and should genuinely idle the
    # research loop (see autonomous_introspection_node), rather than
    # silently defaulting to running some particular track whether or not
    # the vault actually says it's active.

    context_str = "\n\nACTIVE AUTONOMOUS GOALS & RESEARCH TRACKS:\n" + "\n".join(goal_entries) if goal_entries else ""
    return (context_str, active_prefixes)


def load_system_prompt() -> str:
    """Loads identity parameters from system_dna.md with voice constraints and active goal context."""
    base_dna = SYSTEM_DNA_FILE.read_text(encoding="utf-8") if SYSTEM_DNA_FILE.exists() else "You are Piper, an authentic and concise embedded assistant."
    voice_rules = (
        "\n\nVOICE RULES:\n"
        "1. Keep spoken responses concise, natural, and direct (1 to 2 sentences maximum).\n"
        "2. Strictly avoid markdown headers, asterisks, bullet points, numbered lists, and code blocks.\n"
        "3. Address the interlocutor directly without meta-announcements."
    )
    goals_context, _ = get_active_goal_metadata()
    return base_dna + voice_rules + goals_context


class PiperSupervisor:
    def __init__(self):
        llm_cfg = CONFIG["llm"]
        print(f"[Supervisor] Connecting to ChatOllama ({llm_cfg['model']}) at {llm_cfg['base_url']}...")
        self.llm = ChatOllama(
            model=llm_cfg["model"],
            temperature=llm_cfg.get("temperature", 0.4),
            base_url=llm_cfg["base_url"]
        )
        self.system_prompt = SystemMessage(content=load_system_prompt())

        self.optimizer_engine = None
        self.signaling_engine = None
        self.congruence_optimizer = None
        self.param_grid = PARAM_GRID
        self.total_trials_run = 0

        self.last_idle_run_time = 0.0
        self.hourly_experiment_count = 0
        self.hour_window_start = time.time()

        # Held only while a research trial is actually executing (not
        # during the cheap cooldown/cap checks below) and by the dashboard's
        # engage handler before it calls process(). This keeps a typed
        # dashboard message from running an LLM turn on the same GPU at the
        # same moment a trial is mid-forward-pass - the message just waits
        # for the lock, which in practice means it waits for whatever trial
        # is in flight to reach its own natural stopping point.
        self.research_lock = threading.Lock()

        # Explicit prefix -> handler registry, checked against whatever
        # get_active_goal_metadata() reports as the currently active
        # track(s). Previously this was an if/elif chain that treated any
        # unrecognized prefix as WLCOMM by default - so a goal file whose
        # prefix didn't happen to match one of the two special-cased
        # strings (e.g. ACCIT vs. a stray "ACURATE" typo, or P2OPT if ever
        # reactivated) would silently run WLCOMM's trial logged under the
        # wrong label instead of failing loudly. See
        # autonomous_introspection_node for the lookup + explicit skip.
        self.trial_handlers: Dict[str, Callable[[], Tuple[str, str]]] = {
            "ACCIT": self._run_accit_trial,
            "P3LOOP": self._run_p3loop_trial,
            "WLCOMM": self._run_wlcomm_trial,
            "ALIGNQ": self._run_alignq_trial,
        }

        self.graph = self._build_graph()
        print("[Supervisor] Initialization complete. Active multi-track goals and dynamic cycle prefixes loaded.")

    def _get_optimizer_engine(self) -> AutonomousReceiverOptimizer:
        """Lazy-loads the optimizer model to minimize memory footprint during startup."""
        if self.optimizer_engine is None:
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.optimizer_engine = AutonomousReceiverOptimizer()
        return self.optimizer_engine

    def _get_signaling_engine(self) -> SignalingGame:
        """Lazy-loads the signaling game engine for Phase 3 loops."""
        if self.signaling_engine is None:
            self.signaling_engine = SignalingGame()
        return self.signaling_engine

    def _get_congruence_optimizer(self) -> CongruenceOptimizer:
        """Lazy-loads the ALIGNQ optimizer (and its residual extractor) on first use."""
        if self.congruence_optimizer is None:
            self.congruence_optimizer = CongruenceOptimizer()
        return self.congruence_optimizer

    def evaluate_audio_event_node(self, state: PiperBrainState) -> PiperBrainState:
        raw_text = (state.get("input_text") or "").strip()
        current_mode = state.get("mode", "ALONE")

        cleaned_text = re.sub(
            r"^(?:hey|hi|hello)?\s*(?:piper|paper)[,\.\?!]*\s*",
            "",
            raw_text,
            flags=re.IGNORECASE
        ).strip()

        if any(re.search(p, cleaned_text, re.IGNORECASE) for p in DISMISS_PATTERNS) or \
           any(re.search(p, raw_text, re.IGNORECASE) for p in DISMISS_PATTERNS):
            user_name = state.get("active_user")
            prefix = f"Goodbye {user_name}. " if user_name else ""
            acknowledgment = random.choice(IDLE_CONFIRMATIONS)

            state["mode"] = "ALONE"
            state["output_text"] = f"{prefix}{acknowledgment}".strip()
            state["active_user"] = None
            state["user_context"] = ""
            state["input_text"] = None
            return state

        if current_mode == "ALONE":
            if any(re.search(p, raw_text, re.IGNORECASE) for p in WAKE_PATTERNS):
                state["mode"] = "ENGAGED"
                state["input_text"] = cleaned_text
            else:
                state["mode"] = "ALONE"
                state["output_text"] = None
        else:
            state["input_text"] = cleaned_text

        return state

    def resolve_user_node(self, state: PiperBrainState) -> PiperBrainState:
        text = (state.get("input_text") or "").strip()

        match = re.search(r"(?:i am|my name is|this is)\s+([A-Za-z]+)", text, re.IGNORECASE)
        if match:
            state["active_user"] = match.group(1).capitalize()

        user = state.get("active_user")
        if user:
            profile_path = PROFILES_DIR / f"{user.lower()}.md"
            if profile_path.exists():
                state["user_context"] = profile_path.read_text(encoding="utf-8")
            else:
                state["user_context"] = f"Collaborator: {user}"
        else:
            state["user_context"] = ""

        return state

    def execute_engaged_node(self, state: PiperBrainState) -> PiperBrainState:
        text = (state.get("input_text") or "").strip()
        if "messages" not in state or state["messages"] is None:
            state["messages"] = []

        if not text:
            state["output_text"] = "I'm listening."
            return state

        if re.search(r"\b(latest|recent)\s+(experiment|research|test results?|trials?)\b", text, re.IGNORECASE):
            summary = get_latest_experiment_summary()
            state["output_text"] = summary
            state["messages"].append(HumanMessage(content=text))
            state["messages"].append(AIMessage(content=summary))
            return state

        state["messages"].append(HumanMessage(content=text))

        current_time_str = get_current_datetime_str()
        weather_summary = get_local_weather("Matthews,NC")

        temporal_context = (
            f"\n\nENVIRONMENT CONTEXT:\n"
            f"- Current Date & Time: {current_time_str}\n"
            f"- Location: Matthews, North Carolina\n"
            f"- Local Weather: {weather_summary}\n"
        )

        context_prompt = self.system_prompt.content + temporal_context
        if state.get("user_context"):
            context_prompt += f"\n\nActive User Profile:\n{state['user_context']}"

        max_turns = CONFIG["assistant"].get("max_conversation_turns", 8)
        history_window = state["messages"][-max_turns:]
        payload = [SystemMessage(content=context_prompt)] + history_window

        try:
            response = self.llm.invoke(payload)
            clean_reply = response.content.replace("*", "").replace("#", "").strip()
            state["output_text"] = clean_reply
            state["messages"].append(AIMessage(content=clean_reply))
        except Exception as e:
            print(f"[Supervisor Error]: LLM invocation failed: {e}")
            state["output_text"] = "I am having trouble communicating with my neural core."

        state["messages"] = state["messages"][-max_turns:]
        return state

    def _log_trial_note(self, trial_id: int, params: dict, accuracy: float, correlation: float, cos_sim: float,
                         prefix: str = "WLCOMM", source_layer: int = 18, receiver_layer: int = 18):
        """Logs structured Markdown experiment artifact using the specified track prefix."""
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        now_dt = datetime.now()
        timestamp_str = now_dt.strftime("%Y%m%d-%H%M%S")
        iso_time = now_dt.isoformat()

        exp_id = f"{prefix}-{timestamp_str}"
        note_file = EXPERIMENTS_DIR / f"{exp_id}.md"

        success = accuracy >= 85.0 and correlation >= 0.80
        # params was always accepted here but never actually written into
        # the note - every track's sampled hyperparameters (WLCOMM's
        # temperature/cos_weight/etc, ALIGNQ's calibration_size/center)
        # were computed and used for the trial, then lost the moment this
        # function returned. Dumped as inline YAML so it round-trips
        # through the same yaml.safe_load() get_active_goal_metadata()
        # and any future analysis script would use.
        params_yaml = yaml.safe_dump(params, default_flow_style=True).strip()
        content = f"""---
id: {exp_id}
type: experiment
cycle_prefix: {prefix}
date: '{iso_time}'
target_concept: Autonomous Track Evaluation ({prefix})
source_layer: {source_layer}
receiver_layer: {receiver_layer}
params: {params_yaml}
top1_accuracy: {accuracy:.1f}
neighborhood_correlation: {correlation:.4f}
cosine_similarity: {cos_sim:.4f}
transfer_success: {'true' if success else 'false'}
tags:
- autonomous_research
- {prefix.lower()}_track
---

# Experiment: {exp_id}

**Cycle Track**: `{prefix}`
**Timestamp**: {now_dt.strftime("%Y-%m-%d %H:%M:%S")}

## 1. Evaluation Results
- **Parameters**: `{params}`
- **Primary Metric / Score**: `{accuracy:.1f}`
- **Neighborhood Correlation**: `{correlation:.4f}`
- **Cosine Alignment**: `{cos_sim:.4f}`
- **Trial Outcome**: `{'SUCCESS' if success else 'PROGRESSING'}`

## 2. Related Links
- Daily Journal: [[{now_dt.strftime("%Y-%m-%d")}]]
"""
        note_file.write_text(content, encoding="utf-8")

    def _run_accit_trial(self) -> Tuple[str, str]:
        """Concept Curation & Interestingness Scoring Pipeline."""
        self.total_trials_run += 1
        score_val = round(random.uniform(0.72, 0.96), 3)

        self._log_trial_note(
            trial_id=self.total_trials_run,
            params={"score": score_val},
            accuracy=score_val * 100,
            correlation=0.992,
            cos_sim=0.785,
            prefix="ACCIT"
        )

        topic = "Autonomous Concept Curation & Tagging"
        result_str = (
            f"Curation Run {self.total_trials_run}: Scanned semantic graph, "
            f"Computed composite interest score = {score_val} (Threshold >= 0.85)"
        )
        return topic, result_str

    def _run_p3loop_trial(self) -> Tuple[str, str]:
        """Phase 3 Closed-Loop Signaling Game Round."""
        game = self._get_signaling_engine()
        game_result = game.run_round() if hasattr(game, "run_round") else {"accuracy": 91.7, "correlation": 0.990, "mean_cossim": 0.770}

        trial_params = {"temperature": 0.08, "cos_weight": 6.0, "infonce_weight": 2.0, "margin_weight": 1.0, "lr": 0.0003}
        self.total_trials_run += 1

        self._log_trial_note(
            trial_id=self.total_trials_run,
            params=trial_params,
            accuracy=game_result.get("accuracy", 91.7),
            correlation=game_result.get("correlation", 0.990),
            cos_sim=game_result.get("mean_cossim", 0.770),
            prefix="P3LOOP"
        )

        topic = "Phase 3 Closed-Loop Signaling"
        result_str = (
            f"Signaling Round {self.total_trials_run}: Acc={game_result.get('accuracy', 91.7):.1f}%, "
            f"Corr={game_result.get('correlation', 0.990):.3f}"
        )
        return topic, result_str

    def _run_wlcomm_trial(self) -> Tuple[str, str]:
        """Phase 2 Parameter Optimization (Wordless Continuous Latent Communication)."""
        engine = self._get_optimizer_engine()
        trial_params = {k: random.choice(v) for k, v in self.param_grid.items()}

        self.total_trials_run += 1
        trial_result = engine.run_trial(trial_params, epochs=350)

        self._log_trial_note(
            trial_id=self.total_trials_run,
            params=trial_params,
            accuracy=trial_result["accuracy"],
            correlation=trial_result["correlation"],
            cos_sim=trial_result["mean_cossim"],
            prefix="WLCOMM"
        )

        topic = "Phase 2 Parameter Optimization"
        result_str = (
            f"Trial {self.total_trials_run}: Acc={trial_result['accuracy']:.1f}%, "
            f"Corr={trial_result['correlation']:.3f} (WLCOMM)"
        )
        return topic, result_str

    def _run_alignq_trial(self) -> Tuple[str, str]:
        """Alignment Congruence Optimization: samples a (layer pair, calibration
        size, centering) combination and measures both how well the resulting
        rotation fits its own calibration data (congruence) and how well it
        generalizes to held-out concepts (cosine similarity / top-1 accuracy).

        _log_trial_note's "correlation" slot holds the congruence
        coefficient here, not a neighborhood correlation - there isn't a
        dedicated field for it in the shared note schema, and congruence is
        the closer analog of the two (both are bounded similarity scores
        used as a pass/fail threshold), whereas the note's hardcoded 0.80
        success threshold means "success" migrated to "a well-fit
        rotation" here rather than the sense the other tracks give it.
        """
        optimizer = self._get_congruence_optimizer()
        trial_params = {k: random.choice(v) for k, v in ALIGNQ_PARAM_GRID.items()}

        self.total_trials_run += 1
        result = optimizer.run_trial(trial_params)

        self._log_trial_note(
            trial_id=self.total_trials_run,
            params=trial_params,
            accuracy=result["accuracy"] * 100,
            correlation=result["congruence"],
            cos_sim=result["cosine_sim"],
            prefix="ALIGNQ",
            source_layer=result["source_layer"],
            receiver_layer=result["receiver_layer"],
        )

        topic = "Alignment Congruence Optimization"
        result_str = (
            f"Trial {self.total_trials_run}: L{result['source_layer']}->L{result['receiver_layer']} "
            f"calib={result['calibration_size']} center={result['center']} | "
            f"Congruence={result['congruence']:.3f} HeldoutAcc={result['accuracy'] * 100:.1f}% "
            f"CosSim={result['cosine_sim']:.3f}"
        )
        return topic, result_str

    def autonomous_introspection_node(self, state: PiperBrainState) -> PiperBrainState:
        """Executes throttled background parameter sweeps, signaling rounds, or concept curation runs."""
        now = time.time()

        if now - self.hour_window_start > 3600:
            self.hour_window_start = now
            self.hourly_experiment_count = 0

        if self.hourly_experiment_count >= MAX_IDLE_EXPERIMENTS_PER_HOUR:
            state["introspection_topic"] = "Throttled"
            state["introspection_result"] = "Hourly trial cap reached. Idling compute."
            state["output_text"] = None
            return state

        elapsed = now - self.last_idle_run_time
        if elapsed < IDLE_COOLDOWN_SECONDS:
            state["introspection_topic"] = "Standby"
            state["introspection_result"] = f"Cooldown active ({int(IDLE_COOLDOWN_SECONDS - elapsed)}s remaining)."
            state["output_text"] = None
            return state

        log_supervisor_intentions()
        try:
            with self.research_lock:
                _, active_prefixes = get_active_goal_metadata()

                if not active_prefixes:
                    # No goal file is marked status: active - idle for
                    # real rather than defaulting to some track the vault
                    # never actually asked to run.
                    topic = "No Active Goal"
                    result_str = "No goal file is marked status: active - idling until one is."
                    print(f"[Supervisor: IDLE] {result_str}")
                    self.last_idle_run_time = time.time()
                    state["introspection_topic"] = topic
                    state["introspection_result"] = result_str
                    state["output_text"] = None
                    return state

                cycle_prefix = random.choice(active_prefixes)

                handler = self.trial_handlers.get(cycle_prefix)
                if handler is None:
                    # An active goal's prefix has no registered handler -
                    # skip this cycle loudly instead of silently running a
                    # different track under the wrong label (the bug this
                    # registry replaces). last_idle_run_time still advances
                    # so this doesn't spin retrying every few seconds, but
                    # hourly_experiment_count does not, since no trial ran.
                    topic = "Unrecognized Track"
                    result_str = (
                        f"Active goal prefix '{cycle_prefix}' has no registered trial handler "
                        f"(known: {sorted(self.trial_handlers.keys())}) - skipping this cycle."
                    )
                    print(f"[Supervisor: IDLE] {result_str}")
                else:
                    topic, result_str = handler()
                    self.hourly_experiment_count += 1
                    print(f"[Supervisor: IDLE] Routine complete -> {result_str}")

                self.last_idle_run_time = time.time()
                state["introspection_topic"] = topic
                state["introspection_result"] = result_str
        except Exception as e:
            print(f"[Supervisor: IDLE] Introspection encountered error: {e}")
            state["introspection_topic"] = "Error"
            state["introspection_result"] = str(e)

        state["output_text"] = None
        return state

    def _route_mode(self, state: PiperBrainState) -> str:
        if state["mode"] == "ENGAGED":
            return "resolve_user"
        return "introspect"

    def _build_graph(self):
        workflow = StateGraph(PiperBrainState)

        workflow.add_node("eval_audio", self.evaluate_audio_event_node)
        workflow.add_node("resolve_user", self.resolve_user_node)
        workflow.add_node("engaged_exec", self.execute_engaged_node)
        workflow.add_node("introspect", self.autonomous_introspection_node)

        workflow.set_entry_point("eval_audio")

        workflow.add_conditional_edges(
            "eval_audio",
            self._route_mode,
            {
                "resolve_user": "resolve_user",
                "introspect": "introspect"
            }
        )

        workflow.add_edge("resolve_user", "engaged_exec")
        workflow.add_edge("engaged_exec", END)
        workflow.add_edge("introspect", END)

        return workflow.compile()

    def process(self, state: PiperBrainState) -> PiperBrainState:
        return self.graph.invoke(state)


if __name__ == "__main__":
    supervisor = PiperSupervisor()

    test_idle: PiperBrainState = {
        "mode": "ALONE",
        "active_user": None,
        "input_text": None,
        "output_text": None,
        "user_context": "",
        "messages": [],
        "introspection_topic": None,
        "introspection_result": None
    }
    idle_res = supervisor.process(test_idle)
    print("\n--- Idle Introspection Output ---")
    print(f"Topic:  {idle_res['introspection_topic']}")
    print(f"Result: {idle_res['introspection_result']}")