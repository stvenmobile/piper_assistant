# Piper Assistant: Autonomous Cognitive & Geometric Explorer

**Piper Assistant** is an agentic, self-reflecting edge AI system running locally on an **NVIDIA Jetson Orin NX (16GB)**. The project focuses on studying the **geometric structure of reasoning**—extracting and mapping residual activation trajectories, manifold topology, and emergent conceptual links within localized neural networks.

Instead of heavy external camera/servo loops, Piper explores her own latent space during idle cycles and interacts with collaborators via conversational memory and an ultra-low-latency voice I/O pipeline.

---

## 1. Core Architectural Pillars

* **Autonomous Introspection Engine:** During idle periods, Piper performs curiosity-driven exploratory reasoning across disparate knowledge domains, tracking trajectory curvature and latent clustering across transformer layers.
* **Dynamic Geometry & Knowledge Graphing:** Discovered semantic bridges and topological invariants are compiled directly into an interactive, multi-dimensional Obsidian vault.
* **Two-Tier Speech Architecture:** Flexible, hardware-aware TTS routing supporting high-prosody GPU neural diffusion and lightweight CPU-bound inference.
* **Sub-20ms Deterministic Intent Bypass:** Fast-path pattern evaluation intercepting routine status queries (date, time, environment) before hitting the LLM reasoning core.
* **Conversational Interlocutor Memory:** Persistent markdown user profiles (`profiles/{user}.md`) dynamically loaded into context upon speaker identification.

---

## 2. Hardware & Runtime Environment

* **Host Platform:** NVIDIA Jetson Orin NX Engineering Reference DevKit (16GB Unified RAM)
* **OS / Environment:** Ubuntu 22.04 LTS (JetPack 6.1 / L4T 36.4.0, Headless)
* **Compute Acceleration:** CUDA 12.6, cuDNN 9.3, PyTorch 2.x, TensorRT 10.3
* **Audio Routing:** PipeWire USB I/O (Microphone In / DAC Speaker Out @ 48kHz)
* **Remote Reasoning Core:** Ollama server on local LAN (`llama3.2:3b`)

---

## 3. Speech Synthesis Engines (Dual-Tier)

Piper includes two swappable TTS engines selectable in `config.yaml` (`audio.tts_engine`):

| Feature / Engine | Kokoro-82M (`kokoro`) | Piper TTS (`piper`) |
| :--- | :--- | :--- |
| **Compute Target** | **GPU (CUDA)** | **CPU (int8/fp32)** |
| **Hardware Requirement** | NVIDIA Jetson / Discrete CUDA GPU | Low-power CPU / Single-Board Computer |
| **Prosody & Realism** | Human-grade prosody, natural breathing/inflection | Clean, robotic-to-natural acoustic models |
| **Default Voice** | `af_heart` (American Female) / `am_adam` | `en_US-ryan-high` / `en_US-amy-medium` |
| **Sample Rate** | 24,000 Hz (Resampled to 48 kHz hardware target) | 22,050 Hz (Resampled to 48 kHz hardware target) |
| **VRAM Footprint** | ~330 MB unified memory | < 60 MB system RAM |

---

## 4. Deterministic Local Bypass (Zero-LLM Latency)

To minimize network and token generation overhead, incoming utterances are evaluated against a fast-path regex intent matcher (`src/piper_brain/quick_responder.py`). Queries matching deterministic domains return in under 20ms without invoking Ollama:

* **Temporal Ground Truth:** Real-time date, day of week, and local time.
* **Local Environmental Conditions:** Real-time weather, temperature, humidity, and wind for Matthews, NC via `wttr.in` REST endpoint.
* **Assistant State & Status:** Immediate readiness checks, wake confirmations, and session terminations.

---

## 5. Package Layout

```text
piper_assistant/
+-- system_dna.md                  # Core identity, epistemic drives & voice constraints
+-- config.yaml                    # Hardware bindings, voice selection, & model endpoints
+-- task_requests.md               # Collaborator inbox/outbox for tasks and queries
+-- daily_journal.md               # Log of latent discoveries & completed tasks
+-- profiles/                      # Persistent markdown context files per user (e.g. steve.md)
+-- src/
¦   +-- main.py                    # Master event loop and state machine coordinator
¦   +-- piper_brain/               # LangGraph supervisor, fast responder, and tool callers
¦   ¦   +-- supervisor.py          # StateGraph routing (ALONE vs. ENGAGED)
¦   ¦   +-- quick_responder.py     # Deterministic regex query matcher
¦   ¦   +-- tools.py               # Environmental ground truth & weather integrations
¦   +-- piper_audio/               # Audio subsystem
¦   ¦   +-- listener.py            # faster-whisper continuous STT with dynamic VAD
¦   ¦   +-- speaker.py             # Piper-TTS CPU fallback engine
¦   ¦   +-- kokoro_speaker.py      # Kokoro-82M CUDA GPU acceleration engine
¦   ¦   +-- models/                # Local ONNX voice weights
¦   +-- piper_geometry/            # PyTorch residual stream hooks & manifold metrics
¦   +-- piper_tools/               # Obsidian vault compiler mapping latent reasoning graphs
+-- .venv/
```

## 6. Operational State Machine
```text


                  +-------------------------------+
                  ¦   Awaiting Audio / Requests   ¦
                  +-------------------------------+
                                  ¦
                    +---------------------------+
                    ?                           ?
             +--------------+            +--------------+
             ¦  STATE: IDLE ¦            ¦STATE: ENGAGED¦
             ¦(Introspection¦            ¦ (Collaborator¦
             ¦ & Geometry)  ¦            ¦  Execution)  ¦
             +--------------+            +--------------+
```

* STATE: IDLE (Introspective Researcher): When no user is active, Piper computes residual activation trajectories, measures curvature across transformer layers, and logs topological mappings to the vault.

* STATE: ENGAGED (Collaborator): When triggered by speech or directives, Piper identifies the interlocutor, loads their profile context, handles local intents or escalates to Ollama, and delivers voice synthesis.

## 7. Setup & Execution
### 7.1 Virtual Environment Activation

```Bash
cd ~/piper_assistant
source .venv/bin/activate
```

### 7.2 Configuration

Adjust parameters in config.yaml:

```YAML
audio:
  tts_engine: "kokoro"       # Options: "kokoro" (GPU) or "piper" (CPU)
  kokoro_voice: "af_heart"   # af_heart, af_bella, am_adam, am_michael
  voice_model: "en_US-ryan-high.onnx"
  volume: 0.45
```

### 7.3 Running the Assistant
```Bash
python3 src/main.py
(Press q + Enter in the terminal to stop cleanly).
```

---