Piper Assistant: Autonomous Cognitive & Geometric Explorer

**Piper Assistant** is an agentic, self-reflecting edge AI system running locally on an **NVIDIA Jetson Orin NX (16GB)**. The project focuses on studying the **geometric structure of reasoning**—extracting and mapping residual activation trajectories, manifold topology, and emergent conceptual links within localized neural networks.

Instead of heavy external camera/servo loops, Piper explores her own latent space during idle cycles and interacts with collaborators via conversational memory and an ultra-low-latency voice I/O pipeline.

---

## 1. Core Architectural Pillars

* **Autonomous Introspection Engine:** During idle periods, Piper performs curiosity-driven exploratory reasoning across disparate knowledge domains, tracking trajectory curvature and latent clustering across transformer layers.
* **Non-Verbal AI Communication Protocol:** Investigating direct machine-to-machine semantic transfer via continuous vector geometry (residual activation tensors and orthogonal rotations) bypassing human tokenization.
* **Dynamic Geometry & Knowledge Graphing:** Discovered semantic bridges and topological invariants are compiled directly into an interactive Obsidian vault (`~/piper_assistant/obsidian/`).
* **Two-Tier Speech Architecture:** Flexible, hardware-aware TTS routing supporting high-prosody GPU neural diffusion (Kokoro-82M) and lightweight CPU-bound inference (Piper-TTS).
* **Sub-20ms Deterministic Intent Bypass:** Fast-path pattern evaluation intercepting routine status queries (date, time, environment) before hitting the LLM reasoning core.
* **Conversational Interlocutor Memory:** Persistent markdown user profiles (`profiles/{user}.md`) dynamically loaded into context upon speaker identification.

---

## 2. Machine-to-Machine Communication Research Program

The core experimental program investigates whether artificial intelligences can communicate concepts directly in their native geometry without relying on discrete human words:
```Text
┌───────────────────────────────────────────────────────────┐
│              Level 1: Autonomous Lab (M2M)               │
│                                                           │
│  [Source Model A] ──> [Residual Hook (L10)] ──> Tensor xA  │
│                                                   │       │
│                                                   ▼       │
│  [Receiver Model B] <── Tensor xB <── [Procrustes W_AB]   │
│         │                                                 │
│         ▼                                                 │
│  Zero-Shot Semantic Classification (No Words Exchanged)   │
└─────────────────────────────┬─────────────────────────────┘
│
▼
┌───────────────────────────────────────────────────────────┐
│            Level 2: Human Reporting Window                │
│                                                           │
│  - Automated Obsidian Experiments, Concepts & Journals    │
│  - Spoken Voice Summaries to Interlocutor                 │
└───────────────────────────────────────────────────────────┘
```


The research protocol is structured into three concrete phases:
1. **Extraction (`src/piper_geometry/extractor.py`)**: Hooking into intermediate transformer residual streams (e.g., Layer 10/12) to extract normalized latent trajectories $\mathbf{x} \in \mathbb{R}^{B \times S \times D}$.
2. **Rotation & Alignment (`src/piper_geometry/aligner.py`)**: Computing canonical Procrustes rotation matrices ($W = U V^T$) to align disparate architectural coordinate spaces.
3. **Blind Validation (`src/piper_geometry/tester.py`)**: Presenting held-out concept tensors to a secondary receiver model to empirically measure zero-shot understanding without tokenization.

---

## 3. Hardware & Runtime Environment

* **Host Platform:** NVIDIA Jetson Orin NX Engineering Reference DevKit (16GB Unified RAM)
* **OS / Environment:** Ubuntu 22.04 LTS (JetPack 6.1 / L4T 36.4.0, Headless)
* **Compute Stack:** CUDA 12.6, cuDNN 9.3, PyTorch 2.x, TensorRT 10.3
* **Audio Routing:** PipeWire USB I/O (Microphone In / DAC Speaker Out @ 48kHz)
* **Remote Reasoning Core:** Ollama server on local LAN (`llama3.2:3b`)

---

## 4. Speech Synthesis Engines (Dual-Tier)

| Feature / Engine | Kokoro-82M (`kokoro`) | Piper TTS (`piper`) |
| :--- | :--- | :--- |
| **Compute Target** | **GPU (CUDA)** | **CPU (int8/fp32)** |
| **Hardware Requirement** | NVIDIA Jetson / Discrete CUDA GPU | Low-power CPU / Single-Board Computer |
| **Prosody & Realism** | Human-grade prosody, natural breathing/inflection | Clean, robotic-to-natural acoustic models |
| **Default Voice** | `af_heart` (American Female) / `am_adam` | `en_US-ryan-high` / `en_US-amy-medium` |
| **Sample Rate** | 24,000 Hz (Resampled to 48 kHz hardware target) | 22,050 Hz (Resampled to 48 kHz hardware target) |
| **VRAM Footprint** | ~330 MB unified memory | < 60 MB system RAM |

---

## 5. Package Layout

```text
piper_assistant/
├── system_dna.md                  # Core identity, epistemic drives & voice constraints
├── config.yaml                    # Hardware bindings, voice selection, & model endpoints
├── obsidian/                      # Knowledge vault root
│   ├── Experiments/               # Level 1 empirical test notes (EXP-YYYYMMDD-###.md)
│   ├── Concepts/                  # Manifold topological concept profiles
│   └── Journals/                  # Daily synthesis and experiment logs
├── profiles/                      # Persistent markdown context files per user (e.g. steve.md)
├── src/
│   ├── main.py                    # Master event loop and state machine coordinator
│   ├── piper_brain/               # LangGraph supervisor, fast responder, and tool callers
│   │   ├── supervisor.py          # StateGraph routing (ALONE vs. ENGAGED)
│   │   ├── quick_responder.py     # Deterministic regex query matcher
│   │   └── tools.py               # Environmental ground truth & weather integrations
│   ├── piper_audio/               # Audio subsystem (Whisper STT, Kokoro/Piper TTS)
│   ├── piper_geometry/            # Latent extractor, Procrustes aligner, and metrics
│   │   ├── extractor.py           # PyTorch residual stream hook extractor
│   │   ├── aligner.py             # Cross-model geometric rotation engine
│   │   └── tester.py              # Blind semantic validation protocol
│   └── piper_tools/               # Obsidian vault compiler & graph generator
└── .venv/
```

## 6. Setup & Execution
### 6.1 Virtual Environment Activation
```Bash
cd ~/piper_assistant
source .venv/bin/activate
```
### 6.2 Running the System
The Orin NX now runs headless, so the **web dashboard is the only active
input/output surface** - voice (`piper_audio/`: Whisper STT, Kokoro/Piper
TTS) stays in the codebase, configured but unused by `main.py`, in case
it's wanted again later.

```Bash
python3 src/main.py
```

This starts the background research loop (throttled trials, gated to one
in-flight at a time via `PiperSupervisor.research_lock`) and serves the
dashboard at `http://<orin-ip>:8080` (configurable under `dashboard:` in
`config.yaml`). Open it from a browser on the same network to see current
status, the latest experiment summary, recent journal activity, and a text
box to send Piper an instruction or question - typing there takes the same
lock a research trial holds while it runs, so a message sent mid-trial
just waits for that trial to finish rather than competing for the GPU.
Stop the process with Ctrl+C.

## 7. ai-graph Bridge (ESP32 Dual Concept-Graph Visualization)

`src/piper_geometry/export_dual_graph.py` bridges Piper's Procrustes-alignment
research to [ai-graph](https://github.com/stvenmobile/ai-graph), a CrowPanel
ESP32-S3 display that renders two concept graphs side by side. It runs the
same hand-picked concept set through one or two (model, layer) extractions,
builds a cosine-similarity graph per side, and writes the result to
`data/dual_graph.json`.

By default it reproduces the validated same-model cross-layer comparison
(Qwen2.5-0.5B-Instruct, Layer 8 vs. Layer 14). Pass `--target-model` to
compare two genuinely different models instead:

```Bash
cd ~/piper_assistant
python3 -m piper_geometry.export_dual_graph \
    --model Qwen/Qwen2.5-0.5B-Instruct --source-layer 8 \
    --target-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --target-layer 11
```

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--model` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace repo id for side A |
| `--source-layer` | `8` | Transformer block index to hook for side A |
| `--target-model` | (same as `--model`) | HuggingFace repo id for side B - a different value compares two models instead of two layers of one model |
| `--target-layer` | `14` | Transformer block index to hook for side B |

Model weights come from `AutoModelForCausalLM.from_pretrained` (`extractor.py`)
- no separate "pull" step is needed for ungated repos, the first run just
  downloads and caches them from the Hub, which takes longer and needs more
  disk the first time.

The ESP32 board fetches `dual_graph.json` over plain HTTP, so it currently
needs a file server pointed at `data/`:

```Bash
cd ~/piper_assistant
python3 -m http.server 8420 --directory data
```

(`ufw allow 8420/tcp` if the firewall blocks it). The board's `secrets.h`
points `JETSON_HOST`/`JETSON_PORT` at this server and re-fetches once on
boot.

---

## 8. Obsidian Research Compiler
The Obsidian compiler serves as the translation layer between Level 1 (machine latent dynamics) and Level 2 (human reporting):

```text
┌──────────────────────────────────────┐
│       Level 1: Autonomous Lab        │
│  (Latent Trajectories / Injections)  │
└──────────────────┬───────────────────┘
                   │ Emits Metrics, Tensors, & Run Results
                   ▼
┌──────────────────────────────────────┐
│    Obsidian Compiler (piper_tools)   │
│  - Frontmatter Schema Generator      │
│  - Trajectory / Alignment Plotter    │
│  - Automated Graph Linker            │
└──────────────────┬───────────────────┘
                   │ Generates Markdown Notes & Dataview Tables
                   ▼
┌──────────────────────────────────────┐
│        Level 2: Human Window         │
│  - Obsidian Interactive Graph        │
│  - Spoken Voice Summaries to User    │
└──────────────────────────────────────┘
```

---