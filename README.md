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
```Bash
python3 src/main.py
(Press q + Enter in the terminal to cleanly terminate).
```

## 7. Obsidian Research Compiler
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