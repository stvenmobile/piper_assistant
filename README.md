# Piper Assistant: Autonomous Cognitive & Geometric Explorer

**Piper Assistant** is an agentic, self-reflecting AI system running locally on an **NVIDIA Jetson Orin NX (16GB)**. The project focuses on studying the **geometric structure of reasoning**—extracting and mapping residual activation trajectories, manifold topology, and emergent conceptual links within localized neural networks.

Instead of external camera/servo loops, Piper explores her own latent space during idle cycles and interacts with collaborators via conversational memory and a lightweight USB audio stack.

---

## 1. Core Architectural Pillars

* **Autonomous Introspection Engine:** During idle periods, Piper performs curiosity-driven exploratory reasoning across disparate knowledge domains, tracking trajectory curvature and latent clustering across transformer layers.
* **Dynamic Geometry & Knowledge Graphing:** Discovered semantic bridges and topological invariants are compiled directly into an interactive, multi-dimensional Obsidian vault.
* **Conversational Interlocutor Memory:** Eliminates computer vision in favor of conversational self-introduction and persistent markdown context profiles (`profiles/{user}.md`).
* **Edge-Native Inference:** Optimized to run entirely within the 16GB unified memory envelope of the Jetson Orin NX under a headless Linux environment.

---

## 2. Hardware & Runtime Context

* **Host Platform:** NVIDIA Jetson Orin NX Engineering Reference DevKit (16GB RAM)
* **OS / Environment:** Ubuntu 22.04 LTS (JetPack 6.1 / L4T 36.4.0, Multi-User Headless)
* **Compute Stack:** CUDA 12.6, cuDNN 9.3, PyTorch / CTranslate2
* **Audio Routing:** PipeWire USB I/O (Microphone In / Speaker Out)

---

## 3. Package Architecture & Layout

```text
piper_assistant/
+-- system_dna.md                  # Core identity, epistemic drives & introspection rules
+-- task_requests.md               # Collaborator inbox/outbox for tasks and queries
+-- daily_journal.md               # Chronological log of latent discoveries & completed tasks
+-- profiles/                      # Persistent markdown context files per speaker (e.g. steve.md)
+-- src/
¦   +-- piper_brain/               # LangGraph supervisor orchestrating Introspection vs. Collaboration
¦   +-- piper_geometry/            # PyTorch residual stream hooks, PCA, & manifold topology metrics
¦   +-- piper_audio/               # USB Voice I/O: faster-whisper (STT) and piper-tts (TTS)
¦   +-- piper_tools/               # Obsidian vault compiler mapping latent reasoning graphs
+-- .venv/
```


## 4. Operational Modes (LangGraph State Machine)
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
STATE: IDLE (The Introspective Researcher): When no collaborator is actively speaking or querying, Piper generates cross-domain prompts, computes residual stream activations, evaluates manifold curvature, and records discovered isomorphisms in daily_journal.md and the Obsidian vault.

STATE: ENGAGED (The Collaborator): When a user speaks or submits a task in task_requests.md, Piper identifies the speaker, loads their persistent profile, executes the directive, and reports findings via text or voice.

## 5. Development & Environment Setup
### 5.1 Activating the Environment
```Bash
cd ~/piper_assistant
source .venv/bin/activate
```

### 5.2 Compiling Knowledge Graphs
To rebuild the Obsidian geometric knowledge graph from newly logged activation states:

```Bash
python3 -m piper_tools.obsidian_vault_builder
python3 -m piper_tools.vault_data_linker
Point Obsidian to ~/piper_assistant/src/piper_tools/assets/reasoning_geometry_vault to explore the interactive latent graph.
```

