# Architectural Blueprint: Mathematics of Reasoning Engine

The goal for Piper's IDLE state is to move beyond passive waiting and transform idle compute cycles into an autonomous mechanistic interpretability engine. Rather than running open-ended prompt loops that consume GPU memory blindly, Piper systematically probes, instruments, and visualizes the latent representations and geometric trajectories of transformer reasoning on the Jetson Orin NX.       

```
[IDLE Mode Triggered]
                 │
                 ▼
     ┌───────────────────────┐
     │ 1. Hypothesis / Prompt │ (Disparate cross-domain seed pairs)
     │       Generator       │
     └──────────┬────────────┘
                │
                ▼
     ┌───────────────────────┐
     │ 2. PyTorch Residual   │ (Extract hidden states across layers 0..N)
     │     Activation Hooks  │
     └──────────┬────────────┘
                │
                ▼
     ┌───────────────────────┐
     │ 3. Geometric Metric   │ (Curvature, Geodesic Drift, PCA/UMAP,
     │       Computer        │  Intrinsic Dimensionality, Cosine Alignment)
     └──────────┬────────────┘
                │
                ▼
     ┌───────────────────────┐
     │ 4. Topological Graph  │ (Compile discoveries into Markdown notes
     │     & Vault Logger    │  with frontmatter and Obsidian wikilinks)
     └───────────────────────┘
```

## Core Components & Objectives
1. Minimal Local Probe Model (src/piper_geometry/probe.py)Objective: Use a small, quantized local model running directly on the Orin GPU (e.g., Qwen-2.5-0.5B/1.5B or TinyLlama via PyTorch/Hugging Face) specifically for hook access.Why: Ollama exposes only final generated text via HTTP, hiding intermediate tensor layers. A lightweight local PyTorch model gives zero-overhead access to the exact residual stream vectors across all layers ($L_0 \rightarrow L_N$).

2. Geometric & Topological Metric Pipeline (src/piper_geometry/metrics.py)Geodesic Drift & Trajectory Curvature: Measures how sharply representations change direction from input tokens to output conclusions (quantifying the "cognitive leap" or phase transition in multi-step reasoning).Layer-Wise Cosine Alignment: Tracks where disparate concepts (e.g., fluid dynamics vs. economic inflation) converge in semantic space to detect emergent cross-domain analogies.Intrinsic Dimensionality (ID Estimation): Computes local manifold compression across intermediate layers (evaluating how representations compress before decompression into vocabulary tokens).

3. Synthetic Introspection Scheduler (src/piper_geometry/curiosity.py)Objective: Programmatic generation of probing prompt pairs combining two unconnected domains (e.g., thermodynamic entropy and information theory, or topological knots and code syntax).Execution: Evaluates whether cross-layer trajectories reveal mathematical isomorphisms or collapse into distinct semantic clusters.

4. Obsidian Knowledge Graph Compiler (src/piper_tools/vault_compiler.py)Objective: Automatically translate computed numeric metrics into structured Markdown files inside your Obsidian vault.Structure: Each note includes YAML frontmatter (layer depth, trajectory curvature, manifold dimension) and dynamic [[wikilinks]] connecting isomorphic concepts, turning Piper's idle cycles into an interactive topological graph.

## Resource Envelope Strategy on Jetson Orin NX
-  Memory Budget: Reserve 1.5GB–2.5GB unified RAM for the local probe model and activation buffers, leaving plenty of room for Kokoro-82M (~330MB) and system services.Instant Yield on Speech: The LangGraph state machine ensures that the moment VAD detects user input on the microphone, the background introspection loop pauses immediately, prioritizing user interaction without audio stutter.