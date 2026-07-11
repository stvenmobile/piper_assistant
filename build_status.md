# Piper 3.0 Phase Checklists & Build Tracker

## 🟩 Phase 1: Core Architecture & Workspace Setup
* [ ] Initialize workspace repository and packages (`piper_brain`, `piper_drivers`, `piper_tools`, `piper_interfaces`).
* [ ] Deploy primary `piper_supervisor.py` LangGraph skeleton into `piper_brain`.
* [ ] Define custom ROS 2 topics/triggers in `piper_interfaces` (specifically `/hermes/wake_word_trigger`).

## 🟨 Phase 2: Centralized Tools Library (`piper_tools`)
* [ ] **Research & Scrape Tool**: Web-based content gatherer and text-synthesizer (Read-Only) **[NEXT TARGET TASK]**.
* [ ] **RAG / Vector Knowledge Base Interface**: Tool to store and fetch technical telemetry and document guides.
* [ ] **Code Verification Suite**: Integrated AST/Syntax compiler checker and `colcon build` trigger tool.
* [ ] **HRI / Socialization Logic**: Conversational text engine driven by Ollama.

## 🟦 Phase 3: Hardware Porting & Drivers (`piper_drivers`)
* [ ] Recycle `camera_node.py` and resolve GStreamer sequencing dependencies.
* [ ] Recycle `vision_tracking_node.py` (YOLO JSON interface pipeline).
* [ ] Recycle `servo_node.py` for head tracking and neck posture stabilization.

## 🟪 Phase 4: Summarization & State Consolidator
* [ ] Build local SQLite database logging layer for permanent episodic state retention.
* [ ] Update Dashboard pipeline to pull directly from the LangGraph context log engine.
