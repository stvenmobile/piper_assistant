# Piper Project: ROS 2 Distributed Robot — UM790 Brain Node

This repository houses the orchestration brain, Human-Robot Interaction (HRI) interfaces, data synthesis pipelines, and localized long-term knowledge graphs for the **Piper** distributed robot system. Running on the **UM790 Pro MiniPC**, this codebase acts as the central executive hub, communicating over the network with the standalone `piper_jetson` hardware driver node.

```

## 1. Distributed ROS 2 Configuration

To maintain zero-latency state synchronization with the peripheral Jetson hardware, the local network layer must remain locked to the following boundaries:

*   **Domain ID:** `42` (isolates multi-node traffic across the local network)
*   **RMW Implementation:** `CycloneDDS`
*   **ROS 2 Distribution:** `Jazzy` (Ubuntu 24.04 LTS context)

> ⚠️ **Critical Sourcing Sequence:**
> Always source the baseline framework before overlaying local workspaces:
> `source /opt/ros/jazzy/setup.bash && source install/setup.bash`

```

## 2. Build & Execution Protocol

All baseline commands must be managed from the root `/home/steve/piper_assistant` directory workspace.

### 2.1 Compiling the Brain Space
```bash
cd ~/piper_assistant
rm -rf build/ install/ log/
colcon build --symlink-install
```

### 2.2 Launching the Central Stack
To spin up the centralized execution layer (Flask Dashboard console, LangGraph Agent Supervisor, and Autonomous Perception tracking loop), run:
```bash
./src/start_piper_brain.sh
```

---

## 3. Package Architecture & Layout

Following the repository decoupling, this platform focuses entirely on high-level cognition, knowledge compilation, and user interaction:

| Package | Entry Points / Modules | Description |
| :--- | :--- | :--- |
| **`piper_brain`** | `dashboard_node` | Runs the Flask Web Dashboard (`:5000`), handling real-time HRI, manual servo overrides, and video routing. |
| | `piper_supervisor` | The core LangGraph agent supervisor orchestrating autonomous workflows, research lenses, and Human-in-the-Loop gates. |
| | `autonomous_drawing` | Runs the background scheduling loop that requests camera buffers to compile abstract physical edge sketches. |
| **`piper_tools`** | `run_synthesis` | Manages high-reasoning LLM calls via OpenRouter to compile physical architecture summaries. |
| | `obsidian_vault_builder` | Parses synthesized technical readouts into clean markdown files linked for graphing. |
| | `vault_data_linker` | Exports scraped relational database context indices down to static markdown logs. |
| | `build_reading_dashboards`| Compiles unified scrollable document summaries across the research directories. |
| **`piper_interfaces`**| `ExecuteResearch.action` | Global custom action and message definitions enabling priority-preemptive skill executions across platforms. |

---

## 4. 🗂️ Knowledge Discovery & The Obsidian Vault

The `piper_tools` suite features a dedicated pipeline designed to transform raw web scrapes and relational telemetry databases into a dynamic, fully cross-linked Obsidian Knowledge Vault.

### 4.1 Vault Architecture
All knowledge structures are compiled directly inside the sub-package asset tree:
`~/piper_assistant/src/piper_tools/piper_tools/assets/world_model_vault/`

*   **`World Model Blueprint.md`** — The master graphical directory node tracking all active operational theories.
*   **`/Sources/`** — Contains flat markdown conversions of all 74+ ingested database literature entries.
*   **`/Dashboards/`** — Holds automated scrollable viewports grouping historical context records by concept tags.

### 4.2 Rebuilding the Knowledge Graph
To clean out intermediate assets, call out to frontier models, and compile the linked vault folders sequentially, run the module sequence from the workspace root:

```bash
# 1. Pull down a fresh, synthesized architectural abstract file
python3 -m piper_tools.run_synthesis

# 2. Extract headers into distinct file-system note nodes
python3 -m piper_tools.obsidian_vault_builder

# 3. Port the underlying relational database rows to markdown source nodes
python3 -m piper_tools.vault_data_linker

# 4. Generate the macro scrollable feed viewports
python3 -m piper_tools.build_reading_dashboards
```

Point your desktop Obsidian application to open `/src/piper_tools/piper_tools/assets/world_model_vault` as a local vault folder to explore the resulting interactive mind map.

---

## 5. System Safeguards & Guidelines

*   **Virtual Environments:** Dependencies are isolated inside `/home/steve/piper_assistant/.venv`. Execution tasks called outside wrapper scripts should leverage explicit path injections targeting this space.
*   **Filename Constraints:** System instructions inside `concept_extractor.py` are strictly gated to force cloud models to output title headers completely clean of illegal file character strings (`/`, `\`, `:`).
*   **Drawing Verbosity Controls:** The `autonomous_drawing` loop runs in quiet mode by default to prevent terminal logging spam. Pass `-v` or `--verbose` explicitly at run time to re-enable underlying frame diagnostic prints.