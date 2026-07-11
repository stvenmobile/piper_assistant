#!/usr/bin/env python3
import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, TypedDict

# 👑 PREPEND: Force your virtual environment packages to the absolute front of the line
sys.path.insert(0, '/home/steve/piper_assistant/.venv/lib/python3.12/site-packages')
sys.path.insert(1, '/home/steve/piper_assistant/src/piper_brain/piper_brain')

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Empty
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionClient
from piper_interfaces.action import ExecuteResearch

# LangGraph Engine Core
from langgraph.graph import StateGraph, START, END

# ==============================================================================
# 🧠 EXECUTIVE STATE GRAPH DEFINITION
# ==============================================================================
class PiperAgentState(TypedDict):
    wake_word_detected: bool
    maintenance_due: bool
    active_tasks_left: bool
    last_maintenance_time: float
    short_term_accomplishments: List[str]
    state_logs: List[str]

# ==============================================================================
# 🛠️ LANGGRAPH STATE NODES (The Actions)
# ==============================================================================

def social_node(state: PiperAgentState) -> Dict:
    accomplishment = f"[{datetime.now().strftime('%H:%M')}] Interacted with human operator and updated immediate goals."
    return {
        "wake_word_detected": False,
        "short_term_accomplishments": state["short_term_accomplishments"] + [accomplishment],
        "state_logs": state["state_logs"] + ["Executed Human Socialization Node successfully."]
    }

def maintenance_node(state: PiperAgentState) -> Dict:
    accomplishment = f"[{datetime.now().strftime('%H:%M')}] Completed hourly system sweep and consolidated memory structures."
    return {
        "maintenance_due": False,
        "last_maintenance_time": time.time(),
        "short_term_accomplishments": [],
        "state_logs": state["state_logs"] + ["Executed System Maintenance Node successfully."]
    }

def execute_task_node(state: PiperAgentState) -> Dict:
    return {
        "short_term_accomplishments": state["short_term_accomplishments"],
        "state_logs": state["state_logs"] + ["Dispatched active ledger execution frame."]
    }

def background_research_node(state: PiperAgentState) -> Dict:
    return {
        "short_term_accomplishments": state["short_term_accomplishments"],
        "state_logs": state["state_logs"] + ["Dispatched continuous background research frame."]
    }

# ==============================================================================
# 🔀 THE EXECUTIVE GRAPH ROUTER
# ==============================================================================
def executive_router(state: PiperAgentState) -> str:
    """ Strictly evaluates flags sequentially to maintain explicit structural priorities """
    if state["wake_word_detected"]:
        return "social"
    if state["maintenance_due"]:
        return "maintenance"
    if state["active_tasks_left"]:
        return "task_execution"
    return "background_research"

# ==============================================================================
# 🤖 ROS 2 MAIN EXECUTIVE NODE
# ==============================================================================
class PiperSupervisorNode(Node):
    def __init__(self):
        super().__init__('piper_supervisor')
        self.cb_group = ReentrantCallbackGroup()
        
        # 💡 REDIRECTED: Set workspace root to point directly to the consolidated tool suite assets folder
        self.workspace_root = "/home/steve/piper_assistant/src/piper_tools/piper_tools/assets"
        
        # Keep operational configuration scratchpads located locally in the brain tasks path
        self.brain_tasks_dir = "/home/steve/piper_assistant/src/piper_brain/piper_brain/tasks"
        self.ledger_path = os.path.join(self.brain_tasks_dir, "current_tasks.md")
        self.progress_path = os.path.join(self.brain_tasks_dir, "task_progress.md")
        self.history_path = os.path.join(self.brain_tasks_dir, "history_ledger.md")
        
        # Asynchronous State Monitoring Triggers
        self.wake_word_flag = False
        self.last_maint_timestamp = time.time()
        self.maintenance_interval = 5400.0 # 1.5 Hours

        # Scheduling & Quiet-Logging States
        self.last_logged_mode = None
        self.continuous_index = 0

        # Targeted semantic search lenses to force DuckDuckGo to discover deep URLs
        self.research_lenses = [
            '"world model" architecture "autonomous intelligence"',
            'robotics "world models" "physical space" "action representation"',
            '"physical representation" modeling reasoning "embodied cognition"',
            '"self-supervised" "future generation" "spatial reasoning"'
        ]
        self.lens_index = 0

        # Action Interface for Research Operations
        self._research_client = ActionClient(self, ExecuteResearch, '/hermes/execute_research', callback_group=self.cb_group)
        self.research_goal_active = False
        self.current_running_task = None
        self.current_running_type = None

        # Communications Subsystem
        self.status_pub = self.create_publisher(String, '/hermes/status_stream', 10)
        self.wake_sub = self.create_subscription(Empty, '/hermes/wake_word_trigger', self._wake_callback, 10, callback_group=self.cb_group)
        self.approval_sub = self.create_subscription(String, '/hermes/human_approval', self._approval_callback, 10, callback_group=self.cb_group)

        # External Cloud Frontier Model Connectivity (GLM-5.2)
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")

        # Cloud API Rate Limiting / Safety Net Tracking
        self._last_cloud_execution_time = 0.0
        self._cloud_cooldown_period = 300.0  # 5 Minute mandatory cooldown between ANY cloud hits
        
        # Logging Core
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        logging.info("🛡️  Piper LangGraph Executive Node armed and loading engines...")
        logging.info("📢 [ENGINE] Social interaction routine monitoring wake words.")
        logging.info("⚙️  [ENGINE] System maintenance chronometer armed.")
        logging.info("🚀 [ENGINE] Prioritized cloud/local hybrid task scheduler initialized.")
        
        # Compile LangGraph
        self.agent_graph = self._build_executive_graph()
        
        # Start sequential ticking loop
        self.create_timer(1.0, self._executive_ticking_loop, callback_group=self.cb_group)
        logging.info("✨ All executive subsystems online. Commencing background operations.")

    def _wake_callback(self, msg):
        logging.info("📥 [ASYNC INTERRUPT] Wake word heard.")
        self.wake_word_flag = True

    def _approval_callback(self, msg):
        pass 

    def _parse_and_select_task(self) -> tuple:
        """ Parses current_tasks.md extracting tasks based on explicit priorities.
            Returns: (task_type, task_string) or (None, None)
        """
        if not os.path.exists(self.ledger_path):
            return None, None

        one_time_tasks = []
        continuous_tasks = []

        with open(self.ledger_path, 'r') as f:
            for line in f:
                clean_line = line.strip()
                if clean_line.startswith("- [ ]"):
                    one_time_tasks.append(clean_line[5:].strip())
                elif clean_line.startswith("- [*]"):
                    continuous_tasks.append(clean_line[5:].strip())

        # Rule 1: One-time tasks take absolute priority
        if one_time_tasks:
            return "one_time", one_time_tasks[0]

        # Rule 2: Round-robin scheduling using your custom '*' symbol
        if continuous_tasks:
            if self.continuous_index >= len(continuous_tasks):
                self.continuous_index = 0
            return "continuous", continuous_tasks[self.continuous_index]

        return None, None

    def _mark_one_time_task_complete(self, task_string: str):
        """ Rewrites current_tasks.md to toggle - [ ] into - [x] """
        if not os.path.exists(self.ledger_path):
            return
        
        updated_lines = []
        with open(self.ledger_path, 'r') as f:
            for line in f:
                if line.strip().startswith("- [ ]") and task_string in line:
                    line = line.replace("- [ ]", "- [x]", 1)
                updated_lines.append(line)
                
        with open(self.ledger_path, 'w') as f:
            f.writelines(updated_lines)
        logging.info(f"📝 [LEDGER] Checked off completed one-time task: '{task_string[:50]}...'")

    def _execute_one_time_cloud_task(self, task_string: str):
        """ Diverts any one-time ledger task out to GLM-5.2 using the concept_extractor tool """
        # 💡 FIX: Route pathing inside the co-located toolkit inner folder
        sys.path.insert(1, '/home/steve/piper_assistant/src/piper_tools/piper_tools')
        from concept_extractor import execute_frontier_cloud_task
        
        success = execute_frontier_cloud_task(
            task_description=task_string,
            api_key=self.openrouter_key,
            workspace_root=self.workspace_root
        )
        
        if success:
            self._mark_one_time_task_complete(task_string)
        else:
            logging.warning("⚠️ [EXECUTIVE] Cloud task execution returned a failure code. Leaving ledger open.")

    def _flush_progress_to_history(self):
        """ Archives task_progress.md into history_ledger.md and cleans scratchpad """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        raw_activity_data = ""
        
        if os.path.exists(self.progress_path):
            with open(self.progress_path, 'r') as f:
                raw_activity_data = f.read().replace("# Active Task Progress Scratchpad", "").strip()

        log_entry = f"\n\n## 🗓️ [{timestamp}] Run Summary: {self.current_running_task} ({self.current_running_type.upper()})\n"
        if raw_activity_data:
            log_entry += f"{raw_activity_data}\n"
        else:
            log_entry += "*No specific text blocks generated during this execution cycle.*\n"
        log_entry += "---\n"
        
        with open(self.history_path, 'a') as f:
            f.write(log_entry)
            
        with open(self.progress_path, 'w') as f:
            f.write(f"# Active Task Progress Scratchpad\n*Current Focus: Awaiting next scheduling assignment...*\n")

    def _build_executive_graph(self):
        builder = StateGraph(PiperAgentState)
        builder.add_node("social", social_node)
        builder.add_node("maintenance", maintenance_node)
        builder.add_node("task_execution", execute_task_node)
        builder.add_node("background_research", background_research_node)
        
        builder.add_conditional_edges(
            START, executive_router,
            {
                "social": "social",
                "maintenance": "maintenance",
                "task_execution": "task_execution",
                "background_research": "background_research"
            }
        )
        builder.add_edge("social", END)
        builder.add_edge("maintenance", END)
        builder.add_edge("task_execution", END)
        builder.add_edge("background_research", END)
        return builder.compile()

    def _executive_ticking_loop(self):
        current_time = time.time()
        maint_due = (current_time - self.last_maint_timestamp) >= self.maintenance_interval
        
        # Pull priority targets from the task ledger
        task_type, target_task = self._parse_and_select_task()
        tasks_present = target_task is not None

        # Route operational conditions
        target_mode = "background_research"
        if self.wake_word_flag:
            target_mode = "social"
        elif maint_due:
            target_mode = "maintenance"
        elif tasks_present:
            target_mode = "task_execution"

        # State transition logger guard
        if target_mode != self.last_logged_mode:
            if target_mode == "social":
                logging.info("📢 [STATE: SOCIAL] Interrupt received. Moving to HRI mode.")
            elif target_mode == "maintenance":
                logging.info("⚙️ [STATE: MAINTENANCE] Diagnostics timeline reached. Consolidating memories.")
            elif target_mode == "task_execution":
                logging.info(f"🚀 [STATE: ACTIVE LEDGER] Targeted operational block found: ({task_type.upper()})")
            elif target_mode == "background_research":
                logging.info("📚 [STATE: BACKGROUND RESEARCH] Tasks clear. Polling knowledge indices.")
            self.last_logged_mode = target_mode

        # Execute Graph Sequence
        current_state: PiperAgentState = {
            "wake_word_detected": self.wake_word_flag,
            "maintenance_due": maint_due,
            "active_tasks_left": tasks_present,
            "last_maintenance_time": self.last_maint_timestamp,
            "short_term_accomplishments": [],
            "state_logs": []
        }
        updated_state = self.agent_graph.invoke(current_state)
        
        # Synchronize back properties
        self.wake_word_flag = updated_state["wake_word_detected"]
        if not updated_state["maintenance_due"] and maint_due:
            self.last_maint_timestamp = updated_state["last_maintenance_time"]

        # Run Action client worker handoffs or divert to cloud frontier models
        if tasks_present and not self.research_goal_active and target_mode == "task_execution":
            
            if task_type == "one_time":
                current_timestamp = time.time()
                time_since_last_hit = current_timestamp - self._last_cloud_execution_time
                
                # SAFETY GATE: Verify if the mandatory 5-minute window has passed
                if time_since_last_hit >= self._cloud_cooldown_period:
                    self._last_cloud_execution_time = current_timestamp
                    self._execute_one_time_cloud_task(target_task)
                else:
                    remaining_cooldown = int(self._cloud_cooldown_period - time_since_last_hit)
                    if current_time % 10 == 0: 
                        logging.warning(
                            f"🛡️ [SAFETY NET] One-time task detected, but cloud API is cooling down. "
                            f"Skipping execution to prevent runaway charges. Retrying in {remaining_cooldown}s..."
                        )
                        
            elif task_type == "continuous":
                if self._research_client.wait_for_server(timeout_sec=0.1):
                    self.research_goal_active = True
                    self.current_running_task = target_task
                    self.current_running_type = task_type
                    
                    current_lens = self.research_lenses[self.lens_index % len(self.research_lenses)]
                    sequential_query = f"{target_task} {current_lens}"
                    
                    with open(self.progress_path, 'w') as f:
                        f.write(f"# Active Task Progress Scratchpad\n**Focus Lens:** {current_lens}\n\n")

                    goal_msg = ExecuteResearch.Goal()
                    goal_msg.research_topic = sequential_query
                    
                    self.get_logger().info(f"📡 [EXECUTIVE] Dispatching targeted lens goal over Action client: '{sequential_query}'")
                    send_goal_future = self._research_client.send_goal_async(goal_msg, feedback_callback=self._research_feedback_callback)
                    send_goal_future.add_done_callback(self._research_response_callback)

        # Clear tracking locks on short-lived loops
        if not self.wake_word_flag and target_mode == "social":
            self.last_logged_mode = None
        elif not updated_state["maintenance_due"] and target_mode == "maintenance":
            self.last_logged_mode = None

    def _research_feedback_callback(self, feedback_msg):
        url = feedback_msg.feedback.current_url_processing
        with open(self.progress_path, 'a') as f:
            f.write(f"* [{datetime.now().strftime('%H:%M:%S')}] Ingesting target url index: {url}\n")

    def _research_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            logging.warning("⚠️ Task goal rejected by worker node infrastructure.")
            self.research_goal_active = False
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._research_result_callback)

    def _research_result_callback(self, future):
        result = future.result().result
        logging.info(f"🏁 [EXECUTIVE] Task frame concluded. Resources mapped: {result.pages_ingested}")
        
        with open(self.progress_path, 'a') as f:
            f.write(f"\n### 📊 Run Conclusion\n* Status: Execution Returned Success\n* Total Pages Added to Knowledge Base: {result.pages_ingested}\n")
            
        self._flush_progress_to_history()
        
        if self.current_running_type == "continuous":
            self.continuous_index += 1
            self.lens_index += 1  
            
        self.research_goal_active = False
        self.current_running_task = None
        self.current_running_type = None
        self.last_logged_mode = None

def main(args=None):
    rclpy.init(args=args)
    node = PiperSupervisorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()