#!/usr/bin/env python3
import os
import sys

# Force the worker process to read from your local virtual environment packages first
sys.path.insert(0, '/home/steve/piper_assistant/.venv/lib/python3.12/site-packages')
sys.path.insert(1, '/home/steve/piper_assistant/src/piper_tools')

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from piper_interfaces.action import ExecuteResearch
from piper_tools.research_engine import PiperResearchEngine

class HermesResearchActionServer(Node):
    def __init__(self):
        super().__init__('hermes_research_node')
        self.cb_group = ReentrantCallbackGroup()
        
        # Instantiate our working pipeline engine locally with expanded max search boundaries
        self.research_engine = PiperResearchEngine(max_results=5)

        # Start the action hosting framework
        self._action_server = ActionServer(
            self,
            ExecuteResearch,
            '/hermes/execute_research',
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self.cb_group
        )
        self.get_logger().info("🎓 [HERMES ENGINE] Research Action Server fully loaded and waiting for goals.")

    def _goal_callback(self, goal_request):
        self.get_logger().info(f"📥 Received new research request for: '{goal_request.research_topic}'")
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        self.get_logger().info("✂️ Received request to cancel active web-scraping sweep.")
        return CancelResponse.ACCEPT

    async def _execute_callback(self, goal_handle):
        """ Runs asynchronously, executing the long LLM and scraping tasks without stalling ROS """
        topic_query = goal_handle.request.research_topic
        feedback_msg = ExecuteResearch.Feedback()
        result = ExecuteResearch.Result()

        self.get_logger().info(f"🚀 [WORKER] Initiating web compilation loop for query: {topic_query}")

        try:
            # Step 1: Query indices (Now returns a shuffled array internally)
            search_results = self.research_engine.execute_web_search(topic_query)
            pages_counter = 0

            # Step 2: Iterate and stream feedback back to Piper while doing heavy text extraction
            for item in search_results:
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.success = False
                    return result

                # 💡 THE PROACTIVE GUARD: Drop out early if the URL is already processed in the SQLite DB
                if self.research_engine.is_url_duplicate(item["url"]):
                    self.get_logger().info(f"⏭️ [PRE-FILTER] URL already present in SQLite tables. Skipping: {item['url']}")
                    continue

                feedback_msg.current_url_processing = item["url"]
                goal_handle.publish_feedback(feedback_msg)

                # Execute raw scraping & local LLM structured database parsing pass
                scraped_content = self.research_engine.scrape_url_content(item["url"])
                if scraped_content["success"]:
                    structured_json = self.research_engine.parser.parse_scraped_text(scraped_content["raw_text"], item["url"])
                    if structured_json and "topical_data" in structured_json:
                        structured_json["topical_data"]["url"] = item["url"]
                        self.research_engine._insert_structured_payload(structured_json)
                        pages_counter += 1

            # Wrap up execution transaction frame
            goal_handle.succeed()
            result.success = True
            result.pages_ingested = pages_counter
            self.get_logger().info(f"✨ [WORKER] Completed sweep execution. Ingested {pages_counter} new resources successfully.")
            
        except Exception as e:
            self.get_logger().error(f"❌ [WORKER] Pipeline execution failed unexpectedly: {e}")
            goal_handle.abort()
            result.success = False
            result.pages_ingested = 0

        return result

def main(args=None):
    rclpy.init(args=args)
    node = HermesResearchActionServer()
    executor = MultiThreadedExecutor()
    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()