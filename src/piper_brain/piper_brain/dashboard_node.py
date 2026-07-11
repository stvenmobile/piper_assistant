#!/usr/bin/env python3
import os
import sys

# 👑 FORCE INTERPRETER TO LOOK IN VENV FIRST
sys.path.insert(0, '/home/steve/piper_assistant/.venv/lib/python3.12/site-packages')

import time
import json
import logging
import sqlite3
import threading
import re
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify

# Pure ROS 2 Index Package Locators
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Vector3
from std_msgs.msg import String

# ==========================================================================
# PATH RESOLUTION CORE
# ==========================================================================
current_script_dir = os.path.dirname(os.path.abspath(__file__))
if "install/" in current_script_dir:
    workspace_root = os.path.abspath(os.path.join(current_script_dir, "../../../../src/piper_brain/piper_brain"))
    resolved_template_dir = os.path.join(workspace_root, "templates")
    resolved_assets_dir = os.path.join(workspace_root, "assets")
    resolved_tasks_dir = os.path.join(workspace_root, "tasks")
else:
    resolved_template_dir = os.path.join(current_script_dir, "templates")
    resolved_assets_dir = os.path.join(current_script_dir, "assets")
    resolved_tasks_dir = os.path.join(current_script_dir, "tasks")

# ==========================================================================
# FLASK SERVER CONTEXT
# ==========================================================================
_app = Flask(__name__, template_folder=resolved_template_dir, static_folder=resolved_assets_dir, static_url_path='/assets')
_log = logging.getLogger('werkzeug')
_log.setLevel(logging.ERROR)

# 💡 Shared Variable for Traditional Direct JPEG Storage Stream Mapping
_latest_frame_jpeg = None

# Base layout tracking metrics
_system_state_snapshot = {
    "state": "SOLO", 
    "active_task": "System running autonomously. Conducting spatial research.",
    "objects": [],
    "matrix_report": {
        "P1_Top_Left": [],
        "P2_Top_Right": [],
        "P3_Bottom_Right": [],
        "P4_Bottom_Left": []
    }
}

# ==========================================================================
# FILE SYSTEM MEMORY LOCATORS
# ==========================================================================
TASK_LEDGER_PATH = os.path.join(resolved_tasks_dir, "current_tasks.md")
PROGRESS_PATH = os.path.join(resolved_tasks_dir, "task_progress.md")
HISTORY_LEDGER_PATH = os.path.join(resolved_tasks_dir, "history_ledger.md")
DB_PATH = os.path.join(resolved_assets_dir, "piper_memory.db")

_last_known_objects = set()
_current_pan = 90.0
_current_tilt = 70.0
_global_node_instance = None
_latest_sketch_filename = None

# ==========================================================================
# SQLITE PERSISTENCE INITIALIZATION
# ==========================================================================
def init_database():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detected_objects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            xmin REAL, ymin REAL, xmax REAL, ymax REAL,
            center_x REAL, center_y REAL
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_objects_timestamp ON detected_objects(timestamp)")
    conn.commit()
    conn.close()

init_database()

def log_detection_to_db(timestamp, label, confidence, bbox=None, centroid=None):
    try:
        xmin, ymin, xmax, ymax = bbox if bbox else (None, None, None, None)
        cx, cy = centroid if centroid else (None, None)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO detected_objects (timestamp, label, confidence, xmin, ymin, xmax, ymax, center_x, center_y)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, label, confidence, xmin, ymin, xmax, ymax, cx, cy))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DATABASE ERROR] Failed to write telemetry record: {e}")

# --------------------------------------------------------------------------
# FLASK WEB ENDPOINTS
# --------------------------------------------------------------------------
@_app.route("/")
def index():
    return render_template("index.html")

@_app.route("/video_feed")
def video_feed():
    def _generate_stream():
        global _latest_frame_jpeg
        while True:
            if _latest_frame_jpeg is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + _latest_frame_jpeg + b"\r\n")
            time.sleep(0.04)
    return Response(_generate_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

@_app.route("/api/state", methods=["GET"])
def get_state():
    global _system_state_snapshot
    payload = {
        "state": _system_state_snapshot["state"],
        "active_task": _system_state_snapshot["active_task"],
        "objects": _system_state_snapshot["objects"],
        "matrix_report": _system_state_snapshot["matrix_report"],
        "latest_sketch": _latest_sketch_filename,
        "boxes": []
    }
    raw_cache = _system_state_snapshot.get("raw_boxes_cache", [])
    for label, confidence, bbox, centroid in raw_cache:
        if bbox and None not in bbox:
            payload["boxes"].append({
                "label": label, "confidence": int(confidence * 100),
                "xmin": bbox[0], "ymin": bbox[1], "xmax": bbox[2], "ymax": bbox[3]
            })
    return jsonify(payload)

@_app.route("/api/task_progress", methods=["GET"])
def get_task_progress():
    if not os.path.exists(PROGRESS_PATH):
        return jsonify({"content": "Awaiting pipeline task initialization sequences..."})
    try:
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return jsonify({"content": content if content else "Scratchpad clear. Waiting."})
    except Exception as e:
        return jsonify({"content": f"Error reading tracking file data: {str(e)}"})

@_app.route("/api/history_ledger", methods=["GET"])
def get_history_ledger():
    if not os.path.exists(HISTORY_LEDGER_PATH):
        return jsonify({"content": "No historical transactions registered yet."})
    try:
        with open(HISTORY_LEDGER_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        tail_content = "".join(lines[-50:])
        return jsonify({"content": tail_content if tail_content else "History log stack empty."})
    except Exception as e:
        return jsonify({"content": f"Error reading database ledger updates: {str(e)}"})

@_app.route("/api/history", methods=["GET"])
def get_detection_history():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT label, MAX(timestamp) as last_seen, COUNT(*) as occurrence_count, MAX(confidence) as last_conf
            FROM detected_objects GROUP BY label ORDER BY last_seen DESC LIMIT 8
        """)
        rows = cursor.fetchall()
        conn.close()
        
        history_list = []
        for row in rows:
            local_time_str = time.strftime('%H:%M:%S', time.localtime(row[1]))
            raw_conf = row[3] if row[3] is not None else 1.0
            history_list.append({"label": row[0], "last_seen": local_time_str, "count": row[2], "confidence": int(raw_conf * 100)})
        return jsonify({"history": history_list})
    except Exception as e:
        return jsonify({"history": [], "error": str(e)})

@_app.route("/api/jog", methods=["POST"])
def manual_jog_servo():
    global _current_pan, _current_tilt, _global_node_instance
    data = request.json or {}
    direction = data.get("direction", "").lower()
    step = float(data.get("step", 5.0))

    if direction == "left": _current_pan += step
    elif direction == "right": _current_pan -= step
    elif direction == "up": _current_tilt -= step    
    elif direction == "down": _current_tilt += step  
    
    _current_pan = max(0.0, min(180.0, _current_pan))
    _current_tilt = max(0.0, min(180.0, _current_tilt))

    if _global_node_instance:
        jog_cmd = Vector3()
        jog_cmd.x = _current_pan; jog_cmd.y = _current_tilt; jog_cmd.z = 1.0
        _global_node_instance.servo_pub.publish(jog_cmd)
        return jsonify({"status": "success", "pan": _current_pan, "tilt": _current_tilt})
    return jsonify({"status": "error", "message": "ROS instance mapping unavailable."})

# --------------------------------------------------------------------------
# ROS 2 CORE INTERFACE NODE
# --------------------------------------------------------------------------
class UM790DashboardNode(Node):
    def __init__(self):
        super().__init__('um790_dashboard_node')
        global _global_node_instance
        _global_node_instance = self
        self.data_lock = threading.Lock()

        # Communications Infrastructure Mapping
        self.img_sub = self.create_subscription(CompressedImage, '/piper/camera0/image_raw/compressed', self._image_callback, 10)
        self.object_sub = self.create_subscription(String, '/piper/perception/tracked_objects_json', self._object_callback, 10)
        self.servo_pub = self.create_publisher(Vector3, '/piper/neck/set_position', 10)

        # Threaded Watchers
        threading.Thread(target=self._sketchbook_filesystem_watcher, daemon=True).start()
        
        # Trigger single-shot automatic matrix mapping 10.0 seconds post-initialization
        self.matrix_scan_timer = self.create_timer(10.0, self._trigger_initial_matrix_scan)

    def _trigger_initial_matrix_scan(self):
        self.matrix_scan_timer.cancel()
        threading.Thread(target=self.execute_spatial_scan, daemon=True).start()

    def _sketchbook_filesystem_watcher(self):
        global _latest_sketch_filename
        # 💡 FIX: Target the active live generation path directly
        sketch_dir = "/home/steve/piper_assistant/src/piper_brain/piper_brain/assets/sketchbook"
        while True:
            try:
                if os.path.exists(sketch_dir):
                    files = [f for f in os.listdir(sketch_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if files:
                        files.sort(key=lambda x: os.path.getmtime(os.path.join(sketch_dir, x)), reverse=True)
                        with self.data_lock:
                            _latest_sketch_filename = files[0]
            except Exception:
                pass
            time.sleep(3.0)

    def _image_callback(self, msg):
        global _latest_frame_jpeg
        try:
            _latest_frame_jpeg = msg.data.tobytes()
        except Exception as e:
            self.get_logger().error(f"❌ [IMAGE STREAM ERROR] Buffer crash: {str(e)}")

    def _object_callback(self, msg):
        self._parse_object_payload(msg.data)

    def execute_spatial_scan(self):
        global _current_pan, _current_tilt, _system_state_snapshot
        with self.data_lock:
            home_p = _current_pan; home_t = _current_tilt
            
        waypoints = [
            {"id": "P1_Top_Left",     "p": home_p + 20.0, "t": home_t + 10.0},
            {"id": "P2_Top_Right",    "p": home_p - 20.0, "t": home_t + 10.0},
            {"id": "P3_Bottom_Right", "p": home_p - 20.0, "t": home_t - 10.0},
            {"id": "P4_Bottom_Left",  "p": home_p + 20.0, "t": home_t - 10.0}
        ]
        
        for wp in waypoints:
            wp["p"] = max(0.0, min(180.0, wp["p"]))
            wp["t"] = max(0.0, min(180.0, wp["t"]))
            with self.data_lock:
                _current_pan = wp['p']; _current_tilt = wp['t']
            
            cmd = Vector3()
            cmd.x = wp['p']; cmd.y = wp['t']; cmd.z = 1.0
            self.servo_pub.publish(cmd)
            time.sleep(2.0)
            
            with self.data_lock:
                current_seen = list(_system_state_snapshot.get("objects", []))
                formatted_seen = [item.upper() for item in current_seen]
                _system_state_snapshot["matrix_report"][wp['id']] = formatted_seen if formatted_seen else ["CLEAR"]

        with self.data_lock:
            _current_pan = home_p; _current_tilt = home_t
        cmd = Vector3()
        cmd.x = home_p; cmd.y = home_t; cmd.z = 1.0
        self.servo_pub.publish(cmd)

    def _parse_object_payload(self, raw_string_data):
        global _system_state_snapshot, _last_known_objects
        try:
            telemetry_data = json.loads(raw_string_data)
            detected_objects = telemetry_data if isinstance(telemetry_data, list) else telemetry_data.get("objects", [])
            
            cleaned_labels = []; processed_objects_for_db = []; current_frame_objects = set()
            label_parser = re.compile(r"([a-zA-Z0-9_\s\-]+)(?:\s*\((\d+)%\))?")

            for obj in detected_objects:
                raw_label = obj.get("label", "unknown")
                match = label_parser.match(raw_label)
                if match:
                    clean_label = match.group(1).strip().lower()
                    pct = match.group(2)
                    confidence = float(pct) / 100.0 if pct else float(obj.get("confidence", 1.0))
                else:
                    clean_label = raw_label; confidence = float(obj.get("confidence", 1.0))

                cleaned_labels.append(clean_label)
                current_frame_objects.add(clean_label)
                
                xmin = obj.get("xmin"); ymin = obj.get("ymin"); xmax = obj.get("xmax"); ymax = obj.get("ymax")
                cx = (xmin + xmax) / 2.0 if None not in (xmin, xmax) else None
                cy = (ymin + ymax) / 2.0 if None not in (ymin, ymax) else None
                processed_objects_for_db.append((clean_label, confidence, (xmin, ymin, xmax, ymax), (cx, cy)))

            unique_labels = list(set(cleaned_labels))
            _last_known_objects = current_frame_objects

            with self.data_lock:
                _system_state_snapshot["objects"] = unique_labels
                _system_state_snapshot["raw_boxes_cache"] = processed_objects_for_db
                _system_state_snapshot["state"] = "TEAMING" if "person" in unique_labels else "SOLO"
                _system_state_snapshot["active_task"] = "Collaborator tracked via Edge YOLO arrays." if "person" in unique_labels else "System running autonomously."

            for label, conf, bbox, centroid in processed_objects_for_db:
                log_detection_to_db(time.time(), label, conf, bbox, centroid)
        except Exception:
            pass

def run_ros_loop(args=None):
    rclpy.init(args=args)
    node = UM790DashboardNode()
    try:
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

def main(args=None):
    # 💡 FIX: Safely channel launch arguments into thread instantiation context
    ros_thread = threading.Thread(target=run_ros_loop, args=(args,), daemon=True)
    ros_thread.start()
    time.sleep(0.2)
    print("🚀 Piper Perception Web Console online at http://0.0.0.0:5000")
    _app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()