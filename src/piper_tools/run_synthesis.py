#!/usr/bin/env python3
import os
import sys

# Ensure the local virtual environment packages are discoverable
sys.path.insert(0, '/home/steve/piper_assistant/.venv/lib/python3.12/site-packages')

# 💡 FIX: Inject the exact directory path so neighbor modules resolve cleanly
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

# Now the local package import will resolve perfectly
from concept_extractor import execute_frontier_cloud_task

# Configure Workspace Environment Paths
API_KEY = os.getenv("OPENROUTER_API_KEY")
WORKSPACE_ROOT = "/home/steve/piper_assistant/src/piper_tools/piper_tools/assets"

directive = (
    "Extract exactly 10 high-level architectural concepts defining world model mechanics based on the database. "
    "For each concept, provide a high-level abstract definition, a deep technical explanation of how it handles "
    "physics simulation or predictive planning, and a clean reference list of the source URLs from our database "
    "records that directly align with or support it."
)

if __name__ == "__main__":
    print("🚀 Invoking GLM-5.2 via OpenRouter (with xhigh reasoning effort enabled)...")
    success = execute_frontier_cloud_task(directive, API_KEY, WORKSPACE_ROOT)
    if success:
        print("✅ Core architecture file successfully generated as model_concepts.md!")
    else:
        print("❌ Cloud synthesis pass failed.")
