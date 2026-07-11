#!/usr/bin/env python3
import os
import sys
import sqlite3
import requests
import logging

sys.path.insert(0, '/home/steve/piper_assistant/.venv/lib/python3.12/site-packages')

# 💡 Re-routed pathing targets assets sub-directory
DB_PATH = "/home/steve/piper_assistant/src/piper_tools/piper_tools/assets/piper_knowledge_base.db"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def execute_frontier_cloud_task(task_description: str, api_key: str, workspace_root: str) -> bool:
    """ Gathers full SQLite context and executes arbitrary high-reasoning tasks using GLM-5.2 via OpenRouter. """
    if not api_key or api_key == "your_openrouter_api_key_here":
        logging.error("❌ [FRONTIER] OpenRouter API key missing or invalid.")
        return False

    # Pull the absolute full Knowledge Base context state
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT url, topic, content FROM topical_data;")
    rows = cursor.fetchall()
    conn.close()
    
    kb_payload = ""
    for idx, (url, topic, content) in enumerate(rows):
        kb_payload += f"\n--- KB RECORD {idx+1} ---\nURL: {url}\nTOPIC: {topic}\nCONTENT: {content}\n"

    system_instruction = (
        "You are a Principal AI Systems Architect and Data Engineer. Your task is to execute deep-reasoning "
        "synthesis, mindmapping, or development directives across a provided local database context.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "- Focus deeply on structural mechanics, technical relationships, paradigms, and execution definitions.\n"
        "- Respond strictly with the markdown-formatted content or documentation requested by the task.\n"
        "- Do not include conversational introductory or concluding chat prose.\n"
        # ✨ NEW STRICT INSTRUCTION FOR VAULT HYGIENE
        "- IMPORTANT: All markdown headers (e.g., '## Concept Name') will be parsed directly into operating system filenames. "
        "Therefore, you MUST ensure that no header text contains illegal filename characters, specifically forward slashes (/), "
        "backslashes (\\), or colons (:). Use hyphens (-), spaces, or parentheses instead (e.g., use 'H-JEPA and ThinkJEPA' instead of 'H-JEPA / ThinkJEPA')."
    )

    logging.info(f"🧠 [FRONTIER] Packaging {len(rows)} KB records for GLM-5.2 reasoning pass...")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "z-ai/glm-5.2",
        "provider": {"allow_fallbacks": False},
        "extra_body": {
            "enable_thinking": True,
            "reasoning_effort": "xhigh"
        },
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Task Directive to Execute:\n{task_description}\n\n=== SOURCE KNOWLEDGE BASE ===\n{kb_payload}"}
        ]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=450)
        
        if response.status_code == 200:
            markdown_output = response.json()['choices'][0]['message']['content']
            
            # 💡 Relocate generation file directly into your tool asset repository
            output_file_path = "/home/steve/piper_assistant/src/piper_tools/piper_tools/assets/model_concepts.md"
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            
            with open(output_file_path, "w") as f:
                f.write(markdown_output)
                
            logging.info(f"✨ [FRONTIER] Task execution successfully committed to disk: {output_file_path}")
            return True
        else:
            logging.error(f"❌ [FRONTIER] OpenRouter API Error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logging.error(f"❌ [FRONTIER] Failed to communicate with OpenRouter gateway: {e}")
        return False