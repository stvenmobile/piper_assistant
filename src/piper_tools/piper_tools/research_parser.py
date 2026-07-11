#!/usr/bin/env python3
import os  # Added for secure environment variable lookup
import json
import logging
import requests
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PiperResearchParser:
    # SECURED: Removed plaintext key and defaulted the argument to None
    def __init__(self, model_name: str = "z-ai/glm-5.2", openrouter_key: str = None):
        self.model_name = model_name
        # Look up the key from environmental context if not explicitly passed
        self.openrouter_key = openrouter_key or os.getenv("OPENROUTER_API_KEY")
        
        # Automatically route based on model type
        if "glm" in model_name.lower():
            self.api_url = "https://openrouter.ai/api/v1/chat/completions"
            logging.info(f"🌐 [LLM PARSER] Routing to Cloud Frontier Model via OpenRouter: {self.model_name}")
        else:
            self.api_url = "http://localhost:11434/api/chat"
            logging.info(f"🏠 [LLM PARSER] Routing to Local Instance via Ollama: {self.model_name}")

    def parse_scraped_text(self, raw_text: str, source_url: str) -> Dict[str, Any]:
        """ Uses a local or cloud LLM to extract structured entities matching our database schema """
        
        # 🚀 Cloud Horizon Optimization: If using GLM, skip truncation entirely!
        if "glm" in self.model_name.lower():
            processed_text = raw_text
            logging.info(f"🧠 [LLM PARSER] Sending complete text frame ({len(raw_text)} chars) to cloud engine...")
        else:
            MAX_PARSE_CHARS = 40000
            if len(raw_text) > MAX_PARSE_CHARS:
                logging.info(f"✂️ [LLM PARSER] Truncating massive text wall down to {MAX_PARSE_CHARS} characters for local window.")
                processed_text = raw_text[:MAX_PARSE_CHARS] + "\n\n[... Content Truncated for Local Context Window Efficiency ...]"
            else:
                processed_text = raw_text

        system_prompt = """You are an expert knowledge extraction agent. Analyze the provided text scraped from a website and output a strictly formatted JSON object that breaks down what was learned into a structured Knowledge Base scheme.

Your output must follow this JSON schema exactly:
{
  "topical_data": {
    "topic": "The main technical subject or guide name",
    "category": "High-level grouping string like Robotics, Programming, DevOps, AI",
    "keywords": "comma, separated, list, of, matching, tags",
    "content": "A high-retention summary of what was learned, keeping critical design rules, paths, or code instructions, but stripping out fluff."
  },
  "people": [
    {
      "name": "Full Name of relevant pioneer/researcher/author mentioned",
      "field": "Their field of study or focus",
      "dates": "Birth-Death if historical, or 'Living'",
      "url": "A reference URL or 'Unknown'",
      "contribution": "Summary of what they contributed or discovered"
    }
  ],
  "diagrams": [
    {
      "topic": "Context of the graphic",
      "category": "Type of diagram (e.g., Architecture Flow, Pinout Schematic, UML)",
      "url": "The explicit source URL string of the image if mentioned, or 'Unknown'",
      "notes": "What this diagram represents or shows"
    }
  ],
  "suggested_related_links": [
    "A list of raw URLs mentioned in the text that seem highly valuable to visit next"
  ]
}

Rules:
1. Return ONLY valid raw JSON. Do not wrap it in markdown code blocks (```json) or add introductory text.
2. If no People or Diagrams are mentioned, leave their respective arrays empty [].
"""

        user_content = f"Source URL: {source_url}\n\nScraped Web Content:\n{processed_text}"

        # Build payload based on API specification
        if "openrouter.ai" in self.api_url:
            headers = {
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1
            }
        else:
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1}
            }

        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=120)
            if response.status_code == 200:
                resp_json = response.json()
                if "openrouter.ai" in self.api_url:
                    raw_response = resp_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                else:
                    raw_response = resp_json.get("message", {}).get("content", "").strip()
                
                structured_data = json.loads(raw_response)
                logging.info("✅ [LLM PARSER] Successfully extracted schema JSON object via cloud endpoint.")
                return structured_data
            else:
                logging.error(f"❌ [LLM PARSER] API returned error code {response.status_code}: {response.text}")
        except Exception as e:
            logging.error(f"❌ [LLM PARSER] Failed to parse content via cloud LLM: {e}")

        return {}

if __name__ == "__main__":
    parser = PiperResearchParser()
    sample_text = "Dennis Ritchie created C at Bell Labs."
    res = parser.parse_scraped_text(sample_text, "[https://example.com](https://example.com)")
    print(json.dumps(res, indent=2))