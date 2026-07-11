#!/usr/bin/env python3
import os
import sqlite3
import logging
import requests
import mimetypes
import random  
from typing import List, Dict, Any
from ddgs import DDGS
import trafilatura

# Sibling imports from the same package
from piper_tools.research_parser import PiperResearchParser
from piper_tools.kb_manager import DB_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PiperResearchEngine:
    def __init__(self, max_results: int = 5): 
        self.max_results = max_results
        self.parser = PiperResearchParser()
        
        # 💡 Redirected Local media cache directory to tools assets
        self.assets_dir = "/home/steve/piper_assistant/src/piper_tools/piper_tools/assets/research"
        os.makedirs(self.assets_dir, exist_ok=True)

    def is_url_duplicate(self, url: str) -> bool:
        """ 🛡️ Proactively inspects the ledger to see if a URL is already known """
        if not url:
            return False
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT 1 FROM topical_data WHERE url = ?", (url,))
            exists = cursor.fetchone() is not None
            return exists
        except Exception as e:
            logging.error(f"⚠️ [DATABASE] Error checking URL duplication state: {e}")
            return False
        finally:
            conn.close()

    def execute_web_search(self, query: str) -> List[Dict[str, str]]:
        """ Programmatically queries DuckDuckGo, shuffles results, and filters known links """
        logging.info(f"🔍 [RESEARCH] Querying web indices for: '{query}'")
        discovered_targets = []
        
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=self.max_results)
                if results:
                    for r in results:
                        discovered_targets.append({
                            "title": r.get("title", "Untitled"),
                            "url": r.get("href", ""),
                            "snippet": r.get("body", "")
                        })
        except Exception as e:
            logging.error(f"❌ [RESEARCH] DuckDuckGo search execution failed: {e}")
            
        logging.info(f"🎯 [RESEARCH] Found {len(discovered_targets)} raw candidate URLs.")
        
        if len(discovered_targets) > 1:
            random.shuffle(discovered_targets)
            logging.info("🎲 [RESEARCH] Randomly shuffled search targets to maximize discovery path variation.")

        return discovered_targets

    def scrape_url_content(self, url: str) -> Dict[str, Any]:
        """ Fetches HTML payload and extracts clean structural text """
        logging.info(f"📥 [RESEARCH] Scraping target resource: {url}")
        result = {"url": url, "raw_text": "", "success": False}
        
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                return result
            
            clean_text = trafilatura.extract(downloaded, output_format='txt')
            if clean_text:
                result["raw_text"] = clean_text
                result["success"] = True
                logging.info(f"✅ [RESEARCH] Successfully extracted {len(clean_text)} characters.")
        except Exception as e:
            logging.error(f"❌ [RESEARCH] Error occurred while parsing {url}: {e}")
            
        return result

    def _download_diagram_asset(self, img_url: str, diagram_id: int) -> str:
        """ Downloads web image data locally and returns file path references """
        if not img_url or img_url.lower() == "unknown" or not img_url.startswith("http"):
            return "Unknown"

        try:
            logging.info(f"📥 [ASSET MANAGER] Localizing binary diagram asset: {img_url}")
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            response = requests.get(img_url, timeout=15, headers=headers, stream=True)
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '').split(';')[0]
                ext = mimetypes.guess_extension(content_type) or '.jpg'
                
                local_filename = f"diagram_{diagram_id}{ext}"
                local_path = os.path.join(self.assets_dir, local_filename)
                
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
                logging.info(f"💾 [ASSET MANAGER] Diagram cached locally at: {local_path}")
                return local_path
        except Exception as e:
            logging.error(f"❌ [ASSET MANAGER] Failed to download diagram asset: {e}")
            
        return "Unknown"

    def _insert_structured_payload(self, data: Dict[str, Any]) -> None:
        """ Unpacks LLM JSON schema, saves images, commits to SQLite, and mirrors to Obsidian """
        if not data or "topical_data" not in data:
            return

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            top_info = data["topical_data"]
            cursor.execute('''
                INSERT OR IGNORE INTO topical_data (topic, category, url, keywords, content)
                VALUES (?, ?, ?, ?, ?)
            ''', (top_info.get("topic"), top_info.get("category"), top_info.get("url"), top_info.get("keywords"), top_info.get("content")))
            
            topic_id = cursor.lastrowid

            if not topic_id:
                logging.info(f"⏭️ [DATABASE] URL already mapped in ledger. Skipping: {top_info.get('url')}")
                conn.close()
                return

            # 2. People Insert
            for person in data.get("people", []):
                if not person.get("name"):
                    continue
                cursor.execute("SELECT id FROM people WHERE name = ?", (person["name"],))
                row = cursor.fetchone()
                person_id = row[0] if row else None
                
                if not person_id:
                    cursor.execute('''
                        INSERT OR IGNORE INTO people (name, field, dates, url, contribution)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (person.get("name"), person.get("field"), person.get("dates"), person.get("url"), person.get("contribution")))
                    person_id = cursor.lastrowid
                
                cursor.execute('INSERT OR IGNORE INTO topic_people (topic_id, person_id) VALUES (?, ?)', (topic_id, person_id))

            # 3. Diagrams Insert & Download
            for diagram in data.get("diagrams", []):
                cursor.execute('''
                    INSERT OR IGNORE INTO diagrams (topic, category, url, image_path, notes)
                    VALUES (?, ?, ?, ?, ?)
                ''', (diagram.get("topic"), diagram.get("category"), diagram.get("url"), "Pending Download", diagram.get("notes")))
                diagram_id = cursor.lastrowid
                
                local_asset_path = self._download_diagram_asset(diagram.get("url"), diagram_id)
                cursor.execute('UPDATE diagrams SET image_path = ? WHERE id = ?', (local_asset_path, diagram_id))
                cursor.execute('INSERT OR IGNORE INTO topic_diagrams (topic_id, diagram_id) VALUES (?, ?)', (topic_id, diagram_id))

            # 4. Self-Referencing Topic Links Map
            for alternate_url in data.get("suggested_related_links", []):
                cursor.execute("SELECT id FROM topical_data WHERE url = ?", (alternate_url,))
                matched_topic = cursor.fetchone()
                if matched_topic:
                    cursor.execute('INSERT OR IGNORE INTO topic_relationships (source_topic_id, target_topic_id) VALUES (?, ?)', (topic_id, matched_topic[0]))

            conn.commit()
            logging.info("✨ [DATABASE] Transaction committed successfully to Knowledge Base tables.")

            # ==================================================================
            # 🪟 LIVE OBSIDIAN VAULT BRIDGE
            # ==================================================================
            try:
                # 💡 REDIRECTED: Shift target path out of piper_brain tasks into tools assets folder
                vault_sources_dir = "/home/steve/piper_assistant/src/piper_tools/piper_tools/assets/world_model_vault/Sources"
                
                # Proactively ensure the parent chain exists before attempting file writes
                os.makedirs(vault_sources_dir, exist_ok=True)
                    
                from piper_tools.vault_data_linker import match_concepts, sanitize_filename
                
                matched_tags = match_concepts(top_info.get("content", ""), top_info.get("keywords", ""))
                link_strings = ", ".join([f"[[{tag}]]" for tag in matched_tags])
                
                live_note = (
                    f"# Source: {top_info.get('topic')}\n\n"
                    f"## Database Metadata\n"
                    f"* **Database Record ID**: `{topic_id}`\n"
                    f"* **Original Source URL**: {top_info.get('url')}\n"
                    f"* **Scrape Ingestion Date**: `Live Pipeline Stream`\n"
                    f"* **Local Category**: `{top_info.get('category')}`\n"
                    f"* **Architectural Concepts Supported**: {link_strings}\n\n"
                    f"## Extracted Content Summary\n"
                    f"{top_info.get('content', '').strip()}\n"
                )
                
                safe_filename = sanitize_filename(f"Source_{topic_id}_{top_info.get('topic')}")
                with open(os.path.join(vault_sources_dir, f"{safe_filename}.md"), 'w', encoding='utf-8') as f:
                    f.write(live_note)
                    
                logging.info(f"⚡ [OBSIDIAN BRIDGE] Streaming unique node directly to vault: {safe_filename}.md")
            except Exception as bridge_err:
                logging.error(f"⚠️ [OBSIDIAN BRIDGE] Automated edge generation failed: {bridge_err}")

        except Exception as e:
            conn.rollback()
            logging.error(f"❌ [DATABASE] Transaction failed: {e}")
        finally:
            conn.close()

    def perform_full_research_sweep(self, topic_query: str) -> None:
        """ Orchestrates the complete pipeline: Search, Scrape, Parse, and Save """
        logging.info(f"🚀 [RESEARCH ENGINE] Launching full sweep for: '{topic_query}'")
        search_results = self.execute_web_search(topic_query)
        
        for item in search_results:
            if self.is_url_duplicate(item["url"]):
                logging.info(f"⏭️ [PRE-FILTER] URL already captured in ledger. Skipping sweep path: {item['url']}")
                continue

            scraped_content = self.scrape_url_content(item["url"])
            if scraped_content["success"]:
                structured_json = self.parser.parse_scraped_text(scraped_content["raw_text"], item["url"])
                
                if structured_json and "topical_data" in structured_json:
                    structured_json["topical_data"]["url"] = item["url"]
                
                self._insert_structured_payload(structured_json)

if __name__ == "__main__":
    engine = PiperResearchEngine(max_results=5)
    engine.perform_full_research_sweep("ROS 2 Jazzy Python virtual environments standard guidelines")