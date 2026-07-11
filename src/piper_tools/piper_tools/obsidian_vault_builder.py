#!/usr/bin/env python3
import os
import re

def build_obsidian_vault():
    source_path = "/home/steve/piper_assistant/src/piper_tools/piper_tools/assets/model_concepts.md"
    vault_dir = "/home/steve/piper_assistant/src/piper_tools/piper_tools/assets/world_model_vault"
    
    if not os.path.exists(source_path):
        print(f"❌ Source file not found at {source_path}")
        return

    os.makedirs(vault_dir, exist_ok=True)
    
    with open(source_path, 'r') as f:
        content = f.read()

    # Split the file by markdown horizontal rules or headers to extract the concepts
    concepts = re.split(r'---|\n## ', content)
    
    concept_links = []
    
    for concept in concepts:
        if not concept.strip() or "10 High-Level Architectural Concepts" in concept:
            continue
            
        # Clean up and extract the concept title
        lines = concept.strip().split('\n')
        title_raw = lines[0].replace('## ', '').strip()
        # Remove leading numbers like "1. " for cleaner file naming
        title = re.sub(r'^\d+\.\s*', '', title_raw)
        
        if not title:
            continue
            
        concept_links.append(title)
        concept_body = '\n'.join(lines[1:])
        
        # ✨ FIX: Sanitize the title string for the filesystem filename ONLY
        # This preserves the pristine title inside the note text but keeps the file safe.
        safe_filename = title.replace('/', '-').replace(':', '-').strip()
        
        # Parse URLs and turn them into Obsidian Backlinks/Footnotes if desired
        note_content = f"# {title}\n\n## Meta Navigation\n* **Context**: [[World Model Blueprint]]\n\n## Content\n{concept_body.strip()}"
        
        # Write individual concept node file using the safe filename
        note_path = os.path.join(vault_dir, f"{safe_filename}.md")
        with open(note_path, 'w') as nf:
            nf.write(note_content)
        print(f"📄 Created Node: {safe_filename}.md")

    # Create the Master Dashboard File
    blueprint_content = (
        "# 🧠 World Model Core Blueprint\n\n"
        "## 🏗️ Architectural Concept Map\n"
        "This master view outlines the core operational pillars compiled across our research database. "
        "Click any node to explore its internal mechanisms, mathematical frameworks, and source code references.\n\n"
        "### Operational Pillars\n"
    )
    for link in concept_links:
        blueprint_content += f"* [[{link}]]\n"
        
    with open(os.path.join(vault_dir, "World Model Blueprint.md"), 'w') as bf:
        bf.write(blueprint_content)
        
    print("✨ Master Blueprint Node generated successfully.")
    print(f"👉 Point your Obsidian app to open this folder as a vault: {vault_dir}")

if __name__ == "__main__":
    build_obsidian_vault()