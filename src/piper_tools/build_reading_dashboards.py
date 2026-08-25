#!/usr/bin/env python3
import os
import glob

def generate_scrollable_dashboards():
    vault_dir = "/home/steve/piper_assistant/src/piper_tools/piper_tools/assets/world_model_vault"
    sources_dir = os.path.join(vault_dir, "Sources")
    dashboards_dir = os.path.join(vault_dir, "Dashboards")
    
    os.makedirs(dashboards_dir, exist_ok=True)
    
    vault_dir = "/home/steve/piper_assistant/src/piper_tools/piper_tools/assets/world_model_vault"
    sources_dir = os.path.join(vault_dir, "Sources")
    dashboards_dir = os.path.join(vault_dir, "Dashboards")
    
    os.makedirs(dashboards_dir, exist_ok=True)
    
    # ✨ DYNAMIC LOOKUP: Read concepts directly from the previously generated file
    source_markdown = "/home/steve/piper_assistant/src/piper_tools/piper_tools/assets/model_concepts.md"
    concepts = []
    
    if os.path.exists(source_markdown):
        import re
        with open(source_markdown, 'r', encoding='utf-8') as sf:
            for line in sf:
                # Find all lines starting with "## " headers
                if line.startswith("## "):
                    # Extract the raw title name
                    title_raw = line.replace("## ", "").strip()
                    # Strip out numbers like "1. " to perfectly match the vault names
                    clean_title = re.sub(r'^\d+\.\s*', '', title_raw)
                    if clean_title:
                        concepts.append(clean_title)
    
    # Fallback to the original baseline if the file isn't populated yet
    if not concepts:
        print("⚠️ Warning: model_concepts.md empty or missing. Falling back to default list.")
        concepts = ["Latent-Space Predictive Architectures (JEPA)", "Recurrent State-Space Models (RSSM) & Imagined Rollouts"]

    # Initialize index mapping lists dynamically
    concept_buckets = {concept: [] for concept in concepts}
    
    # Initialize index mapping lists
    concept_buckets = {concept: [] for concept in concepts}
    
    # 2. Scan every generated source file in the vault to find its concept mappings
    source_files = glob.glob(os.path.join(sources_dir, "*.md"))
    print(f"🔍 Reading through {len(source_files)} source nodes to compile reading frames...")
    
    for file_path in source_files:
        filename = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check which concept tags are present inside the file's metadata links
        for concept in concepts:
            if f"[[{concept}]]" in content:
                # Store the clean markdown name without extension for embedding syntax
                note_name = filename.replace(".md", "")
                concept_buckets[concept].append(note_name)

    # 3. Generate the 10 intermediate dashboard reading frames
    for concept, linked_notes in concept_buckets.items():
        dashboard_filename = f"Dashboard - {concept}.md"
        dashboard_path = os.path.join(dashboards_dir, dashboard_filename)
        
        # Build a structured, beautiful scrollable feed document
        dashboard_md = (
            f"# 🎛️ Reading Dashboard: {concept}\n\n"
            f"## Meta Navigation\n"
            f"* **Master Registry**: [[World Model Blueprint]]\n"
            f"* **Core Concept Definitions**: [[{concept}]]\n"
            f"* **Total Ingested Supporting Resources**: `{len(linked_notes)}` references\n\n"
            f"---\n\n"
            f"## 📚 Continuous Data Feed Summary\n"
            f"The sections below stream directly from your active database scrapes. "
            f"Scroll down to read them sequentially without leaving this viewport.\n\n"
        )
        
        if not linked_notes:
            dashboard_md += "*No background data scrapes have been indexed for this paradigm yet.*\n"
        else:
            for idx, note_name in enumerate(sorted(linked_notes)):
                dashboard_md += f"### 📑 Resource entry [{idx + 1}/{len(linked_notes)}]\n"
                # This syntax tells Obsidian to render the full file text inline!
                dashboard_md += f"![[{note_name}]]\n\n"
                dashboard_md += "---\n\n"
                
        with open(dashboard_path, 'w', encoding='utf-8') as df:
            df.write(dashboard_md)
            
        print(f"✨ Compiled Continuous Dashboard viewport: {dashboard_filename} ({len(linked_notes)} embedded feeds)")

if __name__ == "__main__":
    generate_scrollable_dashboards()