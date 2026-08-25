#!/usr/bin/env python3
import os
import sys
import sqlite3
import re

sys.path.insert(0, '/home/steve/piper_assistant/.venv/lib/python3.12/site-packages')

# 💡 Database path pointing directly to assets
DB_PATH = "/home/steve/piper_assistant/src/piper_tools/piper_tools/assets/piper_knowledge_base.db"

def sanitize_filename(name: str) -> str:
    return re.sub(r'[\/:*?"<>|]', '', name)[:100].strip()

def match_concepts(content: str, keywords_str: str) -> list:
    mappings = []
    text_pool = (content + " " + keywords_str).lower()
    
    concept_rules = {
        "Latent-Space Predictive Architecture": ["latent", "embedding", "jepa", "forward dynamics", "predictor"],
        "Energy-Based Model State Plausibility Evaluation": ["energy", "ebm", "contrastive", "plausibility", "score"],
        "Imagination-Based Planning via Internal Simulation": ["imagination", "dreamer", "muzero", "planet", "trajectory", "rollout"],
        "Object-Centric Structured Partial Observability via Latent Masking": ["object-centric", "masking", "c-jepa", "causal-jepa", "observability"],
        "Hierarchical Multi-Level Planning Across Abstraction Tiers": ["hierarchical", "h-jepa", "abstraction", "tiers", "multi-level"],
        "Recurrent State-Space Models (RSSM) for Stochastic Dynamics": ["rssm", "stochastic", "recurrent state-space", "deterministic", "posterior"],
        "Action-Conditioned Co-Occurrence Matrix Factorization (Generalization Theory)": ["factorization", "matrix", "generalization", "spectral", "regret"],
        "Representational Collapse Prevention via Asymmetric Encoders and Variance Regularization": ["collapse", "vicreg", "covariance", "ema", "regularization"],
        "Counterfactual Reasoning in Learned Latent Dynamics": ["counterfactual", "what if", "causal understanding", "perturb"],
        "Diffusion-Based Generative World Simulation": ["diffusion", "diamond", "denoising", "sora", "cosmos", "multi-modal"]
    }
    
    for concept, keywords in concept_rules.items():
        if any(kw in text_pool for kw in keywords):
            mappings.append(concept)
            
    if not mappings:
        if "planning" in text_pool:
            mappings.append("Imagination-Based Planning via Internal Simulation")
        else:
            mappings.append("Latent-Space Predictive Architecture")
            
    return mappings

def export_and_link_db_to_vault():
    # 💡 Redirected vault directory target to assets
    vault_dir = "/home/steve/piper_assistant/src/piper_tools/piper_tools/assets/world_model_vault"
    sources_dir = os.path.join(vault_dir, "Sources")
    os.makedirs(sources_dir, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, topic, category, url, keywords, content, date_extracted FROM topical_data;")
    rows = cursor.fetchall()
    conn.close()
    
    print(f"📂 Found {len(rows)} database records to export and map...")
    
    for row in rows:
        db_id, topic, category, url, keywords, content, date_extracted = row
        if not topic or not content:
            continue
            
        matched_tags = match_concepts(content, keywords or "")
        link_strings = ", ".join([f"[[{tag}]]" for tag in matched_tags])
        
        source_note = (
            f"# Source: {topic}\n\n"
            f"## Database Metadata\n"
            f"* **Database Record ID**: `{db_id}`\n"
            f"* **Original Source URL**: {url}\n"
            f"* **Scrape Ingestion Date**: `{date_extracted}`\n"
            f"* **Local Category**: `{category}`\n"
            f"* **Architectural Concepts Supported**: {link_strings}\n\n"
            f"## Extracted Content Summary\n"
            f"{content.strip()}\n"
        )
        
        safe_title = sanitize_filename(f"Source_{db_id}_{topic}")
        file_path = os.path.join(sources_dir, f"{safe_title}.md")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(source_note)
            
    print(f"✨ Successfully exported and linked {len(rows)} files into the Vault under 'Sources/'!")

if __name__ == "__main__":
    export_and_link_db_to_vault()