#!/usr/bin/env python3
import sqlite3
import re
from collections import Counter

# 💡 Absolute path to the verified 50-record database
DB_PATH = "/home/steve/piper_assistant/src/piper_brain/piper_brain/tasks/piper_knowledge_base.db"

def extract_expansion_terms():
    original_terms = {
        "research", "world", "model", "architectures", "ai", "systems", "including",
        "jepa", "models", "sora", "latent", "muzero", "model-based", "reinforcement",
        "learning", "strategies", "v-jepa", "i-jepa", "self-supervised", "vision",
        "evaluation", "alphazero", "planning", "learned", "dynamic", "joint",
        "embedding", "predictive", "architecture", "implementation", "deep", "dive",
        "video", "prediction", "scaling", "laws", "tree", "search", "dynamics",
        "coding", "hidden", "state", "representations"
    }

    stop_words = {
        "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "is", 
        "for", "with", "that", "this", "by", "from", "on", "as", "an", "it", "its",
        "are", "was", "were", "be", "been", "which", "using", "used", "data", "results"
    }

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Pulling the content fields directly from your 50 records
        cursor.execute("SELECT content, keywords, topic FROM topical_data")
        rows = cursor.fetchall()
        conn.close()
    except Exception as e:
        print(f"❌ Database read failed: {e}")
        return

    word_counter = Counter()
    
    for content, keywords, topic in rows:
        # Lowercase and combine text blocks to find implicit technical vocab
        combined_text = f"{content or ''} {keywords or ''} {topic or ''}".lower()
        words = re.findall(r'\b[a-z]{3,15}\b', combined_text) 
        
        for word in words:
            if word not in stop_words and word not in original_terms:
                word_counter[word] += 1

    print("\n📊 --- TOP EMERGING UNIQUE TERMS IN DISCOVERED LEDGER ---")
    if not word_counter:
        print(" No unique terms extracted. Check field text values.")
    for term, count in word_counter.most_common(20):
        print(f" * {term:<15} (Found {count} times)")

if __name__ == "__main__":
    extract_expansion_terms()
