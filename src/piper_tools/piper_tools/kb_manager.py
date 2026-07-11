#!/usr/bin/env python3
import os
import sqlite3
import logging

# 💡 Redirection Target: Assets folder inside piper_tools
DB_PATH = "/home/steve/piper_assistant/src/piper_tools/piper_tools/assets/piper_knowledge_base.db"

def initialize_database():
    """ Creates a pristine relational schema for Piper 3.0 Research Data """
    # Ensure asset directory structure exists safely
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    logging.info(f"💾 Initializing Piper Knowledge Base SQLite Database at: {DB_PATH}")

    # 1. Core Topical Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS topical_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            category TEXT,
            url TEXT,
            keywords TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            date_extracted DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 2. Core People Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            field TEXT,
            dates TEXT,
            url TEXT,
            contribution TEXT,
            date_extracted DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 3. Core Diagrams Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diagrams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT,
            category TEXT,
            url TEXT,
            image_path TEXT,
            notes TEXT,
            date_extracted DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 4. Relational Map: Topic-to-Topic
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS topic_relationships (
            source_topic_id INTEGER,
            target_topic_id INTEGER,
            PRIMARY KEY (source_topic_id, target_topic_id),
            FOREIGN KEY (source_topic_id) REFERENCES topical_data(id) ON DELETE CASCADE,
            FOREIGN KEY (target_topic_id) REFERENCES topical_data(id) ON DELETE CASCADE
        )
    ''')

    # 5. Relational Map: Topic-to-People
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS topic_people (
            topic_id INTEGER,
            person_id INTEGER,
            PRIMARY KEY (topic_id, person_id),
            FOREIGN KEY (topic_id) REFERENCES topical_data(id) ON DELETE CASCADE,
            FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE
        )
    ''')

    # 6. Relational Map: Topic-to-Diagram
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS topic_diagrams (
            topic_id INTEGER,
            diagram_id INTEGER,
            PRIMARY KEY (topic_id, diagram_id),
            FOREIGN KEY (topic_id) REFERENCES topical_data(id) ON DELETE CASCADE,
            FOREIGN KEY (diagram_id) REFERENCES diagrams(id) ON DELETE CASCADE
        )
    ''')

    conn.commit()
    conn.close()
    logging.info("✨ Relational Knowledge Base schema deployed successfully.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    initialize_database()