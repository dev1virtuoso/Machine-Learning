import re
import sqlite3
from typing import List, Dict, Any, Optional

class KnowledgeEngine:

    def __init__(self, db_path: str = "knowledge_base.db") -> None:
        self.db_path = db_path
        self._init_db()
        self._cached_entities: Optional[List[str]] = None

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE COLLATE NOCASE,
                wiki_ref TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id INTEGER,
                key TEXT,
                value TEXT,
                wiki_ref TEXT,
                FOREIGN KEY(entity_id) REFERENCES entities(id)
            );
            """
        )
        conn.commit()
        conn.close()

    def add_entity(self, name: str, wiki_ref: Optional[str] = None) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO entities (name, wiki_ref) VALUES (?, ?);",
            (name.strip().lower(), wiki_ref),
        )
        conn.commit()
        conn.close()

    def add_fact(
        self,
        entity_name: str,
        key: str,
        value: str,
        wiki_ref: Optional[str] = None,
    ) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM entities WHERE name = ?;",
            (entity_name.strip().lower(),),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Entity '{entity_name}' not found.")
        entity_id = row[0]
        cur.execute(
            """
            INSERT INTO facts (entity_id, key, value, wiki_ref)
            VALUES (?, ?, ?, ?);
            """,
            (entity_id, key, value, wiki_ref),
        )
        conn.commit()
        conn.close()

    def query_fact(
        self, entity_name: str
    ) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT f.key, f.value, f.wiki_ref
            FROM facts f
            JOIN entities e ON f.entity_id = e.id
            WHERE e.name = ?;
            """,
            (entity_name.strip().lower(),),
        )
        rows = cur.fetchall()
        conn.close()
        return [
            {"key": k, "value": v, "wiki_ref": w} for k, v, w in rows
        ] or []

    def extract_entities(self, text: str) -> List[str] | None:
        """
        Very lightweight N‑gram matcher that returns the longest
        matching entity names.  Returns ``None`` if nothing is found.
        """
        if self._cached_entities is None:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("SELECT name FROM entities;")
            self._cached_entities = [row[0] for row in cur.fetchall()]
            conn.close()

        text_lower = text.lower()
        found = []
        for ent in sorted(self._cached_entities, key=len, reverse=True):
            if re.search(rf"\b{re.escape(ent)}\b", text_lower):
                found.append(ent)
        return found or None
