import sqlite3
from pathlib import Path
from datetime import datetime, timezone
import json
from typing import Any, Optional, Dict, List, Tuple


UTC = timezone.utc

def now_iso() -> str:
    return datetime.now(UTC).isoformat()

class MemoryDB:
    def __init__(self, db_path: str = "./brain/thursday_memory.db") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        #Default connections here
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn
    
    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS facts (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
                );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tool_name TEXT,
                tool_args_json TEXT,
                tool_result_json TEXT,
                created_at TEXT NOT NULL
                );
            CREATE INDEX IF NOT EXISTS idx_messages_created_at
                ON messages(created_at);
            """)
            
    def upsert_fact(self, key: str, value: Any, confidence: float = 1.0, source: Optional[str] = None) -> None:
        value_json = json.dumps(value, ensure_ascii = False)
        ts = now_iso()
        with self._connect() as conn:
            conn.execute("""
            INSERT INTO facts(key, value_json, confidence, source, created_at, updated_at)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(key) DO UPDATE SET
                value_json = excluded.value_json,
                confidence = excluded.confidence,
                         source = excluded.source,
                         updated_at = excluded.updated_at
            """, (key, value_json, confidence, source, ts, ts))

    def upsert_facts(self, facts: List[dict]) -> int:
        """Upsert a batch of facts. Returns number of facts processed."""
        count = 0
        for f in facts:
            key = f.get("key")
            if not key:
                continue
            self.upsert_fact(
                key=key,
                value=f.get("value"),
                confidence=float(f.get("confidence", 1.0)),
                source=f.get("source"),
            )
            count += 1
        return count
    
    def get_fact(self, key: str) -> Optional[Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT value_json FROM facts WHERE key=?", (key,)).fetchone()
            return json.loads(row["value_json"]) if row else None
    def list_facts(self) -> Dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value_json FROM facts ORDER BY key").fetchall()
            return {r["key"]: json.loads(r["value_json"]) for r in rows}
    def log_message (
        self,
        role: str,
        content: str,
        tool_name: Optional[str] = None,
        tool_args: Optional[dict] = None,
        tool_result: Optional[dict] = None 
    ) -> None:
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO messages(role, content, tool_name, tool_args_json, tool_result_json, created_at)
                         VALUES (?,?,?,?,?,?)
                         """, (
                             role,
                             content,
                             tool_name,
                             json.dumps(tool_args, ensure_ascii=False) if tool_args is not None else None,
                             json.dumps(tool_result, ensure_ascii=False) if tool_result is not None else None,
                             now_iso()
                         ))
            
    def recent_messages(self, limit: int = 30) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT role, content, tool_name, tool_args_json, tool_result_json, created_at
                FROM messages
                ORDER BY id DESC
                LIMIT ? 
            """, (limit, )).fetchall()

        out = []
        for r in reversed(rows):
            out.append({
                "role": r["role"],
                "content": r["content"],
                "tool_name": r["tool_name"],
                "tool_args": json.loads(r["tool_args_json"]) if r["tool_args_json"] else None,
                "tool_result": json.loads (r["tool_result_json"]) if r["tool_result_json"] else None,
                "created_at": r["created_at"],
            })
        return out