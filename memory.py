import sqlite3
import uuid
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
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
                );
            """)
            self._migrate_conversations(conn)

    LEGACY_CONVERSATION = "legacy"

    def _migrate_conversations(self, conn: sqlite3.Connection) -> None:
        # Messages predate conversations: add the column once, then file every
        # unassigned message under one "Earlier conversations" thread.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(messages)")}
        if "conversation_id" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN conversation_id TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id)")
        span = conn.execute(
            "SELECT MIN(created_at) AS first, MAX(created_at) AS last FROM messages WHERE conversation_id IS NULL"
        ).fetchone()
        if span["first"]:
            conn.execute(
                "INSERT OR IGNORE INTO conversations(id, title, created_at, updated_at) VALUES (?,?,?,?)",
                (self.LEGACY_CONVERSATION, "Earlier conversations", span["first"], span["last"]),
            )
            conn.execute(
                "UPDATE messages SET conversation_id = ? WHERE conversation_id IS NULL",
                (self.LEGACY_CONVERSATION,),
            )

    # --- conversations -------------------------------------------------------

    def create_conversation(self, title: str) -> Dict[str, str]:
        now = now_iso()
        conv = {"id": uuid.uuid4().hex, "title": title, "created_at": now, "updated_at": now}
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations(id, title, created_at, updated_at) VALUES (:id,:title,:created_at,:updated_at)",
                conv,
            )
        return conv

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, str]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        return dict(row) if row else None

    def list_conversations(self, limit: int = 200) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def rename_conversation(self, conversation_id: str, title: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))
        return cur.rowcount > 0

    def delete_conversation(self, conversation_id: str) -> bool:
        # Facts are global and stay; only this conversation's messages go.
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        return cur.rowcount > 0
            
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
        tool_result: Optional[dict] = None,
        conversation_id: Optional[str] = None,
    ) -> None:
        now = now_iso()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO messages(role, content, tool_name, tool_args_json, tool_result_json, created_at, conversation_id)
                         VALUES (?,?,?,?,?,?,?)
                         """, (
                             role,
                             content,
                             tool_name,
                             json.dumps(tool_args, ensure_ascii=False) if tool_args is not None else None,
                             json.dumps(tool_result, ensure_ascii=False) if tool_result is not None else None,
                             now,
                             conversation_id,
                         ))
            if conversation_id:
                # Most recently used conversations sort to the top of the sidebar.
                conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
            
    def recent_messages(self, limit: int = 30, conversation_id: Optional[str] = None) -> List[dict]:
        where, args = ("WHERE conversation_id = ?", (conversation_id, limit)) if conversation_id else ("", (limit,))
        with self._connect() as conn:
            rows = conn.execute(f"""
                SELECT role, content, tool_name, tool_args_json, tool_result_json, created_at
                FROM messages
                {where}
                ORDER BY id DESC
                LIMIT ?
            """, args).fetchall()

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

    def get_memory_context(self, history_limit: int = 20, include_tools: bool = False) -> Dict[str, Any]:
        facts = self.list_facts()
        msgs = self.recent_messages(history_limit)

        # Compact recent message history for prompt injection
        cleaned_msgs: List[dict] = []
        for m in msgs:
            entry = {"role": m["role"], "content": m["content"]}
            if include_tools:
                entry["tool_name"] = m["tool_name"]
                # Keep tool output only if you really want it; can be noisy.
                entry["tool_result"] = m["tool_result"]
            cleaned_msgs.append(entry)

        # Clean facts: do not return None/empty-string values
        cleaned_facts: Dict[str, Any] = {}
        for key, value in facts.items():
            if value is None or value == "":
                continue
            cleaned_facts[key] = value

        return {"facts": cleaned_facts, "recent_messages": cleaned_msgs}
