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
            CREATE TABLE IF NOT EXISTS alarms (
                id TEXT PRIMARY KEY,
                time TEXT NOT NULL,            -- "HH:MM", local to timezone
                days TEXT NOT NULL DEFAULT '', -- "0,1,2,3,4" (Mon=0); empty = rings once
                timezone TEXT NOT NULL,
                brief TEXT NOT NULL DEFAULT '',
                enabled INTEGER NOT NULL DEFAULT 1,
                last_fired_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
                );
            -- THURSDAY's own memory of each conversation, written once it goes quiet.
            -- user_id is the owner today; accounts later only change what goes in it.
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL DEFAULT 'owner',
                conversation_id TEXT NOT NULL UNIQUE,
                summary TEXT NOT NULL,
                their_mood TEXT NOT NULL DEFAULT '',
                her_feeling TEXT NOT NULL DEFAULT '',
                threads_json TEXT NOT NULL DEFAULT '[]',
                importance INTEGER NOT NULL DEFAULT 2,
                covered_until INTEGER NOT NULL,  -- last message id this memory includes
                happened_at TEXT NOT NULL,       -- time of that message
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
                );
            CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id, happened_at);
            """)
            self._migrate_conversations(conn)
            # The recall note is chosen on a conversation's first turn and then frozen,
            # so her prompt (and Ollama's prompt cache) stays identical for every later turn.
            if "memory_note" not in {r["name"] for r in conn.execute("PRAGMA table_info(conversations)")}:
                conn.execute("ALTER TABLE conversations ADD COLUMN memory_note TEXT")

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
        # Facts are global and stay; this conversation's messages and her memory of it go.
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM memories WHERE conversation_id = ?", (conversation_id,))
            cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        return cur.rowcount > 0

    def set_memory_note(self, conversation_id: str, note: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE conversations SET memory_note = ? WHERE id = ?", (note, conversation_id))

    # --- memories ------------------------------------------------------------

    def last_message_at(self) -> Optional[str]:
        with self._connect() as conn:
            return conn.execute("SELECT MAX(created_at) AS t FROM messages").fetchone()["t"]

    def conversations_to_remember(self, idle_before: str, limit: int = 3) -> List[dict]:
        """Conversations quiet since `idle_before` with turns her memory doesn't cover yet."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT m.conversation_id AS id, MAX(m.id) AS last_id, MAX(m.created_at) AS last_at
                FROM messages m
                WHERE m.tool_name IS NULL AND m.conversation_id IS NOT NULL
                GROUP BY m.conversation_id
                HAVING COUNT(*) >= 2 AND last_at < ?
                   AND last_id > COALESCE((SELECT covered_until FROM memories WHERE conversation_id = m.conversation_id), 0)
                ORDER BY last_at DESC
                LIMIT ?
            """, (idle_before, limit)).fetchall()
        return [dict(r) for r in rows]

    def upsert_memory(self, conversation_id: str, note: dict, covered_until: int, happened_at: str,
                      user_id: str = "owner") -> None:
        now = now_iso()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO memories(id, user_id, conversation_id, summary, their_mood, her_feeling,
                                     threads_json, importance, covered_until, happened_at, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    summary=excluded.summary, their_mood=excluded.their_mood, her_feeling=excluded.her_feeling,
                    threads_json=excluded.threads_json, importance=excluded.importance,
                    covered_until=excluded.covered_until, happened_at=excluded.happened_at, updated_at=excluded.updated_at
            """, (uuid.uuid4().hex, user_id, conversation_id, note["summary"], note.get("their_mood", ""),
                  note.get("her_feeling", ""), json.dumps(note.get("open_threads", []), ensure_ascii=False),
                  note.get("importance", 2), covered_until, happened_at, now, now))

    def recent_memories(self, exclude_conversation: Optional[str], user_id: str = "owner", limit: int = 60) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT conversation_id, summary, their_mood, her_feeling, threads_json, importance, happened_at
                FROM memories WHERE user_id = ? AND conversation_id IS NOT ?
                ORDER BY happened_at DESC LIMIT ?
            """, (user_id, exclude_conversation, limit)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["open_threads"] = json.loads(d.pop("threads_json") or "[]")
            out.append(d)
        return out
            
    # --- alarms --------------------------------------------------------------

    @staticmethod
    def _alarm_row(row: sqlite3.Row) -> dict:
        alarm = dict(row)
        alarm["days"] = [int(d) for d in alarm["days"].split(",") if d]
        alarm["enabled"] = bool(alarm["enabled"])
        return alarm

    def list_alarms(self) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM alarms ORDER BY time").fetchall()
        return [self._alarm_row(r) for r in rows]

    def get_alarm(self, alarm_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM alarms WHERE id = ?", (alarm_id,)).fetchone()
        return self._alarm_row(row) if row else None

    def create_alarm(self, time: str, days: List[int], timezone: str, brief: str, enabled: bool = True) -> dict:
        now = now_iso()
        alarm_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO alarms(id, time, days, timezone, brief, enabled, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (alarm_id, time, ",".join(map(str, sorted(days))), timezone, brief, int(enabled), now, now),
            )
        return self.get_alarm(alarm_id)

    def update_alarm(self, alarm_id: str, **fields: Any) -> Optional[dict]:
        """Update any of time, days, timezone, brief, enabled, last_fired_at."""
        allowed = {"time", "days", "timezone", "brief", "enabled", "last_fired_at"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if "days" in updates:
            updates["days"] = ",".join(map(str, sorted(updates["days"])))
        if "enabled" in updates:
            updates["enabled"] = int(updates["enabled"])
        if updates:
            updates["updated_at"] = now_iso()
            assignments = ", ".join(f"{k} = :{k}" for k in updates)
            with self._connect() as conn:
                conn.execute(f"UPDATE alarms SET {assignments} WHERE id = :id", {**updates, "id": alarm_id})
        return self.get_alarm(alarm_id)

    def delete_alarm(self, alarm_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM alarms WHERE id = ?", (alarm_id,))
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
