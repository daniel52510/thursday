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
            -- His settings with each user (honesty, humor, discretion), in percent (mind.py).
            CREATE TABLE IF NOT EXISTS settings (
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                value INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, name)
                );
            -- His own positions, formed in nightly reflection on what he and that user discussed.
            CREATE TABLE IF NOT EXISTS stances (
                user_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                position TEXT NOT NULL,
                reasoning TEXT NOT NULL DEFAULT '',
                confidence INTEGER NOT NULL DEFAULT 3,  -- 1-5
                changed_from TEXT NOT NULL DEFAULT '',  -- the position he gave up, if he changed his mind
                formed_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, topic)
                );
            -- One nightly mission-log entry per user and local day; followups are what he
            -- means to raise when they next talk ({"text", "offered"}).
            CREATE TABLE IF NOT EXISTS mission_log (
                user_id TEXT NOT NULL,
                day TEXT NOT NULL,
                entry TEXT NOT NULL,
                followups_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                PRIMARY KEY (user_id, day)
                );
            -- Missions: background tasks he works on alone (missions.py). status: queued, running,
            -- done, failed, aborted. steps_json is the audit log of every tool call.
            CREATE TABLE IF NOT EXISTS missions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                goal TEXT NOT NULL,
                timezone TEXT,
                status TEXT NOT NULL,
                steps_json TEXT NOT NULL DEFAULT '[]',
                summary TEXT,
                result TEXT,
                error TEXT,
                conversation_id TEXT,
                announced_at TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT
                );
            -- Lists (grocery, packing, to-do) and notes he keeps for each user (notes.py).
            CREATE TABLE IF NOT EXISTS list_items (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                list_name TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL
                );
            CREATE INDEX IF NOT EXISTS list_items_user ON list_items(user_id, list_name);
            -- Approvals: actions a mission proposed that only happen with his OK (approvals.py).
            -- status: pending, approved, declined, expired, failed.
            CREATE TABLE IF NOT EXISTS approvals (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                mission_id TEXT,
                kind TEXT NOT NULL,
                args_json TEXT NOT NULL,
                summary TEXT NOT NULL,
                why TEXT,
                status TEXT NOT NULL,
                result TEXT,
                created_at TEXT NOT NULL,
                decided_at TEXT
                );
            -- Watches: missions on a schedule that only report a change (missions.py). status:
            -- active, done (what he asked about happened), stopped, expired.
            CREATE TABLE IF NOT EXISTS watches (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                what TEXT NOT NULL,
                notify_when TEXT NOT NULL,
                every_hours INTEGER NOT NULL,
                timezone TEXT,
                status TEXT NOT NULL,
                state TEXT,
                runs INTEGER NOT NULL DEFAULT 0,
                next_run_at TEXT NOT NULL,
                last_run_at TEXT,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL
                );
            -- Routines: a mission on a schedule that reports every time (missions.py), e.g. "every
            -- Sunday at 6 pm, my week ahead". days_json: Mon=0..Sun=6, [] = every day. status: active, stopped.
            CREATE TABLE IF NOT EXISTS routines (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                goal TEXT NOT NULL,
                time TEXT NOT NULL,
                days_json TEXT NOT NULL DEFAULT '[]',
                timezone TEXT NOT NULL,
                status TEXT NOT NULL,
                runs INTEGER NOT NULL DEFAULT 0,
                next_run_at TEXT NOT NULL,
                last_run_at TEXT,
                created_at TEXT NOT NULL
                );
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (user_id, title)
                );
            """)
            self._migrate_conversations(conn)
            # The recall note is chosen on a conversation's first turn and then frozen,
            # so her prompt (and Ollama's prompt cache) stays identical for every later turn.
            conv_cols = {r["name"] for r in conn.execute("PRAGMA table_info(conversations)")}
            if "memory_note" not in conv_cols:
                conn.execute("ALTER TABLE conversations ADD COLUMN memory_note TEXT")
            if "pinned" not in conv_cols:  # pinned to the top of the sidebar (2026-11-06)
                conn.execute("ALTER TABLE conversations ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")
            self._migrate_users(conn)
            if "watch_id" not in {r["name"] for r in conn.execute("PRAGMA table_info(missions)")}:
                conn.execute("ALTER TABLE missions ADD COLUMN watch_id TEXT")
            # A follow-up ("dig deeper on the second one") starts from the mission it builds on.
            if "parent_id" not in {r["name"] for r in conn.execute("PRAGMA table_info(missions)")}:
                conn.execute("ALTER TABLE missions ADD COLUMN parent_id TEXT")

    LEGACY_CONVERSATION = "legacy"
    # Made when voice mode opens, before he says anything (voice_ws._warm), so the first turn's prompt is already in
    # Ollama's cache; it takes its real title from his first words. Hidden while empty.
    PREMADE_TITLE = "\u2063voice"

    def _migrate_users(self, conn: sqlite3.Connection) -> None:
        # Accounts: everything from before them belongs to the admin ("owner"). Facts get
        # an owner too: a user id, or "family" for the memory family members share.
        for table in ("conversations", "alarms"):
            if "user_id" not in {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN user_id TEXT NOT NULL DEFAULT 'owner'")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, updated_at)")
        # "briefing": chime + spoken briefing (alarms from the web UI); "wake": she wakes you
        # herself, softly (alarms she set in conversation). brief holds the topics or, for
        # "wake", why you wanted to be up.
        if "style" not in {r["name"] for r in conn.execute("PRAGMA table_info(alarms)")}:
            conn.execute("ALTER TABLE alarms ADD COLUMN style TEXT NOT NULL DEFAULT 'briefing'")
        if "user_id" not in {r["name"] for r in conn.execute("PRAGMA table_info(facts)")}:
            # The key alone was the primary key; SQLite can't change that in place.
            conn.executescript("""
                CREATE TABLE facts_v2 (
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    source TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, key)
                    );
                INSERT INTO facts_v2 SELECT 'owner', key, value_json, confidence, source, created_at, updated_at FROM facts;
                DROP TABLE facts;
                ALTER TABLE facts_v2 RENAME TO facts;
            """)

    def _migrate_conversations(self, conn: sqlite3.Connection) -> None:
        # Messages predate conversations: add the column once, then file every
        # unassigned message under one "Earlier conversations" thread.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(messages)")}
        if "conversation_id" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN conversation_id TEXT")
        if "attachments_json" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN attachments_json TEXT")  # photos/documents sent with it
        if "voice_note" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN voice_note TEXT")  # how he said it (voice_ws.delivery)
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

    def create_conversation(self, title: str, user_id: str) -> Dict[str, str]:
        now = now_iso()
        conv = {"id": uuid.uuid4().hex, "title": title, "created_at": now, "updated_at": now, "user_id": user_id}
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations(id, title, created_at, updated_at, user_id) "
                "VALUES (:id,:title,:created_at,:updated_at,:user_id)",
                conv,
            )
        return conv

    def get_conversation(self, conversation_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, str]]:
        """With user_id, only that user's conversation: anyone else's id reads as unknown."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM conversations WHERE id = ? AND user_id = COALESCE(?, user_id)",
                               (conversation_id, user_id)).fetchone()
        return dict(row) if row else None

    def list_conversations(self, user_id: str, limit: int = 200) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at, pinned FROM conversations WHERE user_id = ? "
                "AND NOT (title = ? AND NOT EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = conversations.id)) "
                "ORDER BY pinned DESC, updated_at DESC LIMIT ?",
                (user_id, self.PREMADE_TITLE, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def pin_conversation(self, conversation_id: str, pinned: bool, user_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("UPDATE conversations SET pinned = ? WHERE id = ? AND user_id = ?",
                               (int(pinned), conversation_id, user_id))
        return cur.rowcount > 0

    def search_conversations(self, user_id: str, query: str, limit: int = 30) -> List[dict]:
        """His conversations whose title or messages contain `query` (case-insensitive), newest first, each with
        the first matching line as a snippet."""
        like = "%" + query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT c.id, c.title, c.updated_at, c.pinned,
                       (SELECT m.content FROM messages m WHERE m.conversation_id = c.id AND m.tool_name IS NULL
                          AND m.content LIKE ? ESCAPE '\\' ORDER BY m.id LIMIT 1) AS snippet
                FROM conversations c WHERE c.user_id = ? AND (c.title LIKE ? ESCAPE '\\' OR EXISTS (
                    SELECT 1 FROM messages m WHERE m.conversation_id = c.id AND m.tool_name IS NULL
                      AND m.content LIKE ? ESCAPE '\\'))
                ORDER BY c.updated_at DESC LIMIT ?""", (like, user_id, like, like, limit)).fetchall()
        return [dict(r) for r in rows]

    def rename_conversation(self, conversation_id: str, title: str, user_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("UPDATE conversations SET title = ? WHERE id = ? AND user_id = ?",
                               (title, conversation_id, user_id))
        return cur.rowcount > 0

    def drop_premade(self, conversation_id: str, user_id: str) -> bool:
        """A pre-made voice conversation nobody spoke in: gone."""
        conv = self.get_conversation(conversation_id, user_id)
        if not conv or conv["title"] != self.PREMADE_TITLE:
            return False
        with self._connect() as conn:
            if conn.execute("SELECT 1 FROM messages WHERE conversation_id = ? LIMIT 1", (conversation_id,)).fetchone():
                return False
        return self.delete_conversation(conversation_id, user_id)

    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        # Facts stay; this conversation's messages and her memory of it go.
        if not self.get_conversation(conversation_id, user_id):
            return False
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM memories WHERE conversation_id = ?", (conversation_id,))
            cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        return cur.rowcount > 0

    def truncate_conversation(self, conversation_id: str, user_id: str, from_user_turn: int) -> int | None:
        """Regenerate / edit-and-resend: delete his from_user_turn-th message (0-based, his own messages only) and
        everything after it, tool calls included. None if there's no such conversation or turn."""
        if not self.get_conversation(conversation_id, user_id) or from_user_turn < 0:
            return None
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM messages WHERE conversation_id = ? AND role = 'user' "
                               "ORDER BY id LIMIT 1 OFFSET ?", (conversation_id, from_user_turn)).fetchone()
            if not row:
                return None
            cur = conn.execute("DELETE FROM messages WHERE conversation_id = ? AND id >= ?", (conversation_id, row[0]))
        return cur.rowcount

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
                SELECT m.conversation_id AS id, MAX(m.id) AS last_id, MAX(m.created_at) AS last_at,
                       (SELECT user_id FROM conversations WHERE id = m.conversation_id) AS user_id
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
            
    # --- his mind: settings, stances, mission log (mind.py) --------------------

    def get_settings(self, user_id: str) -> Dict[str, int]:
        with self._connect() as conn:
            return {r["name"]: r["value"] for r in conn.execute("SELECT name, value FROM settings WHERE user_id = ?", (user_id,))}

    def set_setting(self, user_id: str, name: str, value: int) -> None:
        with self._connect() as conn:
            conn.execute("INSERT INTO settings(user_id, name, value, updated_at) VALUES (?,?,?,?) "
                         "ON CONFLICT(user_id, name) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                         (user_id, name, int(value), now_iso()))

    def list_stances(self, user_id: str, limit: int = 60) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM stances WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
                                (user_id, limit)).fetchall()
        return [dict(r) for r in rows]

    def upsert_stance(self, user_id: str, topic: str, position: str, reasoning: str, confidence: int) -> None:
        """A new position, or a revision; a changed position remembers the one it replaced."""
        now = now_iso()
        with self._connect() as conn:
            old = conn.execute("SELECT position, changed_from FROM stances WHERE user_id = ? AND topic = ?",
                               (user_id, topic)).fetchone()
            changed_from = (old["position"] if old["position"] != position else old["changed_from"]) if old else ""
            conn.execute("""
                INSERT INTO stances(user_id, topic, position, reasoning, confidence, changed_from, formed_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(user_id, topic) DO UPDATE SET position=excluded.position, reasoning=excluded.reasoning,
                    confidence=excluded.confidence, changed_from=excluded.changed_from, updated_at=excluded.updated_at
            """, (user_id, topic, position, reasoning, int(confidence), changed_from, now, now))

    def delete_stance(self, user_id: str, topic: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM stances WHERE user_id = ? AND topic = ?", (user_id, topic))

    def save_mission_log(self, user_id: str, day: str, entry: str, followups: List[str]) -> None:
        items = json.dumps([{"text": t, "offered": False} for t in followups], ensure_ascii=False)
        with self._connect() as conn:
            conn.execute("INSERT INTO mission_log(user_id, day, entry, followups_json, created_at) VALUES (?,?,?,?,?) "
                         "ON CONFLICT(user_id, day) DO UPDATE SET entry=excluded.entry, followups_json=excluded.followups_json",
                         (user_id, day, entry, items, now_iso()))

    def latest_mission_log(self, user_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM mission_log WHERE user_id = ? ORDER BY day DESC LIMIT 1", (user_id,)).fetchone()
        if not row:
            return None
        log = dict(row)
        log["followups"] = json.loads(log.pop("followups_json") or "[]")
        return log

    def recent_assistant_messages(self, user_id: str, exclude_conversation: Optional[str], limit: int = 40) -> List[dict]:
        """His latest replies to this user in other conversations, newest first (mind.recent_lines_note)."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT m.content, m.created_at FROM messages m JOIN conversations c ON c.id = m.conversation_id
                WHERE c.user_id = ? AND m.conversation_id IS NOT ? AND m.role = 'assistant' AND m.tool_name IS NULL
                ORDER BY m.id DESC LIMIT ?
            """, (user_id, exclude_conversation, limit)).fetchall()
        return [dict(r) for r in rows]

    def last_talked_at(self, user_id: str, exclude_conversation: Optional[str]) -> Optional[str]:
        """When this user last said something in another conversation (mind.time_note)."""
        with self._connect() as conn:
            return conn.execute("""
                SELECT MAX(m.created_at) AS t FROM messages m JOIN conversations c ON c.id = m.conversation_id
                WHERE c.user_id = ? AND m.conversation_id IS NOT ? AND m.role = 'user'
            """, (user_id, exclude_conversation)).fetchone()["t"]

    def mark_followups_offered(self, user_id: str, day: str, texts: List[str]) -> None:
        log = self.latest_mission_log(user_id)
        if not log or log["day"] != day:
            return
        items = [{**f, "offered": f["offered"] or f["text"] in texts} for f in log["followups"]]
        with self._connect() as conn:
            conn.execute("UPDATE mission_log SET followups_json = ? WHERE user_id = ? AND day = ?",
                         (json.dumps(items, ensure_ascii=False), user_id, day))

    # --- alarms --------------------------------------------------------------

    # --- missions ---------------------------------------------------------------

    @staticmethod
    def _mission_row(row: sqlite3.Row) -> dict:
        mission = dict(row)
        mission["steps"] = json.loads(mission.pop("steps_json") or "[]")
        return mission

    def create_mission(self, user_id: str, goal: str, timezone: Optional[str], watch_id: Optional[str] = None,
                       parent_id: Optional[str] = None) -> dict:
        mission_id = uuid.uuid4().hex[:8]
        with self._connect() as conn:
            conn.execute("INSERT INTO missions(id, user_id, goal, timezone, status, created_at, watch_id, parent_id) "
                         "VALUES (?,?,?,?,?,?,?,?)",
                         (mission_id, user_id, goal, timezone, "queued", now_iso(), watch_id, parent_id))
        return self.get_mission(mission_id)

    def create_approval(self, user_id: str, mission_id: Optional[str], kind: str, args: dict, summary: str,
                        why: str) -> dict:
        approval_id = uuid.uuid4().hex[:8]
        with self._connect() as conn:
            conn.execute("INSERT INTO approvals(id, user_id, mission_id, kind, args_json, summary, why, status, created_at) "
                         "VALUES (?,?,?,?,?,?,?,?,?)", (approval_id, user_id, mission_id, kind,
                                                        json.dumps(args, ensure_ascii=False), summary, why, "pending", now_iso()))
        return self.get_approval(approval_id)

    @staticmethod
    def _approval_row(row: sqlite3.Row) -> dict:
        approval = dict(row)
        approval["args"] = json.loads(approval.pop("args_json"))
        return approval

    def get_approval(self, approval_id: str, user_id: Optional[str] = None) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM approvals WHERE id = ? AND user_id = COALESCE(?, user_id)",
                               (approval_id, user_id)).fetchone()
        return self._approval_row(row) if row else None

    def list_approvals(self, user_id: str, status: Optional[str] = None, mission_id: Optional[str] = None,
                       limit: int = 30) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM approvals WHERE user_id = ? AND status = COALESCE(?, status) "
                                "AND COALESCE(mission_id, '') = COALESCE(?, COALESCE(mission_id, '')) "
                                "ORDER BY created_at LIMIT ?", (user_id, status, mission_id, limit)).fetchall()
        return [self._approval_row(r) for r in rows]

    def update_approval(self, approval_id: str, **fields: Any) -> None:
        """Any of status, result, decided_at."""
        updates = {k: v for k, v in fields.items() if k in {"status", "result", "decided_at"}}
        if updates:
            assignments = ", ".join(f"{k} = :{k}" for k in updates)
            with self._connect() as conn:
                conn.execute(f"UPDATE approvals SET {assignments} WHERE id = :id", {**updates, "id": approval_id})  # nosec B608: column names from the allowlist above, values bound

    def create_watch(self, user_id: str, what: str, notify_when: str, every_hours: int, timezone: Optional[str],
                     expires_at: str) -> dict:
        watch_id, now = uuid.uuid4().hex[:8], now_iso()
        with self._connect() as conn:
            conn.execute("INSERT INTO watches(id, user_id, what, notify_when, every_hours, timezone, status, next_run_at, "
                         "expires_at, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (watch_id, user_id, what, notify_when, every_hours, timezone, "active", now, expires_at, now))
        return self.get_watch(watch_id)

    def get_watch(self, watch_id: str, user_id: Optional[str] = None) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM watches WHERE id = ? AND user_id = COALESCE(?, user_id)",
                               (watch_id, user_id)).fetchone()
        return dict(row) if row else None

    def list_watches(self, user_id: Optional[str] = None, status: Optional[str] = None, limit: int = 30) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM watches WHERE user_id = COALESCE(?, user_id) AND status = COALESCE(?, status) "
                                "ORDER BY created_at DESC LIMIT ?", (user_id, status, limit)).fetchall()
        return [dict(r) for r in rows]

    def update_watch(self, watch_id: str, **fields: Any) -> None:
        """Any of status, state, runs, next_run_at, last_run_at."""
        updates = {k: v for k, v in fields.items() if k in {"status", "state", "runs", "next_run_at", "last_run_at"}}
        if updates:
            assignments = ", ".join(f"{k} = :{k}" for k in updates)
            with self._connect() as conn:
                conn.execute(f"UPDATE watches SET {assignments} WHERE id = :id", {**updates, "id": watch_id})  # nosec B608: column names from the allowlist above, values bound

    @staticmethod
    def _routine_row(row: sqlite3.Row) -> dict:
        routine = dict(row)
        routine["days"] = json.loads(routine.pop("days_json") or "[]")
        return routine

    def create_routine(self, user_id: str, goal: str, time: str, days: list, timezone: str, next_run_at: str) -> dict:
        routine_id = uuid.uuid4().hex[:8]
        with self._connect() as conn:
            conn.execute("INSERT INTO routines(id, user_id, goal, time, days_json, timezone, status, next_run_at, created_at) "
                         "VALUES (?,?,?,?,?,?,?,?,?)", (routine_id, user_id, goal, time, json.dumps(days), timezone,
                                                        "active", next_run_at, now_iso()))
        return self.get_routine(routine_id)

    def get_routine(self, routine_id: str, user_id: Optional[str] = None) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM routines WHERE id = ? AND user_id = COALESCE(?, user_id)",
                               (routine_id, user_id)).fetchone()
        return self._routine_row(row) if row else None

    def list_routines(self, user_id: Optional[str] = None, status: Optional[str] = None, limit: int = 30) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM routines WHERE user_id = COALESCE(?, user_id) AND status = COALESCE(?, status) "
                                "ORDER BY created_at DESC LIMIT ?", (user_id, status, limit)).fetchall()
        return [self._routine_row(r) for r in rows]

    def update_routine(self, routine_id: str, **fields: Any) -> None:
        """Any of status, runs, next_run_at, last_run_at."""
        updates = {k: v for k, v in fields.items() if k in {"status", "runs", "next_run_at", "last_run_at"}}
        if updates:
            assignments = ", ".join(f"{k} = :{k}" for k in updates)
            with self._connect() as conn:
                conn.execute(f"UPDATE routines SET {assignments} WHERE id = :id", {**updates, "id": routine_id})  # nosec B608: column names from the allowlist above, values bound

    def get_mission(self, mission_id: str, user_id: Optional[str] = None) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM missions WHERE id = ? AND user_id = COALESCE(?, user_id)",
                               (mission_id, user_id)).fetchone()
        return self._mission_row(row) if row else None

    def list_missions(self, user_id: Optional[str] = None, status: Optional[str] = None, limit: int = 20) -> List[dict]:
        """Newest first; all users' (for the worker) without user_id."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM missions WHERE user_id = COALESCE(?, user_id) AND status = COALESCE(?, status) "
                                "ORDER BY created_at DESC LIMIT ?", (user_id, status, limit)).fetchall()
        return [self._mission_row(r) for r in rows]

    def update_mission(self, mission_id: str, **fields: Any) -> None:
        """Any of status, steps, summary, result, error, conversation_id, announced_at, started_at, finished_at."""
        allowed = {"status", "steps", "summary", "result", "error", "conversation_id", "announced_at", "started_at",
                   "finished_at"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if "steps" in updates:
            updates["steps_json"] = json.dumps(updates.pop("steps"), ensure_ascii=False)
        if updates:
            assignments = ", ".join(f"{k} = :{k}" for k in updates)
            with self._connect() as conn:
                conn.execute(f"UPDATE missions SET {assignments} WHERE id = :id", {**updates, "id": mission_id})  # nosec B608: column names from the allowlist above, values bound

    # --- lists and notes -------------------------------------------------------

    def list_items(self, user_id: str, list_name: Optional[str] = None) -> List[dict]:
        """One list's items, oldest first; every list's without list_name."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM list_items WHERE user_id = ? AND list_name = COALESCE(?, list_name) "
                                "ORDER BY created_at", (user_id, list_name)).fetchall()
        return [dict(r) for r in rows]

    def add_list_item(self, user_id: str, list_name: str, text: str) -> None:
        with self._connect() as conn:
            conn.execute("INSERT INTO list_items(id, user_id, list_name, text, created_at) VALUES (?,?,?,?,?)",
                         (uuid.uuid4().hex, user_id, list_name, text, now_iso()))

    def delete_list_items(self, user_id: str, ids: List[str]) -> None:
        with self._connect() as conn:
            conn.executemany("DELETE FROM list_items WHERE id = ? AND user_id = ?", [(i, user_id) for i in ids])

    def save_note(self, user_id: str, title: str, body: str) -> dict:
        """A new note, or more text on the note with that title."""
        now = now_iso()
        with self._connect() as conn:
            conn.execute("INSERT INTO notes(id, user_id, title, body, created_at, updated_at) VALUES (?,?,?,?,?,?) "
                         "ON CONFLICT(user_id, title) DO UPDATE SET body = notes.body || char(10) || excluded.body, "
                         "updated_at = excluded.updated_at", (uuid.uuid4().hex, user_id, title, body, now, now))
            row = conn.execute("SELECT * FROM notes WHERE user_id = ? AND title = ?", (user_id, title)).fetchone()
        return dict(row)

    def list_notes(self, user_id: str, limit: int = 200) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM notes WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
                                (user_id, limit)).fetchall()
        return [dict(r) for r in rows]

    def delete_note(self, user_id: str, title: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM notes WHERE user_id = ? AND title = ?", (user_id, title))
        return cur.rowcount > 0

    @staticmethod
    def _alarm_row(row: sqlite3.Row) -> dict:
        alarm = dict(row)
        alarm["days"] = [int(d) for d in alarm["days"].split(",") if d]
        alarm["enabled"] = bool(alarm["enabled"])
        return alarm

    def list_alarms(self, user_id: Optional[str] = None) -> List[dict]:
        """One user's alarms; all of them (for the scheduler) without user_id."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM alarms WHERE user_id = COALESCE(?, user_id) ORDER BY time",
                                (user_id,)).fetchall()
        return [self._alarm_row(r) for r in rows]

    def get_alarm(self, alarm_id: str, user_id: Optional[str] = None) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM alarms WHERE id = ? AND user_id = COALESCE(?, user_id)",
                               (alarm_id, user_id)).fetchone()
        return self._alarm_row(row) if row else None

    def create_alarm(self, time: str, days: List[int], timezone: str, brief: str, enabled: bool = True,
                     user_id: str = "owner", style: str = "briefing") -> dict:
        now = now_iso()
        alarm_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO alarms(id, time, days, timezone, brief, enabled, created_at, updated_at, user_id, style) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (alarm_id, time, ",".join(map(str, sorted(days))), timezone, brief, int(enabled), now, now, user_id, style),
            )
        return self.get_alarm(alarm_id)

    def update_alarm(self, alarm_id: str, user_id: Optional[str] = None, **fields: Any) -> Optional[dict]:
        """Update any of time, days, timezone, brief, enabled, last_fired_at."""
        if user_id and not self.get_alarm(alarm_id, user_id):
            return None
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
                conn.execute(f"UPDATE alarms SET {assignments} WHERE id = :id", {**updates, "id": alarm_id})  # nosec B608: column names from the allowlist above, values bound
        return self.get_alarm(alarm_id)

    def delete_alarm(self, alarm_id: str, user_id: Optional[str] = None) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM alarms WHERE id = ? AND user_id = COALESCE(?, user_id)", (alarm_id, user_id))
        return cur.rowcount > 0

    # --- facts: per user, or "family" for the memory family members share -------

    def upsert_fact(self, user_id: str, key: str, value: Any, confidence: float = 1.0, source: Optional[str] = None) -> None:
        value_json = json.dumps(value, ensure_ascii = False)
        ts = now_iso()
        with self._connect() as conn:
            conn.execute("""
            INSERT INTO facts(user_id, key, value_json, confidence, source, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(user_id, key) DO UPDATE SET
                value_json = excluded.value_json,
                confidence = excluded.confidence,
                         source = excluded.source,
                         updated_at = excluded.updated_at
            """, (user_id, key, value_json, confidence, source, ts, ts))

    def upsert_facts(self, facts: List[dict], user_id: str) -> int:
        """Upsert a batch of facts. Returns number of facts processed."""
        count = 0
        for f in facts:
            key = f.get("key")
            if not key:
                continue
            self.upsert_fact(
                user_id,
                key=key,
                value=f.get("value"),
                confidence=float(f.get("confidence", 1.0)),
                source=f.get("source"),
            )
            count += 1
        return count

    def get_fact(self, key: str, user_id: str) -> Optional[Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT value_json FROM facts WHERE user_id = ? AND key = ?", (user_id, key)).fetchone()
            return json.loads(row["value_json"]) if row else None

    def list_facts(self, user_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value_json FROM facts WHERE user_id = ? ORDER BY key", (user_id,)).fetchall()
            return {r["key"]: json.loads(r["value_json"]) for r in rows}

    def delete_fact(self, user_id: str, key: str) -> bool:
        with self._connect() as conn:
            return conn.execute("DELETE FROM facts WHERE user_id = ? AND key = ?", (user_id, key)).rowcount > 0
    def log_message (
        self,
        role: str,
        content: str,
        tool_name: Optional[str] = None,
        tool_args: Optional[dict] = None,
        tool_result: Optional[dict] = None,
        conversation_id: Optional[str] = None,
        attachments: Optional[list] = None,
        voice_note: Optional[str] = None,
    ) -> None:
        now = now_iso()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO messages(role, content, tool_name, tool_args_json, tool_result_json, created_at, conversation_id,
                                     attachments_json, voice_note)
                         VALUES (?,?,?,?,?,?,?,?,?)
                         """, (
                             role,
                             content,
                             tool_name,
                             json.dumps(tool_args, ensure_ascii=False) if tool_args is not None else None,
                             json.dumps(tool_result, ensure_ascii=False) if tool_result is not None else None,
                             now,
                             conversation_id,
                             json.dumps(attachments, ensure_ascii=False) if attachments else None,
                             voice_note,
                         ))
            if conversation_id:
                # Most recently used conversations sort to the top of the sidebar.
                conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
            
    def revise_last_reply(self, conversation_id: str, content: str) -> None:
        """Make the reply to the latest user message `content`: replace it if
        it was saved, add it if not, delete it if `content` is empty. Used when
        a spoken reply is cut off, so history holds only what was heard."""
        with self._connect() as conn:
            user = conn.execute("SELECT MAX(id) FROM messages WHERE conversation_id = ? AND role = 'user'",
                                (conversation_id,)).fetchone()[0] or 0
            reply = conn.execute("""
                SELECT MAX(id) FROM messages
                WHERE conversation_id = ? AND role = 'assistant' AND tool_name IS NULL AND id > ?
            """, (conversation_id, user)).fetchone()[0]
            if reply and content:
                conn.execute("UPDATE messages SET content = ? WHERE id = ?", (content, reply))
            elif reply:
                conn.execute("DELETE FROM messages WHERE id = ?", (reply,))
            elif content:
                conn.execute("INSERT INTO messages(role, content, created_at, conversation_id) VALUES ('assistant', ?, ?, ?)",
                             (content, now_iso(), conversation_id))

    def recent_messages(self, limit: int = 30, conversation_id: Optional[str] = None) -> List[dict]:
        where, args = ("WHERE conversation_id = ?", (conversation_id, limit)) if conversation_id else ("", (limit,))  # constant SQL text
        with self._connect() as conn:
            rows = conn.execute(f"""
                SELECT role, content, tool_name, tool_args_json, tool_result_json, created_at, attachments_json, voice_note
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
                "attachments": json.loads(r["attachments_json"]) if r["attachments_json"] else None,
                "voice_note": r["voice_note"],
            })
        return out

    def prompt_facts(self, user_id: str) -> Dict[str, Any]:
        """A user's facts for the prompt, without None/empty-string values."""
        return {k: v for k, v in self.list_facts(user_id).items()
                if v is not None and v != "" and not k.startswith("_")}  # "_" keys: settings, not facts

    # His custom instructions ("how I want you to talk"): free text, so a reserved fact, not the int settings table.
    def instructions(self, user_id: str) -> str:
        return str(self.get_fact("_instructions", user_id) or "")

    def set_instructions(self, user_id: str, text: str) -> None:
        if text:
            self.upsert_fact(user_id, "_instructions", text, source="settings")
        else:
            self.delete_fact(user_id, "_instructions")

    def delete_memory(self, conversation_id: str, user_id: str) -> bool:
        with self._connect() as conn:
            return conn.execute("DELETE FROM memories WHERE conversation_id = ? AND user_id = ?",
                                (conversation_id, user_id)).rowcount > 0
