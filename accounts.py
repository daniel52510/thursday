"""Who is talking to THURSDAY: users, logins, devices, personas and token limits.

Roles: admin (Daniel), family, friend, guest. Family and admin share the family
memory; guests get a daily token limit. Everything from before accounts belongs
to the admin, whose id is "owner" (the id memories were already filed under).

Passwords are scrypt hashes. A login gets a random session token in an HttpOnly
cookie; only its SHA-256 is stored, so a copy of the database can't log anyone in.
Satellites and scripts use per-device tokens (stored the same way) that act as
one user and can be revoked one by one.

Set the admin password from the host:
    docker exec -it thursday-core python accounts.py passwd daniel
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROLES = ("admin", "family", "friend", "guest")
FAMILY_ROLES = {"admin", "family"}  # share the family memory
OWNER = "owner"                      # the admin's user id
FAMILY = "family"                    # owner id of the shared family facts
OWNER_USERNAME = os.getenv("THURSDAY_OWNER_USERNAME", "daniel")
OWNER_NAME = os.getenv("THURSDAY_OWNER_NAME", "Daniel")
GUEST_DAILY_TOKENS = int(os.getenv("GUEST_DAILY_TOKENS", "20000"))
SESSION_DAYS = 30
COOKIE = "thursday_session"
USERNAME = re.compile(r"^[a-z0-9][a-z0-9._-]{1,31}$")
MIN_PASSWORD = 8
# scrypt at N=2^15: ~50 ms per login on the Mac, and 32 MB per guess for an attacker.
_SCRYPT = {"n": 2 ** 15, "r": 8, "p": 1}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    key = hashlib.scrypt(password.encode(), salt=salt, maxmem=64 * 2 ** 20, dklen=32, **_SCRYPT)
    b64 = lambda b: base64.b64encode(b).decode()
    return f"scrypt${_SCRYPT['n']}${_SCRYPT['r']}${_SCRYPT['p']}${b64(salt)}${b64(key)}"


def check_password(password: str, stored: str | None) -> bool:
    try:
        _, n, r, p, salt, key = (stored or "").split("$")
        got = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p),
                             maxmem=64 * 2 ** 20, dklen=32)
        return hmac.compare_digest(got, base64.b64decode(key))
    except ValueError:
        return False


def public(user: dict) -> dict:
    """A user as the API shows it: never the password hash."""
    return {k: user[k] for k in ("id", "username", "display_name", "role", "persona_id",
                                 "daily_token_limit", "disabled", "created_at") if k in user}


class Accounts:
    def __init__(self, db_path: str = "./brain/thursday_memory.db") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                display_name TEXT NOT NULL,
                role TEXT NOT NULL,
                password_hash TEXT,              -- NULL: can't log in yet
                persona_id TEXT,                 -- NULL: THURSDAY as she is
                daily_token_limit INTEGER,       -- NULL: no limit
                disabled INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
                );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id),
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
                );
            CREATE TABLE IF NOT EXISTS devices (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                user_id TEXT NOT NULL REFERENCES users(id),
                created_at TEXT NOT NULL,
                last_seen_at TEXT,
                revoked INTEGER NOT NULL DEFAULT 0
                );
            CREATE TABLE IF NOT EXISTS token_usage (
                user_id TEXT NOT NULL,
                day TEXT NOT NULL,               -- UTC date
                tokens INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, day)
                );
            CREATE TABLE IF NOT EXISTS personas (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                prompt TEXT NOT NULL,            -- replaces her personality; tool rules always stay
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
                );
            """)
            now = _now()
            conn.execute("INSERT OR IGNORE INTO users(id, username, display_name, role, created_at, updated_at) "
                         "VALUES (?,?,?,?,?,?)", (OWNER, OWNER_USERNAME, OWNER_NAME, "admin", now, now))

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    # --- users -----------------------------------------------------------------

    def get_user(self, user_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None

    def list_users(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM users ORDER BY role = 'admin' DESC, display_name").fetchall()
        return [dict(r) for r in rows]

    def create_user(self, username: str, display_name: str, role: str, password: str,
                    daily_token_limit: int | None = None) -> dict:
        username = username.strip().lower()
        if not USERNAME.match(username):
            raise ValueError("username: 2-32 lowercase letters, digits, . _ -")
        if role not in ROLES or role == "admin":
            raise ValueError("role must be family, friend or guest")
        if len(password) < MIN_PASSWORD:
            raise ValueError(f"password must be at least {MIN_PASSWORD} characters")
        if role == "guest" and daily_token_limit is None:
            daily_token_limit = GUEST_DAILY_TOKENS
        now, user_id = _now(), uuid.uuid4().hex
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO users(id, username, display_name, role, password_hash, daily_token_limit, "
                    "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                    (user_id, username, display_name.strip()[:60] or username, role, hash_password(password),
                     daily_token_limit, now, now))
        except sqlite3.IntegrityError:
            raise ValueError("that username is taken")
        return self.get_user(user_id)

    def update_user(self, user_id: str, **fields) -> dict | None:
        """Any of display_name, role, persona_id, daily_token_limit, disabled, password."""
        if user_id == OWNER and (fields.get("role", "admin") != "admin" or fields.get("disabled")):
            raise ValueError("the admin can't be demoted or disabled")
        if "role" in fields and fields["role"] not in ROLES:
            raise ValueError("unknown role")
        if fields.get("role") == "admin" and user_id != OWNER:
            raise ValueError("there is one admin")
        updates = {k: v for k, v in fields.items()
                   if k in {"display_name", "role", "persona_id", "daily_token_limit", "disabled"}}
        if "password" in fields:
            if len(fields["password"]) < MIN_PASSWORD:
                raise ValueError(f"password must be at least {MIN_PASSWORD} characters")
            updates["password_hash"] = hash_password(fields["password"])
        if "disabled" in updates:
            updates["disabled"] = int(bool(updates["disabled"]))
        if updates:
            updates["updated_at"] = _now()
            sets = ", ".join(f"{k} = :{k}" for k in updates)
            with self._connect() as conn:
                conn.execute(f"UPDATE users SET {sets} WHERE id = :id", {**updates, "id": user_id})  # nosec B608: column names from the allowlist above, values bound
                if "password_hash" in updates or updates.get("disabled"):
                    conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))  # log out everywhere
                if updates.get("disabled"):
                    conn.execute("UPDATE devices SET revoked = 1 WHERE user_id = ?", (user_id,))
        return self.get_user(user_id)

    # --- logins ----------------------------------------------------------------

    def login(self, username: str, password: str) -> tuple[dict, str] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ? AND disabled = 0",
                               (username.strip().lower(),)).fetchone()
        if not row or not check_password(password, row["password_hash"]):
            if not row:
                check_password(password, hash_password("timing"))  # same cost whether or not the user exists
            return None
        token = secrets.token_urlsafe(32)
        expires = datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS)
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE expires_at < ?", (_now(),))
            conn.execute("INSERT INTO sessions VALUES (?,?,?,?)", (_digest(token), row["id"], _now(), expires.isoformat()))
        return dict(row), token

    def session_user(self, token: str | None) -> dict | None:
        if not token:
            return None
        with self._connect() as conn:
            row = conn.execute("""
                SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id
                WHERE s.token_hash = ? AND s.expires_at > ? AND u.disabled = 0
            """, (_digest(token), _now())).fetchone()
        return dict(row) if row else None

    def logout(self, token: str | None) -> None:
        if token:
            with self._connect() as conn:
                conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_digest(token),))

    # --- devices -----------------------------------------------------------------

    def create_device(self, name: str, user_id: str = OWNER, token: str | None = None) -> tuple[dict, str]:
        """Returns the device and its token; the token is shown once and never stored."""
        token = token or secrets.token_urlsafe(32)
        device = {"id": uuid.uuid4().hex, "name": name.strip()[:60] or "Device", "user_id": user_id, "created_at": _now()}
        with self._connect() as conn:
            conn.execute("INSERT INTO devices(id, name, token_hash, user_id, created_at) VALUES (?,?,?,?,?)",
                         (device["id"], device["name"], _digest(token), user_id, device["created_at"]))
        return device, token

    def adopt_legacy_token(self, token: str) -> None:
        """The old shared VOICE_TOKEN becomes the satellite's device token (acting as the
        admin), so the Pi keeps working; revoke it once the Pi has its own token."""
        if not token:
            return
        with self._connect() as conn:
            known = conn.execute("SELECT 1 FROM devices WHERE token_hash = ?", (_digest(token),)).fetchone()
        if not known:
            self.create_device("Satellite (old VOICE_TOKEN)", OWNER, token)

    def device_user(self, token: str | None) -> dict | None:
        if not token:
            return None
        with self._connect() as conn:
            row = conn.execute("""
                SELECT u.*, d.id AS device_id FROM devices d JOIN users u ON u.id = d.user_id
                WHERE d.token_hash = ? AND d.revoked = 0 AND u.disabled = 0
            """, (_digest(token),)).fetchone()
            if row:
                conn.execute("UPDATE devices SET last_seen_at = ? WHERE id = ?", (_now(), row["device_id"]))
        return dict(row) if row else None

    def list_devices(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT id, name, user_id, created_at, last_seen_at FROM devices "
                                "WHERE revoked = 0 ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

    def revoke_device(self, device_id: str) -> bool:
        with self._connect() as conn:
            return conn.execute("UPDATE devices SET revoked = 1 WHERE id = ? AND revoked = 0", (device_id,)).rowcount > 0

    # --- token limits ------------------------------------------------------------

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def tokens_today(self, user_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT tokens FROM token_usage WHERE user_id = ? AND day = ?",
                               (user_id, self._today())).fetchone()
        return row["tokens"] if row else 0

    def add_tokens(self, user_id: str, tokens: int) -> None:
        if tokens > 0:
            with self._connect() as conn:
                conn.execute("INSERT INTO token_usage VALUES (?,?,?) ON CONFLICT(user_id, day) "
                             "DO UPDATE SET tokens = tokens + excluded.tokens", (user_id, self._today(), tokens))

    def over_limit(self, user: dict) -> bool:
        limit = user.get("daily_token_limit")
        return limit is not None and self.tokens_today(user["id"]) >= limit

    # --- personas ----------------------------------------------------------------

    def list_personas(self) -> list[dict]:
        with self._connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM personas ORDER BY name").fetchall()]

    def get_persona(self, persona_id: str | None) -> dict | None:
        if not persona_id:
            return None
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM personas WHERE id = ?", (persona_id,)).fetchone()
        return dict(row) if row else None

    def save_persona(self, name: str, prompt: str, persona_id: str | None = None) -> dict:
        name, prompt = " ".join(name.split())[:40], prompt.strip()[:8000]
        if not name or not prompt:
            raise ValueError("a persona needs a name and a prompt")
        now, persona_id = _now(), persona_id or uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute("INSERT INTO personas VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                         "name = excluded.name, prompt = excluded.prompt, updated_at = excluded.updated_at",
                         (persona_id, name, prompt, now, now))
        return self.get_persona(persona_id)

    def delete_persona(self, persona_id: str) -> bool:
        with self._connect() as conn:
            conn.execute("UPDATE users SET persona_id = NULL WHERE persona_id = ?", (persona_id,))
            return conn.execute("DELETE FROM personas WHERE id = ?", (persona_id,)).rowcount > 0


_shared: Accounts | None = None


def shared() -> Accounts:
    """The process-wide Accounts (its tables are created once)."""
    global _shared
    if _shared is None:
        _shared = Accounts()
    return _shared


class LoginThrottle:
    """Five wrong passwords for a username (or from an address) lock it for a minute."""

    def __init__(self, limit: int = 5, lock_s: float = 60) -> None:
        self.limit, self.lock_s = limit, lock_s
        self._fails: dict[str, list[float]] = {}

    def blocked(self, *keys: str) -> bool:
        now = time.monotonic()
        return any(len([t for t in self._fails.get(k, []) if now - t < self.lock_s]) >= self.limit for k in keys)

    def fail(self, *keys: str) -> None:
        now = time.monotonic()
        for k in keys:
            self._fails[k] = [t for t in self._fails.get(k, []) if now - t < self.lock_s] + [now]

    def clear(self, *keys: str) -> None:
        for k in keys:
            self._fails.pop(k, None)


def _cli() -> None:
    import getpass
    if len(sys.argv) != 3 or sys.argv[1] != "passwd":
        sys.exit("usage: python accounts.py passwd <username>")
    accounts = Accounts(os.path.join(os.getenv("THURSDAY_BRAIN_DIR", "./brain"), "thursday_memory.db"))
    user = next((u for u in accounts.list_users() if u["username"] == sys.argv[2].lower()), None)
    if not user:
        sys.exit(f"no user {sys.argv[2]!r}")
    password = getpass.getpass(f"New password for {user['username']}: ")
    if password != getpass.getpass("Again: "):
        sys.exit("passwords don't match")
    accounts.update_user(user["id"], password=password)
    print(f"password set for {user['username']} ({user['role']})")


if __name__ == "__main__":
    _cli()
