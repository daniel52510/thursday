"""Per-user credentials (Apple Calendar today), encrypted at rest.

AES-256-GCM with a key that never touches the repo, the image or the database:
a file on the Mac (~/.config/thursday/master.key) mounted read-only into the
container. Each secret is bound to its owner and name (GCM associated data), so
a row copied to another user or name fails to decrypt. Values are write-only
from the API: the UI can set, replace or remove a credential, never read it back.

Without the key file the vault is off and credential features say so.
"""
from __future__ import annotations

import base64
import json
import os
import sqlite3
from datetime import datetime, timezone

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEY_FILE = os.getenv("THURSDAY_MASTER_KEY_FILE", "/run/secrets/thursday_master_key")


def _load_key() -> bytes | None:
    try:
        key = base64.b64decode(open(KEY_FILE).read().strip())
    except (OSError, ValueError):
        return None
    return key if len(key) == 32 else None


class Vault:
    def __init__(self, db_path: str = "./brain/thursday_memory.db") -> None:
        self.db_path = db_path
        key = _load_key()
        self._aes = AESGCM(key) if key else None
        with self._connect() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS secrets (
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                nonce BLOB NOT NULL,
                ciphertext BLOB NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, name)
                )""")

    @property
    def available(self) -> bool:
        return self._aes is not None

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def set(self, user_id: str, name: str, value: dict) -> None:
        if not self._aes:
            raise RuntimeError("vault_unavailable")
        nonce = os.urandom(12)
        ct = self._aes.encrypt(nonce, json.dumps(value).encode(), f"{user_id}:{name}".encode())
        with self._connect() as conn:
            conn.execute("INSERT INTO secrets VALUES (?,?,?,?,?) ON CONFLICT(user_id, name) DO UPDATE SET "
                         "nonce = excluded.nonce, ciphertext = excluded.ciphertext, updated_at = excluded.updated_at",
                         (user_id, name, nonce, ct, datetime.now(timezone.utc).isoformat()))

    def get(self, user_id: str, name: str) -> dict | None:
        if not self._aes:
            return None
        with self._connect() as conn:
            row = conn.execute("SELECT nonce, ciphertext FROM secrets WHERE user_id = ? AND name = ?",
                               (user_id, name)).fetchone()
        if not row:
            return None
        return json.loads(self._aes.decrypt(row["nonce"], row["ciphertext"], f"{user_id}:{name}".encode()))

    def delete(self, user_id: str, name: str) -> bool:
        with self._connect() as conn:
            return conn.execute("DELETE FROM secrets WHERE user_id = ? AND name = ?", (user_id, name)).rowcount > 0

    def names(self, user_id: str) -> dict[str, str]:
        """Which credentials a user has saved, and when; never the values."""
        with self._connect() as conn:
            rows = conn.execute("SELECT name, updated_at FROM secrets WHERE user_id = ?", (user_id,)).fetchall()
        return {r["name"]: r["updated_at"] for r in rows}


_shared: Vault | None = None


def shared() -> Vault:
    """The process-wide Vault (the key file is read once)."""
    global _shared
    if _shared is None:
        _shared = Vault()
    return _shared
