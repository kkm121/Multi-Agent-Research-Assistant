from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re

import aiosqlite

_DDL = 'CREATE TABLE IF NOT EXISTS auth_credentials ("identifier" TEXT PRIMARY KEY, "password_hash" TEXT NOT NULL)'
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.@-]{3,64}$")


def _db_path() -> str:
    url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///chainlit.db")
    return url.split(":///", 1)[1] if ":///" in url else "chainlit.db"


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(digest).decode()


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_b64, digest_b64 = stored.split("$")
        digest = hashlib.scrypt(
            password.encode(), salt=base64.b64decode(salt_b64), n=2**14, r=8, p=1, dklen=32
        )
        return hmac.compare_digest(digest, base64.b64decode(digest_b64))
    except Exception:
        return False


async def get_hash(identifier: str) -> str | None:
    async with aiosqlite.connect(_db_path()) as db:
        await db.execute(_DDL)
        async with db.execute(
            'SELECT "password_hash" FROM auth_credentials WHERE "identifier" = ?', (identifier,)
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else None


async def set_hash(identifier: str, password: str) -> None:
    async with aiosqlite.connect(_db_path()) as db:
        await db.execute(_DDL)
        await db.execute(
            'INSERT INTO auth_credentials ("identifier", "password_hash") VALUES (?, ?) '
            'ON CONFLICT("identifier") DO UPDATE SET "password_hash" = excluded."password_hash"',
            (identifier, hash_password(password)),
        )
        await db.commit()


_USAGE_DDL = (
    'CREATE TABLE IF NOT EXISTS usage_daily ("identifier" TEXT NOT NULL, "day" TEXT NOT NULL, '
    '"units" INTEGER NOT NULL, PRIMARY KEY ("identifier", "day"))'
)


async def used_today(identifier: str, day: str) -> int:
    async with aiosqlite.connect(_db_path()) as db:
        await db.execute(_USAGE_DDL)
        async with db.execute(
            'SELECT "units" FROM usage_daily WHERE "identifier" = ? AND "day" = ?', (identifier, day)
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else 0


async def add_usage(identifier: str, day: str, units: int) -> None:
    async with aiosqlite.connect(_db_path()) as db:
        await db.execute(_USAGE_DDL)
        await db.execute(
            'INSERT INTO usage_daily ("identifier", "day", "units") VALUES (?, ?, ?) '
            'ON CONFLICT("identifier", "day") DO UPDATE SET "units" = "units" + excluded."units"',
            (identifier, day, units),
        )
        await db.commit()
