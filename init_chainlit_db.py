"""Create the SQLite schema used by Chainlit's SQLAlchemyDataLayer (+ the auth table). Idempotent."""

import os
import sqlite3

from dotenv import load_dotenv

load_dotenv()

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    "id" TEXT PRIMARY KEY,
    "identifier" TEXT NOT NULL UNIQUE,
    "metadata" TEXT NOT NULL,
    "createdAt" TEXT
);
CREATE TABLE IF NOT EXISTS threads (
    "id" TEXT PRIMARY KEY,
    "createdAt" TEXT,
    "name" TEXT,
    "userId" TEXT,
    "userIdentifier" TEXT,
    "tags" TEXT,
    "metadata" TEXT,
    FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS steps (
    "id" TEXT PRIMARY KEY,
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "threadId" TEXT NOT NULL,
    "parentId" TEXT,
    "streaming" INTEGER NOT NULL,
    "waitForAnswer" INTEGER,
    "isError" INTEGER,
    "metadata" TEXT,
    "tags" TEXT,
    "input" TEXT,
    "output" TEXT,
    "createdAt" TEXT,
    "command" TEXT,
    "start" TEXT,
    "end" TEXT,
    "generation" TEXT,
    "showInput" TEXT,
    "language" TEXT,
    "indent" INTEGER,
    "defaultOpen" INTEGER,
    "autoCollapse" INTEGER,
    "modes" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS elements (
    "id" TEXT PRIMARY KEY,
    "threadId" TEXT,
    "type" TEXT,
    "url" TEXT,
    "chainlitKey" TEXT,
    "name" TEXT NOT NULL,
    "display" TEXT,
    "objectKey" TEXT,
    "size" TEXT,
    "page" INTEGER,
    "language" TEXT,
    "forId" TEXT,
    "mime" TEXT,
    "props" TEXT,
    "autoPlay" INTEGER,
    "playerConfig" TEXT
);
CREATE TABLE IF NOT EXISTS feedbacks (
    "id" TEXT PRIMARY KEY,
    "forId" TEXT NOT NULL,
    "threadId" TEXT,
    "value" INTEGER NOT NULL,
    "comment" TEXT
);
CREATE TABLE IF NOT EXISTS auth_credentials (
    "identifier" TEXT PRIMARY KEY,
    "password_hash" TEXT NOT NULL
);
"""

# Columns added in newer Chainlit versions — add them to databases created by v1.
MIGRATIONS = [
    ("steps", "autoCollapse", "INTEGER"),
    ("steps", "modes", "TEXT"),
    ("elements", "props", "TEXT"),
    ("elements", "autoPlay", "INTEGER"),
    ("elements", "playerConfig", "TEXT"),
    ("feedbacks", "threadId", "TEXT"),
]


def db_path() -> str:
    url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///chainlit.db")
    return url.split(":///", 1)[1] if ":///" in url else "chainlit.db"


def main() -> None:
    conn = sqlite3.connect(db_path())
    conn.executescript(SCHEMA)
    for table, col, typ in MIGRATIONS:
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if col not in cols:
            conn.execute(f'ALTER TABLE {table} ADD COLUMN "{col}" {typ}')
    conn.commit()
    conn.close()
    print(f"SQLite schema ready at {db_path()}")


if __name__ == "__main__":
    main()
