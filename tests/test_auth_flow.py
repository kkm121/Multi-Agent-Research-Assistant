"""Login flow against a throwaway SQLite database."""

import asyncio
import importlib
import json
import sqlite3
import uuid


def test_login_flow(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    url = f"sqlite+aiosqlite:///{db.as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("CHAINLIT_AUTH_SECRET", "x" * 64)

    import init_chainlit_db

    init_chainlit_db.main()
    from ai_core import config

    config.get_settings.cache_clear()
    app = importlib.import_module("frontend.app")
    app._data_layer = None
    # Seed a v1 user (plaintext password in metadata) and an orphan row with no credentials.
    con = sqlite3.connect(db)
    for ident, meta in (("legacy", {"role": "user", "password": "oldpass1"}), ("orphan", {"role": "user"})):
        con.execute("INSERT INTO users VALUES (?,?,?,?)", (str(uuid.uuid4()), ident, json.dumps(meta), "2026-01-01"))
    con.commit()
    con.close()

    login = app.auth_callback

    async def go():
        assert await login("ab", "longenough") is None  # bad username
        assert await login("alice", "short") is None  # short password
        assert (await login("alice", "secret123")).identifier == "alice"  # signup
        assert (await login("alice", "secret123")).identifier == "alice"  # login
        assert await login("alice", "wrong-pass") is None
        assert await login("legacy", "wrongpass") is None
        assert (await login("legacy", "oldpass1")).identifier == "legacy"  # migrated
        assert (await login("legacy", "oldpass1")) is not None  # now via hash
        assert (await login("orphan", "newpass99")).identifier == "orphan"  # reclaimed
        assert await login("orphan", "other-pass") is None

    asyncio.run(go())
    rows = dict(sqlite3.connect(db).execute("SELECT identifier, password_hash FROM auth_credentials").fetchall())
    assert set(rows) == {"alice", "legacy", "orphan"}
    assert all(h.startswith("scrypt$") for h in rows.values())
