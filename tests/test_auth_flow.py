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
    con = sqlite3.connect(db)
    for ident, meta in (("legacy", {"role": "user", "password": "oldpass1"}), ("orphan", {"role": "user"})):
        con.execute("INSERT INTO users VALUES (?,?,?,?)", (str(uuid.uuid4()), ident, json.dumps(meta), "2026-01-01"))
    con.commit()
    con.close()

    login = app.auth_callback

    async def go():
        assert await login("ab", "longenough") is None
        assert await login("alice", "short") is None
        assert (await login("alice", "secret123")).identifier == "alice"
        assert (await login("alice", "secret123")).identifier == "alice"
        assert await login("alice", "wrong-pass") is None
        assert await login("legacy", "wrongpass") is None
        assert (await login("legacy", "oldpass1")).identifier == "legacy"
        assert (await login("legacy", "oldpass1")) is not None
        assert (await login("orphan", "newpass99")).identifier == "orphan"
        assert await login("orphan", "other-pass") is None

    asyncio.run(go())
    rows = dict(sqlite3.connect(db).execute("SELECT identifier, password_hash FROM auth_credentials").fetchall())
    assert set(rows) == {"alice", "legacy", "orphan"}
    assert all(h.startswith("scrypt$") for h in rows.values())


def test_owner_accounts_block_everyone_else(tmp_path, monkeypatch):
    from frontend.auth import hash_password

    db = tmp_path / "o.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db.as_posix()}")
    monkeypatch.setenv("CHAINLIT_AUTH_SECRET", "x" * 64)
    monkeypatch.setenv("AUTH_USERS", f"owner1:{hash_password('owner-pass-123')}")
    import init_chainlit_db

    init_chainlit_db.main()
    from ai_core import config

    config.get_settings.cache_clear()
    app = importlib.import_module("frontend.app")
    app._data_layer = None

    async def go():
        assert (await app.auth_callback("owner1", "owner-pass-123")).identifier == "owner1"
        assert await app.auth_callback("owner1", "wrong-pass-123") is None
        assert await app.auth_callback("stranger", "any-password-1") is None
        await app.get_data_layer().create_user(app.cl.User(identifier="orphan2", metadata={}))
        assert await app.auth_callback("orphan2", "claim-it-123") is None

    asyncio.run(go())
    config.get_settings.cache_clear()


def test_api_requires_key(monkeypatch):
    from fastapi.testclient import TestClient
    from ai_core import config

    monkeypatch.setenv("CHAINLIT_AUTH_SECRET", "x" * 64)
    backend = importlib.import_module("backend.main")
    client = TestClient(backend.app)
    config.get_settings.cache_clear()
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr(config.Settings, "model_config", {**config.Settings.model_config, "env_file": None})
    config.get_settings.cache_clear()
    assert client.get("/api/health").status_code == 200
    assert client.post("/api/research", json={"topic": "abc test"}).status_code == 403
    monkeypatch.setenv("API_KEY", "k" * 32)
    config.get_settings.cache_clear()
    assert client.get("/api/providers").status_code == 401
    assert client.get("/api/providers", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/providers", headers={"X-API-Key": "k" * 32}).status_code == 200
    config.get_settings.cache_clear()
