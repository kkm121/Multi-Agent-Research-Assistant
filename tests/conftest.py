import os

import pytest

from ai_core.config import Settings

_KEYS = [f.upper() for f in Settings.model_fields if f.endswith(("_api_key", "_models")) or f in ("provider_order", "auth_users", "api_key", "allow_signup")]


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)
    yield
