import os

import pytest

from ai_core.config import Settings

_KEYS = [f.upper() for f in Settings.model_fields if f.endswith(("_api_key", "_models")) or f == "provider_order"]


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    """Tests must never see real keys (a developer .env may have been loaded into os.environ)."""
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)
    yield
