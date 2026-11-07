from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

PROVIDERS = ("gemini", "groq", "openrouter", "nvidia", "mistral")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    nvidia_api_key: str = ""
    mistral_api_key: str = ""

    gemini_strong_models: str = "gemini-2.5-flash,gemini-3.5-flash"
    gemini_fast_models: str = "gemini-3.5-flash-lite,gemini-flash-lite-latest,gemini-2.5-flash"
    groq_strong_models: str = "openai/gpt-oss-120b,qwen/qwen3.8-27b"
    groq_fast_models: str = "openai/gpt-oss-20b,qwen/qwen3.8-27b,openai/gpt-oss-120b"
    openrouter_strong_models: str = "nvidia/nemotron-3-ultra-550b-a55b:free,openrouter/free"
    openrouter_fast_models: str = "openrouter/free"
    nvidia_strong_models: str = "mistralai/mistral-large-4-1t-a52b"
    nvidia_fast_models: str = "mistralai/mistral-large-4-1t-a52b"
    mistral_strong_models: str = "mistral-medium-latest"
    mistral_fast_models: str = "mistral-small-latest"

    provider_order: str = "gemini,groq,openrouter,nvidia,mistral"
    llm_timeout_s: float = 60.0
    rate_limit_cooldown_s: float = 30.0

    tavily_api_key: str = ""
    jina_api_key: str = ""
    semantic_scholar_api_key: str = ""
    contact_email: str = ""

    chainlit_auth_secret: str = ""
    database_url: str = "sqlite+aiosqlite:///chainlit.db"
    allow_signup: bool = True
    auth_users: str = ""
    api_key: str = ""
    max_revisions: int = 1

    def provider_key(self, provider: str) -> str:
        return getattr(self, f"{provider}_api_key", "")

    def models(self, provider: str, tier: str) -> list[str]:
        raw = getattr(self, f"{provider}_{tier}_models", "")
        return [m.strip() for m in raw.split(",") if m.strip()]

    def owner_accounts(self) -> dict[str, str]:
        out = {}
        for entry in self.auth_users.split(","):
            name, _, hashed = entry.strip().partition(":")
            if name and hashed:
                out[name.strip()] = hashed.strip()
        return out

    def configured_providers(self) -> list[str]:
        order = [p.strip() for p in self.provider_order.split(",") if p.strip() in PROVIDERS]
        return [p for p in order if self.provider_key(p)]


@lru_cache
def get_settings() -> Settings:
    return Settings()
