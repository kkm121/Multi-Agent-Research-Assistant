"""Async multi-provider LLM router.

Every provider used here exposes an OpenAI-compatible endpoint, so one client type covers
all of them. The router:
  * tries providers in `PROVIDER_ORDER`, skipping any without an API key;
  * per provider, tries an ordered list of models and permanently skips a model that the provider
    reports as gone (free-tier models get retired often — e.g. gemini-2.5-flash-lite in 2026);
  * benches a provider after a 429 (honouring `retry-after`) instead of hammering it;
  * caps concurrent requests per provider so parallel agents don't trip rate limits;
  * returns validated Pydantic objects (via `instructor`) or plain text.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal, TypeVar

import instructor
import openai
from openai import AsyncOpenAI
from pydantic import BaseModel

from .config import Settings, get_settings

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
Tier = Literal["fast", "strong"]

BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "mistral": "https://api.mistral.ai/v1",
}
# Free-tier friendly parallelism per provider.
CONCURRENCY = {"gemini": 3, "groq": 3, "openrouter": 1, "nvidia": 2, "mistral": 1}
# Markdown-JSON parsing works with any model; native JSON mode is faster where supported.
INSTRUCTOR_MODE = {"openrouter": instructor.Mode.MD_JSON, "nvidia": instructor.Mode.MD_JSON}


class LLMUnavailable(RuntimeError):
    """Raised when every configured provider failed or is rate limited."""


class LLMNotConfigured(LLMUnavailable):
    """Raised when no provider API key is set at all."""


@dataclass
class _Provider:
    name: str
    client: AsyncOpenAI
    structured: instructor.AsyncInstructor
    models: dict[str, list[str]]
    sem: asyncio.Semaphore
    benched_until: float = 0.0  # whole provider (bad key, network down)
    dead_models: set[str] = field(default_factory=set)  # retired / not available on this account
    model_benched_until: dict[str, float] = field(default_factory=dict)  # per-model rate limits

    def live_models(self, tier: str) -> list[str]:
        return [m for m in self.models[tier] if m not in self.dead_models]

    def ready_models(self, tier: str) -> list[str]:
        now = time.monotonic()
        return [m for m in self.live_models(tier) if self.model_benched_until.get(m, 0) <= now]

    def next_ready_in(self, tier: str) -> float:
        """Seconds until some model of this tier is usable again (inf if none can ever be)."""
        now = time.monotonic()
        waits = [max(self.benched_until, self.model_benched_until.get(m, 0)) - now for m in self.live_models(tier)]
        return min(waits) if waits else float("inf")


@dataclass
class RouterStats:
    calls: Counter = field(default_factory=Counter)
    failures: Counter = field(default_factory=Counter)


def root_api_error(e: BaseException) -> BaseException:
    """`instructor` wraps provider errors (InstructorRetryException); dig out the openai error."""
    seen = 0
    cur: BaseException | None = e
    while cur is not None and seen < 8:
        if isinstance(cur, (openai.APIStatusError, openai.APIConnectionError)):
            return cur
        for attempt in getattr(cur, "failed_attempts", None) or []:
            exc = getattr(attempt, "exception", None)
            if isinstance(exc, (openai.APIStatusError, openai.APIConnectionError)):
                return exc
        cur = cur.__cause__ or cur.__context__
        seen += 1
    return e


class LLMRouter:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.stats = RouterStats()
        self._providers: list[_Provider] = []
        for name in self.settings.configured_providers():
            client = AsyncOpenAI(
                api_key=self.settings.api_key(name),
                base_url=BASE_URLS[name],
                timeout=self.settings.llm_timeout_s,
                max_retries=0,  # the router does its own failover
            )
            self._providers.append(
                _Provider(
                    name=name,
                    client=client,
                    structured=instructor.from_openai(client, mode=INSTRUCTOR_MODE.get(name, instructor.Mode.JSON)),
                    models={tier: self.settings.models(name, tier) for tier in ("fast", "strong")},
                    sem=asyncio.Semaphore(CONCURRENCY[name]),
                )
            )

    @property
    def provider_names(self) -> list[str]:
        return [p.name for p in self._providers]

    # ------------------------------------------------------------------ public API
    async def structured(
        self,
        response_model: type[T],
        messages: list[dict],
        tier: Tier = "fast",
        max_tokens: int | None = None,
    ) -> T:
        async def call(p: _Provider, model: str) -> T:
            return await p.structured.chat.completions.create(
                model=model,
                response_model=response_model,
                messages=messages,
                max_retries=1,
                **({"max_tokens": max_tokens} if max_tokens else {}),
            )

        return await self._run(call, tier, f"structured:{response_model.__name__}")

    async def text(
        self,
        messages: list[dict],
        tier: Tier = "strong",
        max_tokens: int | None = None,
        validate=None,
    ) -> str:
        """`validate(content)` may raise ValueError to reject an answer and try the next model."""

        async def call(p: _Provider, model: str) -> str:
            resp = await p.client.chat.completions.create(
                model=model,
                messages=messages,
                **({"max_tokens": max_tokens} if max_tokens else {}),
            )
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                raise ValueError("empty completion")
            if validate:
                validate(content)
            return content

        return await self._run(call, tier, "text")

    # ------------------------------------------------------------------ internals
    async def _run(self, call, tier: str, label: str):
        if not self._providers:
            raise LLMNotConfigured(
                "No LLM provider is configured. Set at least one of GEMINI_API_KEY, "
                "GROQ_API_KEY, OPENROUTER_API_KEY, NVIDIA_API_KEY or MISTRAL_API_KEY."
            )
        errors: list[str] = []
        for attempt in range(2):  # second pass waits out the shortest cooldown
            for p in self._providers:
                if p.benched_until > time.monotonic():
                    continue
                for model in p.ready_models(tier):
                    outcome = await self._try(p, model, call, label, errors)
                    if outcome is not _NEXT_MODEL:
                        if outcome is _NEXT_PROVIDER:
                            break
                        self.stats.calls[f"{p.name}:{model}"] += 1
                        return outcome
            if attempt == 0:
                waits = [p.next_ready_in(tier) for p in self._providers]
                soonest = min(waits) if waits else 3600
                if 0 < soonest <= 45:
                    log.info("All providers busy; waiting %.0fs", soonest)
                    await asyncio.sleep(soonest)
                    continue
                if soonest <= 0:
                    continue  # failures were not rate limits; one more pass
            break
        raise LLMUnavailable(f"All LLM providers failed for {label}: " + "; ".join(errors[-6:]))

    async def _try(self, p: _Provider, model: str, call, label: str, errors: list[str]):
        try:
            async with p.sem:
                return await call(p, model)
        except Exception as raw:
            e = root_api_error(raw)
            self.stats.failures[p.name] += 1
            if isinstance(e, openai.RateLimitError):
                # Free-tier quotas (Gemini RPM, Groq TPM, ...) are per model: bench just this model.
                p.model_benched_until[model] = time.monotonic() + self._retry_after(e)
                errors.append(f"{p.name}/{model}: rate limited")
                result = _NEXT_MODEL
            elif isinstance(e, (openai.AuthenticationError, openai.PermissionDeniedError)) or getattr(e, "status_code", 0) == 402:
                self._bench(p, 3600)  # bad key / payment required — stop trying this provider
                errors.append(f"{p.name}: access denied ({getattr(e, 'status_code', '?')})")
                result = _NEXT_PROVIDER
            elif isinstance(e, openai.NotFoundError):
                p.dead_models.add(model)  # retired or not on this account — try the next model
                errors.append(f"{p.name}/{model}: model not found")
                result = _NEXT_MODEL
            elif isinstance(e, openai.InternalServerError):
                errors.append(f"{p.name}/{model}: server error {e.status_code}")  # often model-specific
                result = _NEXT_MODEL
            elif isinstance(e, (openai.APIConnectionError, openai.APITimeoutError)):
                self._bench(p, 15)
                errors.append(f"{p.name}: {type(e).__name__}")
                result = _NEXT_PROVIDER
            else:  # schema validation failure, bad JSON, 400 … — another model may manage
                errors.append(f"{p.name}/{model}: {type(raw).__name__}: {str(raw)[:120]}")
                result = _NEXT_MODEL
            log.warning("LLM %s failed on %s/%s: %s", label, p.name, model, errors[-1])
            return result

    def _bench(self, p: _Provider, seconds: float) -> None:
        p.benched_until = max(p.benched_until, time.monotonic() + seconds)

    def _retry_after(self, err: openai.APIStatusError) -> float:
        try:
            value = err.response.headers.get("retry-after")
            if value:
                return min(float(value), 300.0)
        except Exception:
            pass
        # Gemini and Groq put the wait in the message body ("retry in 51.1s", "try again in 16.6s").
        m = re.search(r"(?:retry in|try again in|retryDelay['\"]?:\s*['\"]?)\s*([\d.]+)\s*s", str(err), re.I)
        if m:
            return min(float(m.group(1)) + 1, 300.0)
        return self.settings.rate_limit_cooldown_s


_NEXT_MODEL = object()
_NEXT_PROVIDER = object()
_router: LLMRouter | None = None


def get_router() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router
