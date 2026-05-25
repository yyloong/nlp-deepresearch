import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0  # seconds

# Global per-base_url semaphores — shared across all client instances for the same endpoint.
# Limits total in-flight requests to any given address regardless of how many Agent instances exist.
_URL_SEMAPHORES: Dict[str, asyncio.Semaphore] = {}
_URL_SEMAPHORES_LOCK = asyncio.Lock()


async def _get_url_semaphore(base_url: str, limit: int) -> asyncio.Semaphore:
    async with _URL_SEMAPHORES_LOCK:
        if base_url not in _URL_SEMAPHORES:
            _URL_SEMAPHORES[base_url] = asyncio.Semaphore(limit)
        return _URL_SEMAPHORES[base_url]


def _resolve_api_key(api_key: str) -> str:
    if api_key and api_key != "dummy":
        return api_key
    for var in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "API_KEY"):
        val = os.environ.get(var, "")
        if val:
            return val
    return api_key


class VLLMClientAsync:
    def __init__(
        self,
        base_url: str,
        api_key: str = "dummy",
        max_concurrent: int = 5,       # per-instance limit (legacy, kept for compat)
        max_concurrent_url: int = 0,   # per-address global limit; 0 = read from env or use max_concurrent
    ) -> None:
        api_key = _resolve_api_key(api_key)

        is_local = "127.0.0.1" in base_url or "localhost" in base_url
        self._is_local = is_local
        self._base_url_key = base_url.rstrip("/")

        # Effective per-address limit: explicit arg > env var > max_concurrent
        env_key = "LOCAL_MAX_CONCURRENT" if is_local else "REMOTE_MAX_CONCURRENT"
        env_val = int(os.environ.get(env_key, 0))
        self._url_limit = max_concurrent_url or env_val or max_concurrent

        # Proxy: read from REMOTE_API_PROXY env var (set via secrets.json), only for remote APIs
        proxy = None if is_local else (os.environ.get("REMOTE_API_PROXY") or None)

        http_client = httpx.AsyncClient(
            http2=False,
            trust_env=False,
            proxy=proxy,
            limits=httpx.Limits(
                max_keepalive_connections=0 if is_local else 20,
                max_connections=self._url_limit + 20,
            ),
        )
        self._client = AsyncOpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            http_client=http_client,
        )
        self._url_semaphore: Optional[asyncio.Semaphore] = None  # lazy-init (needs event loop)

    async def _get_semaphore(self) -> asyncio.Semaphore:
        if self._url_semaphore is None:
            self._url_semaphore = await _get_url_semaphore(self._base_url_key, self._url_limit)
        return self._url_semaphore

    async def chat_completions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        sem = await self._get_semaphore()
        last_exc = None
        for attempt in range(MAX_RETRIES):
            async with sem:
                try:
                    response = await self._client.chat.completions.create(**payload)
                    return response.model_dump()
                except Exception as exc:
                    last_exc = exc
                    msg = str(exc)
                    status = getattr(exc, "status_code", None)
                    is_transient = (
                        status in (429, 500, 502, 503, 504)
                        or "Already borrowed" in msg
                        or "rate_limit" in msg.lower()
                        or "overloaded" in msg.lower()
                    )
                    if not is_transient or attempt >= MAX_RETRIES - 1:
                        raise
                    delay = RETRY_BACKOFF_BASE ** (attempt + 1)
                    logger.warning(
                        "API request failed (attempt %d/%d): %s. Retrying in %.1fs…",
                        attempt + 1, MAX_RETRIES, msg, delay,
                    )
                    await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    async def simple_chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None and tool_choice != "auto":
            payload["tool_choice"] = tool_choice
        elif tool_choice == "auto" and self._is_local:
            payload["tool_choice"] = tool_choice
        if extra_payload:
            payload.update(extra_payload)

        return await self.chat_completions(payload)
