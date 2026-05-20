import asyncio
import logging
from typing import Any, Dict, Optional

import httpx
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0  # seconds


class VLLMClientAsync:
    def __init__(
        self,
        base_url: str,
        api_key: str = "dummy",
        max_concurrent: int = 5,
    ) -> None:
        # Limit concurrent requests to vLLM to avoid "Already borrowed" errors
        # caused by the async engine being overwhelmed.
        self._semaphore = asyncio.Semaphore(max_concurrent)
        http_client = httpx.AsyncClient(
            http2=False,
            trust_env=False,  # ignore HTTP_PROXY / HTTPS_PROXY (vLLM is local)
            limits=httpx.Limits(
                max_keepalive_connections=0,
                max_connections=max_concurrent + 10,
            ),
        )
        self._client = AsyncOpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            http_client=http_client,
        )

    async def chat_completions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        last_exc = None
        for attempt in range(MAX_RETRIES):
            async with self._semaphore:
                try:
                    response = await self._client.chat.completions.create(**payload)
                    return response.model_dump()
                except Exception as exc:
                    last_exc = exc
                    msg = str(exc)
                    # Only retry on transient errors (vLLM race condition,
                    # server overload, rate limits).  Genuine 4xx format
                    # errors should fail immediately.
                    is_transient = (
                        "Already borrowed" in msg
                        or getattr(exc, "status_code", None) in (429, 500, 502, 503, 504)
                    )
                    if not is_transient or attempt >= MAX_RETRIES - 1:
                        raise
                    delay = RETRY_BACKOFF_BASE ** (attempt + 1)
                    logger.warning(
                        "vLLM request failed (attempt %d/%d): %s. Retrying in %.1fs…",
                        attempt + 1, MAX_RETRIES, msg, delay,
                    )
                    await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    async def simple_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        tools: Optional[list[dict[str, Any]]] = None,
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
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if extra_payload:
            payload.update(extra_payload)
        return await self.chat_completions(payload)
