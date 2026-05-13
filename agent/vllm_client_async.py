import asyncio
from typing import Any, Dict, Optional

import httpx
from openai import AsyncOpenAI


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
        async with self._semaphore:
            response = await self._client.chat.completions.create(**payload)
        return response.model_dump()

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
