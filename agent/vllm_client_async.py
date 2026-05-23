import asyncio
import logging
import os
from typing import Any, Dict, Optional

import httpx
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0  # seconds

# API key from environment (checked in order)
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
        max_concurrent: int = 5,
    ) -> None:
        api_key = _resolve_api_key(api_key)

        # Use trust_env for remote APIs (proxy support), skip for local vLLM
        is_local = "127.0.0.1" in base_url or "localhost" in base_url

        http_client = httpx.AsyncClient(
            http2=False,
            trust_env=not is_local,
            limits=httpx.Limits(
                max_keepalive_connections=0 if is_local else 10,
                max_connections=max_concurrent + 10,
            ),
        )
        self._client = AsyncOpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            http_client=http_client,
        )
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._is_local = is_local

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
                    status = getattr(exc, "status_code", None)
                    # Retry on transient errors: rate limits, server errors, vLLM race conditions
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

    @staticmethod
    def _fix_litellm_tool_names(response: Dict[str, Any], tools: Optional[list[dict[str, Any]]]) -> None:
        """Litellm proxies may rename all tools to litellm_unnamed_tool_0.
        Match back by comparing the called parameters against each tool spec's parameters."""
        if not tools:
            return
        tool_names = {t["function"]["name"] for t in tools}
        for choice in response.get("choices", []):
            for tc in (choice.get("message", {}).get("tool_calls", []) or []):
                fn = tc.get("function", {})
                name = fn.get("name", "")
                if name not in tool_names:
                    try:
                        import json as _json
                        called_params = set(_json.loads(fn.get("arguments", "{}")).keys())
                    except Exception:
                        called_params = set()
                    # Find the tool with the most matching parameter names
                    best_match, best_score = None, -1
                    for t in tools:
                        spec_params = set(t["function"].get("parameters", {}).get("properties", {}).keys())
                        if called_params:
                            score = len(called_params & spec_params)
                            if score > best_score:
                                best_score = score
                                best_match = t["function"]["name"]
                    if best_match:
                        fn["name"] = best_match
                        print(f"    [vllm_client] fixed litellm tool: {name} -> {best_match} (params={called_params})", flush=True)

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
        # Skip tool_choice for remote APIs (litellm proxies may crash on it)
        if tool_choice is not None and tool_choice != "auto":
            payload["tool_choice"] = tool_choice
        elif tool_choice == "auto" and self._is_local:
            payload["tool_choice"] = tool_choice
        if extra_payload:
            payload.update(extra_payload)
        resp = await self.chat_completions(payload)
        self._fix_litellm_tool_names(resp, tools)
        return resp
