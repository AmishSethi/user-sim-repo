"""Lightweight async client supporting both local vLLM servers and OpenRouter.

Each backbone is registered as a `ServerConfig` with an `alias`, an optional
`base_url` (vLLM server) or sentinel "openrouter", and a `model` slug. The
client routes requests accordingly. Bounded retries; OpenRouter key from env.
"""
from __future__ import annotations
import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

import httpx


OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _load_openrouter_key() -> str:
    """Read the OpenRouter API key from the OPENROUTER_API_KEY env var."""
    env = os.environ.get("OPENROUTER_API_KEY")
    if env:
        return env.strip()
    raise RuntimeError(
        "OpenRouter key not found. Set OPENROUTER_API_KEY in your environment."
    )


@dataclass
class ServerConfig:
    alias: str            # human-readable alias used by the experiment scripts
    model: str            # model slug as known to the backend
    base_url: Optional[str] = None  # local vLLM URL, or None for OpenRouter
    provider: str = "vllm"          # "vllm" or "openrouter"


class LLMClient:
    """Async chat client over any number of vLLM and OpenRouter backends.

    The client never logs the OpenRouter API key. Bounded exponential backoff
    on transient HTTP errors. Unified `chat(alias, messages)` interface so
    experiment code does not need to know whether a model is local or remote.
    """

    def __init__(self, servers: List[ServerConfig], timeout: float = 240.0,
                 max_retries: int = 5):
        self.servers = {s.alias: s for s in servers}
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(timeout=timeout)
        self._openrouter_key: Optional[str] = None
        if any(s.provider == "openrouter" for s in servers):
            self._openrouter_key = _load_openrouter_key()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _endpoint_and_headers(self, s: ServerConfig) -> tuple[str, Dict[str, str]]:
        if s.provider == "openrouter":
            return (
                f"{OPENROUTER_BASE}/chat/completions",
                {
                    "Authorization": f"Bearer {self._openrouter_key}",
                },
            )
        assert s.base_url, f"vllm server {s.alias} missing base_url"
        return f"{s.base_url}/chat/completions", {}

    async def chat(self, alias: str, messages: List[Dict[str, str]],
                   max_tokens: int = 256, temperature: float = 0.7,
                   stop: Optional[List[str]] = None) -> str:
        if alias not in self.servers:
            raise KeyError(f"unknown server alias: {alias}")
        s = self.servers[alias]
        endpoint, headers = self._endpoint_and_headers(s)
        payload: Dict[str, Any] = {
            "model": s.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop is not None:
            payload["stop"] = stop

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                r = await self._client.post(endpoint, json=payload, headers=headers)
                if r.status_code == 429:
                    # rate limit: longer backoff
                    await asyncio.sleep(min(30.0, 2.0 ** attempt))
                    continue
                if r.status_code >= 500:
                    await asyncio.sleep(min(15.0, 1.5 ** attempt))
                    continue
                r.raise_for_status()
                data = r.json()
                msg = data["choices"][0]["message"]
                content = msg.get("content")
                if content is None or content == "":
                    # reasoning models (e.g. deepseek-r1) sometimes spend the
                    # whole budget on reasoning tokens; surface those so the
                    # caller can still parse a tool call.
                    reasoning = msg.get("reasoning") or ""
                    content = reasoning
                return content or ""
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_err = e
                await asyncio.sleep(min(15.0, 1.5 ** attempt))
            except Exception as e:
                last_err = e
                await asyncio.sleep(min(5.0, 1.2 ** attempt))
        raise RuntimeError(f"chat failed for {alias} after {self.max_retries} retries: {last_err}")

    async def health(self, alias: str) -> bool:
        s = self.servers[alias]
        if s.provider == "openrouter":
            # We just confirm the key is loaded; we trust OpenRouter is up.
            return self._openrouter_key is not None
        try:
            r = await self._client.get(f"{s.base_url}/models")
            return r.status_code == 200
        except Exception:
            return False


# ---------- canonical 6-model panel for the position-paper experiments ----------
PANEL_OPENROUTER: List[ServerConfig] = [
    # small (3)
    ServerConfig(alias="llama3.1-8b",  provider="openrouter",
                 model="meta-llama/llama-3.1-8b-instruct"),
    ServerConfig(alias="qwen2.5-7b",   provider="openrouter",
                 model="qwen/qwen-2.5-7b-instruct"),
    ServerConfig(alias="gemma3-12b",   provider="openrouter",
                 model="google/gemma-3-12b-it"),
    # medium / large (3)
    ServerConfig(alias="llama3.3-70b", provider="openrouter",
                 model="meta-llama/llama-3.3-70b-instruct"),
    ServerConfig(alias="gpt-oss-120b", provider="openrouter",
                 model="openai/gpt-oss-120b"),
    ServerConfig(alias="deepseek-r1",  provider="openrouter",
                 model="deepseek/deepseek-r1"),
]

PANEL_ALIASES: List[str] = [s.alias for s in PANEL_OPENROUTER]
