"""LLM client abstraction supporting OpenAI and Ollama backends."""
from __future__ import annotations

import logging
from collections.abc import Iterator

from askdocs.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Uniform interface to chat-completion APIs.

    Supported providers:
    - "openai"  — uses the openai SDK; set OPENAI_API_KEY + OPENAI_MODEL.
    - "ollama"  — calls the local Ollama REST API; set OLLAMA_BASE_URL + OLLAMA_MODEL.
    """

    def __init__(self, provider: str | None = None) -> None:
        self.provider = provider or settings.llm_provider

    # ── Blocking completion ───────────────────────────────────────────────────

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        if self.provider == "openai":
            return self._openai_complete(system, user, temperature, max_tokens)
        if self.provider == "ollama":
            return self._ollama_complete(system, user, temperature, max_tokens)
        raise ValueError(f"Unknown LLM provider: {self.provider!r}")

    # ── Streaming completion ──────────────────────────────────────────────────

    def stream(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        if self.provider == "openai":
            yield from self._openai_stream(system, user, temperature)
        elif self.provider == "ollama":
            yield from self._ollama_stream(system, user, temperature)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider!r}")

    # ── OpenAI ────────────────────────────────────────────────────────────────

    def _openai_complete(
        self, system: str, user: str, temperature: float, max_tokens: int
    ) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def _openai_stream(
        self, system: str, user: str, temperature: float
    ) -> Iterator[str]:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        stream = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ── Ollama ────────────────────────────────────────────────────────────────

    def _ollama_complete(
        self, system: str, user: str, temperature: float, max_tokens: int
    ) -> str:
        import httpx
        payload = {
            "model": settings.ollama_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        resp = httpx.post(
            f"{settings.ollama_base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def _ollama_stream(
        self, system: str, user: str, temperature: float
    ) -> Iterator[str]:
        import json

        import httpx
        payload = {
            "model": settings.ollama_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
            "options": {"temperature": temperature},
        }
        with httpx.stream(
            "POST",
            f"{settings.ollama_base_url}/api/chat",
            json=payload,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
