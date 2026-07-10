"""Plain-chat OpenRouter wrapper used for scenario generation.

Distinct from `mas.OpenRouterLanguageModel`, which adapts OpenRouter to
Concordia's `LanguageModel` interface for use inside simulations.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from openai import OpenAI

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterChatModel:
    """Chat-only model wrapper for any model id OpenRouter exposes."""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        #: `model_name` is the provider-qualified OpenRouter model id
        #: (e.g. "anthropic/claude-opus-4.7") — OpenRouter is the registry here.
        self._model_name = model_name
        self._client = OpenAI(
            base_url=_OPENROUTER_BASE_URL,
            api_key=api_key or os.environ["OPENROUTER_API_KEY"],
        )

    @property
    def name(self) -> str:
        return self._model_name

    def chat(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = 512,
        **kwargs: Any,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,  # type: ignore[arg-type]
            max_completion_tokens=max_new_tokens,
            **kwargs,
        )
        return response.choices[0].message.content or ""
