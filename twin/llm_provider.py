"""The one LLM provider: Together AI, through its OpenAI-compatible API (the openai SDK, another base URL).

    TOGETHER_API_KEY    required, read from the environment only (never written anywhere)
    TOGETHER_MODEL      required: a chat model id from your Together account (https://api.together.ai/models)
    TOGETHER_BASE_URL   optional, default https://api.together.xyz/v1

Used by the laptop API's explanations (twin/api.py) and the LLM persona demos (twin/policy/llm.py).
"""
from __future__ import annotations

import os

BASE_URL = "https://api.together.xyz/v1"
SETUP = "Set TOGETHER_API_KEY and TOGETHER_MODEL on the laptop to enable the assistant."


def model() -> str:
    return os.environ.get("TOGETHER_MODEL", "").strip()


def ready() -> bool:
    return bool(os.environ.get("TOGETHER_API_KEY", "").strip() and model())


def connection() -> dict:
    """Keyword arguments for openai.OpenAI(...) that point it at Together."""
    return {"api_key": os.environ.get("TOGETHER_API_KEY", "").strip(),
            "base_url": os.environ.get("TOGETHER_BASE_URL", BASE_URL).strip() or BASE_URL}
