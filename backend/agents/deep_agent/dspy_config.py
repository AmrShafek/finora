"""
agents/deep_agent/dspy_config.py
─────────────────────────────────
Factory for dspy.LM — the DSPy replacement for GeminiAdapter.

WHY DSPy?
  BEFORE: Each agent created a GeminiAdapter(api_key, model, temperature)
          using google-generativeai SDK calls directly.
  AFTER:  Each agent calls make_dspy_lm() to get a dspy.LM powered by LiteLLM.
          Swapping the underlying model or provider = one argument change here.

HOW IT WORKS:
  DSPy uses LiteLLM as its backend, which in turn talks to Gemini's REST API.
  LiteLLM requires a provider-prefixed model name: "gemini/gemini-2.5-flash".
  make_dspy_lm() auto-prefixes the short model name so callers don't need to.

ENVIRONMENT:
  Set GEMINI_API_KEY in your .env file.  Alternatively pass api_key= explicitly.
"""

from __future__ import annotations

import os
from typing import Optional

import dspy
from loguru import logger


def make_dspy_lm(
    model: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> dspy.LM:
    """
    Create and return a configured dspy.LM instance.

    Each agent that makes LLM calls creates its own dspy.LM with the
    temperature and token budget appropriate for its task:
      DataAgent:     temperature=0.0   (deterministic planning)
      PatternAgent:  temperature=0.0   (deterministic pattern detection)
      ForecastAgent: temperature=0.1   (slight creativity for narrative)
      InsightAgent:  temperature=0.1   (CFO narrative tone)
      ReportAgent:   temperature=0.2   (natural-sounding report prose)

    Args:
        model:       Gemini model name, e.g. "gemini-2.5-flash".
                     Automatically prefixed with "gemini/" for LiteLLM routing.
        api_key:     Gemini API key. Falls back to GEMINI_API_KEY env var if None.
        temperature: Sampling temperature. 0.0 = fully deterministic (recommended
                     for JSON-output nodes). Slightly higher for narrative nodes.
        max_tokens:  Maximum output tokens for the LM response.

    Returns:
        A configured dspy.LM instance ready for use with FinancialJSONModule.

    Raises:
        ValueError: If no Gemini API key is found.
    """
    resolved_key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not resolved_key:
        raise ValueError(
            "GEMINI_API_KEY not found. "
            "Set it in your .env file or pass api_key= to make_dspy_lm()."
        )

    # Add LiteLLM provider prefix if the caller passed a bare model name
    _KNOWN_PREFIXES = (
        "gemini/", "google/", "vertex_ai/",
        "openai/", "anthropic/", "cohere/", "groq/",
    )
    if not any(model.startswith(p) for p in _KNOWN_PREFIXES):
        litellm_model = f"gemini/{model}"
    else:
        litellm_model = model

    lm = dspy.LM(
        model=litellm_model,
        api_key=resolved_key,
        temperature=temperature,
        max_tokens=max_tokens,
        # cache=False ensures fresh LLM calls (no DSPy response caching)
        cache=False,
    )

    logger.debug(
        f"[DSPy] LM created: model={litellm_model} "
        f"temperature={temperature} max_tokens={max_tokens}"
    )
    return lm
