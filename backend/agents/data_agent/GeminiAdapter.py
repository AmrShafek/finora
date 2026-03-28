"""
src/gemini_adapter.py
━━━━━━━━━━━━━━━━━━━━━
GeminiAdapter — wraps google.generativeai into the same
interface the DataAgent expects.

Why a separate adapter file?
  The DataAgent calls self._llm_json(prompt) → dict.
  That single method is the ONLY place that knows about the LLM SDK.
  Swapping SDK = swapping this file only. Agent logic untouched.

Gemini vs OpenAI differences handled here:
  ┌─────────────────────────┬──────────────────────────────┐
  │ OpenAI                  │ Gemini                       │
  ├─────────────────────────┼──────────────────────────────┤
  │ system role in messages │ system_instruction param     │
  │ response_format JSON    │ GenerationConfig mime_type   │
  │ async client built-in   │ generate_content_async()     │
  │ response.choices[0]...  │ response.text                │
  │ temperature in create() │ GenerationConfig             │
  └─────────────────────────┴──────────────────────────────┘
"""

from __future__ import annotations

import json
import re
from typing import Optional

import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from loguru import logger


class GeminiAdapter:
    """
    Thin async wrapper around google.generativeai.GenerativeModel.

    Usage:
        adapter = GeminiAdapter(api_key="...", model_name="gemini-2.5-flash")
        data    = await adapter.generate_json(system_prompt, user_prompt)
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
    ):
        genai.configure(api_key=api_key)

        self.model_name = model_name
        self._gen_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            # Force JSON output — Gemini native JSON mode
            response_mime_type="application/json",
        )

    def _build_model(self, system_instruction: str) -> genai.GenerativeModel:
        """
        Gemini takes system_instruction at model-construction time,
        not per-message. We build a fresh model instance per call.
        This is cheap — no network call happens here.
        """
        return genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_instruction,
            generation_config=self._gen_config,
        )

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        """
        Send one turn to Gemini and return parsed JSON dict.

        Args:
            system_prompt: Injected as Gemini system_instruction.
            user_prompt:   The actual user message content.

        Returns:
            Parsed dict from Gemini's JSON response.

        Raises:
            ValueError: If Gemini returns invalid JSON or an empty response.
            Exception:  Re-raises any Gemini API error (quota, auth, etc.)
        """
        model = self._build_model(system_prompt)

        logger.debug(f"    Gemini call → model={self.model_name} | prompt_len={len(user_prompt)}")

        response = await model.generate_content_async(user_prompt)

        # Safety: check response isn't blocked
        if not response.candidates:
            raise ValueError(
                "Gemini returned no candidates — "
                "response may have been blocked by safety filters."
            )

        raw_text = response.text.strip()

        if not raw_text:
            raise ValueError("Gemini returned an empty response.")

        # Strip markdown fences if Gemini adds them despite mime_type setting
        raw_text = _strip_json_fences(raw_text)

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Gemini returned invalid JSON: {e} "
                f"| first 300 chars: {raw_text[:300]}"
            )


def _strip_json_fences(text: str) -> str:
    """
    Remove ```json ... ``` or ``` ... ``` wrappers if present.
    Gemini sometimes wraps JSON in fences despite response_mime_type=application/json.
    """
    text = text.strip()
    # Match ```json\n...\n``` or ```\n...\n```
    fenced = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text
