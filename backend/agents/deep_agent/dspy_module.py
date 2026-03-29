"""
agents/deep_agent/dspy_module.py
─────────────────────────────────
FinancialJSONModule — the universal DSPy replacement for GeminiAdapter.

BEFORE (GeminiAdapter pattern used in every agent):
    raw = await self._gemini.generate_json(system_prompt, user_prompt)
    # → calls google.generativeai SDK directly

AFTER (FinancialJSONModule):
    raw = await self._dspy_module.acall(system_prompt, user_prompt)
    # → calls dspy.LM (LiteLLM backend) — provider-agnostic

WHY THIS APPROACH?
  1. Prompt preservation:  The carefully crafted JSON-schema prompts in each
     agent are passed VERBATIM to the LM as system + user messages.  No DSPy
     prompt scaffolding is injected that could confuse the JSON output format.
  2. LM-agnostic:         Swapping from Gemini to GPT-4o or Claude = one line
     change in make_dspy_lm(), nothing else changes.
  3. Proper dspy.Module:  This is a first-class DSPy module — it participates
     in DSPy traces, can be compiled with DSPy optimizers (BootstrapFewShot,
     MIPROv2), and benefits from DSPy's observability infrastructure.
  4. Async support:       DSPy's LM is synchronous. acall() wraps forward()
     with asyncio.to_thread() so all async agent nodes work without blocking.
  5. Identical error handling: JSON fence stripping and error messages are
     identical to those in GeminiAdapter for a true drop-in replacement.
"""

from __future__ import annotations

import asyncio
import json
import re

import dspy
from loguru import logger


class FinancialJSONModule(dspy.Module):
    """
    Universal DSPy module that makes one LLM call and returns a parsed JSON dict.

    Every LLM node in every agent creates one of these (or uses a shared instance)
    and calls .acall(system_prompt, user_prompt) → dict.

    This is the ONLY class that knows about DSPy / LiteLLM.
    All agent logic above this remains unchanged from the original DeepAgent code.

    Thread safety:
        DSPy LM calls are synchronous.  Multiple concurrent acall() invocations
        are safe because asyncio.to_thread() dispatches each to a worker thread,
        and dspy.context(lm=...) scopes the LM per-call to avoid cross-agent
        configuration leakage.
    """

    def __init__(self, lm: dspy.LM):
        """
        Args:
            lm: A dspy.LM instance created via deep_agent.make_dspy_lm().
               Each agent creates its own LM with its specific temperature and
               token budget. Multiple agents with different LMs coexist safely.
        """
        super().__init__()
        self._lm = lm

    # ──────────────────────────────────────────────────────────────────────
    # SYNCHRONOUS CORE  (dspy.Module.forward convention)
    # ──────────────────────────────────────────────────────────────────────

    def forward(self, system_prompt: str, user_prompt: str) -> dict:
        """
        Synchronous LM call → parsed JSON dict.

        Calls the DSPy LM directly with chat messages, preserving the exact
        system + user prompt structure from each agent.  This is identical
        in behavior to GeminiAdapter.generate_json().

        Args:
            system_prompt: The agent's system instruction (role + output rules).
            user_prompt:   The formatted user request with data and JSON schema.

        Returns:
            Parsed dict from the LM's JSON response.

        Raises:
            ValueError: If the LM returns empty text or invalid JSON.
            Exception:  Re-raises any LM API error (quota, auth, network).
        """
        logger.debug(
            f"    [DSPy→LM] call | "
            f"sys={len(system_prompt)}c usr={len(user_prompt)}c"
        )

        # dspy.context scopes the LM to this call only, so parallel agents
        # with different temperatures do not interfere with each other.
        with dspy.context(lm=self._lm):
            completions = self._lm(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ]
            )

        # DSPy LM returns a list of completion strings (one per candidate)
        if isinstance(completions, list) and completions:
            raw_text = completions[0]
        elif isinstance(completions, str):
            raw_text = completions
        else:
            raise ValueError(
                f"DSPy LM returned unexpected type: {type(completions).__name__}. "
                "Expected list[str] or str."
            )

        if not raw_text or not str(raw_text).strip():
            raise ValueError("DSPy LM returned an empty response.")

        raw_text = _strip_json_fences(str(raw_text).strip())
        logger.debug(f"    [DSPy←LM] response: {len(raw_text)} chars")

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"DSPy LM returned invalid JSON: {exc} "
                f"| first 300 chars: {raw_text[:300]}"
            ) from exc

    # ──────────────────────────────────────────────────────────────────────
    # ASYNC WRAPPER  (used by all async agent nodes)
    # ──────────────────────────────────────────────────────────────────────

    async def acall(self, system_prompt: str, user_prompt: str) -> dict:
        """
        Async wrapper: runs the synchronous forward() in a thread pool.

        All agent graph nodes are async coroutines.  This bridges DSPy's
        synchronous LM API with asyncio without blocking the event loop.

        Args:
            system_prompt: Passed directly to forward().
            user_prompt:   Passed directly to forward().

        Returns:
            Parsed JSON dict, identical to forward().
        """
        return await asyncio.to_thread(self.forward, system_prompt, user_prompt)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _strip_json_fences(text: str) -> str:
    """
    Remove ```json ... ``` or ``` ... ``` markdown wrappers if present.

    LLMs sometimes add markdown fences even when instructed to return raw JSON.
    This function is identical to GeminiAdapter._strip_json_fences() for
    exact drop-in equivalence.
    """
    text = text.strip()
    fenced = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text
