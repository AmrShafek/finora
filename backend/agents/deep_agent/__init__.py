"""
agents/deep_agent/__init__.py
─────────────────────────────
Shared infrastructure for the DeepAgent + DSPy + LangGraph refactor.

Exports:
  make_dspy_lm         — factory for dspy.LM configured with Gemini via LiteLLM
  FinancialJSONModule  — universal DSPy module replacing GeminiAdapter.generate_json()

Usage (in each agent's __init__ or agent file):
    from agents.deep_agent import make_dspy_lm, FinancialJSONModule

    lm     = make_dspy_lm(model="gemini-2.5-flash", temperature=0.0)
    module = FinancialJSONModule(lm=lm)
    result = await module.acall(system_prompt, user_prompt)  # → dict
"""

from .dspy_config import make_dspy_lm
from .dspy_module import FinancialJSONModule

__all__ = ["make_dspy_lm", "FinancialJSONModule"]
