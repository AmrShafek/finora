"""
agents/forecast_agent/dspy_modules.py
───────────────────────────────────────
DSPy Signatures and Modules for every LLM-calling node in the ForecastAgent graph.

NODE MAP:
  Node 2 — parse_financials   → ParseFinancialsModule
  Node 3 — generate_forecast  → GenerateForecastModule
  Node 5 — enrich_narrative   → ForecastNarrativeModule

REPLACING GeminiAdapter:
  BEFORE: raw = await self._gemini.generate_json(system_prompt, user_prompt)
  AFTER:  raw = await self._parse_mod.acall(data=...)
          raw = await self._forecast_mod.acall(snapshot=..., patterns=..., question=...)
          raw = await self._narrative_mod.acall(forecast_json=..., ...)
"""

from __future__ import annotations

from typing import Optional

import dspy

from ..deep_agent.dspy_module import FinancialJSONModule


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DSPy SIGNATURES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ParseSnapshotSignature(dspy.Signature):
    """
    Extract a compact financial snapshot from raw historical data.
    Returns the latest figures, average growth rate, and trend direction.
    """
    data: str = dspy.InputField(desc="Raw historical financial data string")

    latest_revenue:       Optional[float] = dspy.OutputField(desc="Most recent revenue figure")
    latest_expenses:      Optional[float] = dspy.OutputField(desc="Most recent expenses figure")
    latest_net_profit:    Optional[float] = dspy.OutputField(desc="Most recent net profit figure")
    latest_profit_margin: Optional[float] = dspy.OutputField(desc="Profit margin as decimal e.g. 0.20 for 20%")
    avg_growth_rate:      Optional[float] = dspy.OutputField(desc="Average growth rate as decimal e.g. 0.12 for 12%")
    trend_direction:      str             = dspy.OutputField(desc="growing|declining|stable|volatile")
    data_period:          str             = dspy.OutputField(desc="e.g. '2021 Q1 to 2024 Q4'")
    currency:             str             = dspy.OutputField(desc="Currency code e.g. USD")
    num_periods:          int             = dspy.OutputField(desc="Number of periods parsed")
    parsing_notes:        str             = dspy.OutputField(desc="Caveats about data quality or gaps")


class GenerateForecastSignature(dspy.Signature):
    """
    Generate a structured financial forecast with short-term and annual projections.
    All numbers must be grounded in the historical snapshot — no hallucination.
    """
    snapshot: str = dspy.InputField(desc="JSON-serialised FinancialSnapshot")
    patterns: str = dspy.InputField(desc="Pattern analysis from PatternAgent (truncated to 500 chars)")
    question: str = dspy.InputField(desc="User's forecast question")

    short_term: dict          = dspy.OutputField(desc="Short-term forecast: {period, revenue, expenses, net_profit, reasoning}")
    annual:     dict          = dspy.OutputField(desc="Annual forecast: {year, revenue, expenses, net_profit, profit_margin, reasoning}")
    growth_rate: float        = dspy.OutputField(desc="Expected growth rate as decimal e.g. 0.12 for 12%")
    growth_reasoning: str     = dspy.OutputField(desc="Why this growth rate is expected")
    risks:      list[str]     = dspy.OutputField(desc="2-3 specific, grounded risks")
    confidence: str           = dspy.OutputField(desc="High|Medium|Low")
    confidence_explanation: str = dspy.OutputField(desc="Why this confidence level")
    assumptions: list[str]   = dspy.OutputField(desc="Key assumptions underlying the forecast")


class ForecastNarrativeSignature(dspy.Signature):
    """
    Write a 3-4 sentence professional forecast narrative with specific numbers.
    Covers short-term/annual forecasts, growth drivers, top 2 risks, and confidence.
    """
    forecast_json:   str = dspy.InputField(desc="JSON-serialised ForecastOutput")
    data_period:     str = dspy.InputField(desc="Historical data period")
    latest_revenue:  str = dspy.InputField(desc="Most recent revenue figure")
    trend_direction: str = dspy.InputField(desc="Historical trend direction")
    patterns:        str = dspy.InputField(desc="Pattern summary (truncated)")
    question:        str = dspy.InputField(desc="User question for context")

    narrative: str = dspy.OutputField(desc="3-4 sentence professional forecast paragraph")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DSPy MODULES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ParseSnapshotModule(dspy.Module):
    """
    Node 2 DSPy Module: extract financial snapshot from raw data.
    Wraps FinancialJSONModule with _PARSE_PROMPT from ForecastAgent.
    """

    def __init__(
        self,
        json_module: FinancialJSONModule,
        system_prompt: str,
        parse_prompt_template: str,
    ):
        super().__init__()
        self._json_module = json_module
        self._system      = system_prompt
        self._template    = parse_prompt_template

    async def acall(self, data: str) -> dict:
        """Returns dict for FinancialSnapshot.model_validate()."""
        user_prompt = self._template.format(data=data)
        return await self._json_module.acall(self._system, user_prompt)


class GenerateForecastModule(dspy.Module):
    """
    Node 3 DSPy Module: generate structured financial forecast.
    Wraps FinancialJSONModule with _FORECAST_PROMPT template.
    Appends _RETRY_ADDENDUM when retry_errors are provided.
    """

    def __init__(
        self,
        json_module: FinancialJSONModule,
        system_prompt: str,
        forecast_prompt_template: str,
        retry_addendum_template: str,
    ):
        super().__init__()
        self._json_module    = json_module
        self._system         = system_prompt
        self._template       = forecast_prompt_template
        self._retry_addendum = retry_addendum_template

    async def acall(
        self,
        snapshot: str,
        patterns: str,
        question: str,
        retry_errors: Optional[list[str]] = None,
    ) -> dict:
        """Returns dict for ForecastOutput.model_validate()."""
        user_prompt = self._template.format(
            snapshot=snapshot,
            patterns=patterns,
            question=question,
        )
        if retry_errors:
            user_prompt += self._retry_addendum.format(
                errors="\n".join(f"  - {e}" for e in retry_errors)
            )
        return await self._json_module.acall(self._system, user_prompt)


class ForecastNarrativeModule(dspy.Module):
    """
    Node 5 DSPy Module: write professional forecast narrative.
    Wraps FinancialJSONModule with _NARRATIVE_PROMPT template.
    Returns {"narrative": str}.
    """

    def __init__(
        self,
        json_module: FinancialJSONModule,
        system_prompt: str,
        narrative_prompt_template: str,
    ):
        super().__init__()
        self._json_module = json_module
        self._system      = system_prompt
        self._template    = narrative_prompt_template

    async def acall(
        self,
        forecast_json:   str,
        data_period:     str,
        latest_revenue:  str,
        trend_direction: str,
        patterns:        str,
        question:        str,
    ) -> dict:
        """Returns {"narrative": str}."""
        user_prompt = self._template.format(
            forecast_json=forecast_json,
            data_period=data_period,
            latest_revenue=latest_revenue,
            trend_direction=trend_direction,
            patterns=patterns,
            question=question,
        )
        return await self._json_module.acall(self._system, user_prompt)
