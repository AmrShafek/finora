"""
agents/insight_agent/dspy_modules.py
──────────────────────────────────────
DSPy Signatures and Modules for every LLM-calling node in the InsightAgent graph.

NODE MAP:
  Node 2 — build_context      → BuildContextModule
  Node 3 — generate_insights  → GenerateInsightsModule
  Node 5 — enrich_narrative   → InsightNarrativeModule

REPLACING GeminiAdapter:
  BEFORE: raw = await self._gemini.generate_json(system_prompt, user_prompt)
  AFTER:  raw = await self._context_mod.acall(data=..., patterns=..., forecast=...)
          raw = await self._insights_mod.acall(context=..., data=..., ...)
          raw = await self._narrative_mod.acall(insights_json=..., ...)
"""

from __future__ import annotations

from typing import Optional

import dspy

from ..deep_agent.dspy_module import FinancialJSONModule


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DSPy SIGNATURES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BuildContextSignature(dspy.Signature):
    """
    Extract and synthesize key financial signals from three text sources:
    raw data, pattern analysis, and forecast summary.
    Returns a structured context object for downstream insight generation.
    """
    data:     str = dspy.InputField(desc="Raw financial data string")
    patterns: str = dspy.InputField(desc="Pattern analysis text from PatternAgent")
    forecast: str = dspy.InputField(desc="Forecast summary text from ForecastAgent")

    latest_revenue:        Optional[float] = dspy.OutputField(desc="Most recent revenue")
    latest_net_profit:     Optional[float] = dspy.OutputField(desc="Most recent net profit")
    latest_profit_margin:  Optional[float] = dspy.OutputField(desc="Profit margin as decimal")
    revenue_trend:         str             = dspy.OutputField(desc="growing|stable|declining|volatile")
    data_period:           str             = dspy.OutputField(desc="Data period e.g. 'Q1 2023 to Q1 2024'")
    currency:              str             = dspy.OutputField(desc="Currency code e.g. USD")
    top_pattern:           str             = dspy.OutputField(desc="Most important pattern in one sentence")
    pattern_count:         int             = dspy.OutputField(desc="Number of patterns detected")
    has_anomaly:           bool            = dspy.OutputField(desc="Whether an anomaly was detected")
    anomaly_description:   str             = dspy.OutputField(desc="Anomaly description or empty string")
    forecast_growth_rate:  Optional[float] = dspy.OutputField(desc="Forecast growth rate as decimal")
    forecast_confidence:   str             = dspy.OutputField(desc="High|Medium|Low")
    forecast_period:       str             = dspy.OutputField(desc="Forecast period string")
    forecast_revenue:      Optional[float] = dspy.OutputField(desc="Forecast revenue figure")
    overall_health_signal: str             = dspy.OutputField(desc="growing|stable|declining|volatile")
    synthesis_notes:       str             = dspy.OutputField(desc="Caveats or data quality notes")


class GenerateInsightsSignature(dspy.Signature):
    """
    Synthesize financial context, patterns, and forecasts into strategic insights for a CFO.
    Every insight must have specific evidence grounded in actual numbers.
    """
    context:  str = dspy.InputField(desc="JSON-serialised SynthesisContext")
    data:     str = dspy.InputField(desc="Original financial data")
    patterns: str = dspy.InputField(desc="Pattern analysis text")
    forecast: str = dspy.InputField(desc="Forecast text")
    question: str = dspy.InputField(desc="User question")

    direct_answer:      str        = dspy.OutputField(desc="Direct specific answer with actual numbers (min 20 chars)")
    insights:           list[dict] = dspy.OutputField(desc="List of insight dicts: title, explanation, evidence, urgency")
    actions:            list[dict] = dspy.OutputField(desc="List of action dicts: action, rationale, expected_impact, urgency")
    key_risk:           str        = dspy.OutputField(desc="Top risk in 1-2 sentences with data reference")
    health_score:       int        = dspy.OutputField(desc="Financial health score 1-10")
    health_explanation: str        = dspy.OutputField(desc="2-sentence explanation with evidence")
    health_trend:       str        = dspy.OutputField(desc="improving|stable|declining|volatile")
    executive_summary:  str        = dspy.OutputField(desc="2-3 sentence CFO-level summary (min 30 chars)")


class InsightNarrativeSignature(dspy.Signature):
    """
    Write a 4-5 sentence CFO-ready executive narrative covering health, top insight,
    recommended action, and key risk. Must include specific numbers.
    """
    insights_json:  str = dspy.InputField(desc="JSON-serialised InsightOutput")
    data_period:    str = dspy.InputField(desc="Data period string")
    latest_revenue: str = dspy.InputField(desc="Latest revenue figure")
    health_score:   int = dspy.InputField(desc="Financial health score 1-10")
    health_trend:   str = dspy.InputField(desc="improving|stable|declining|volatile")
    question:       str = dspy.InputField(desc="User question")

    narrative: str = dspy.OutputField(desc="4-5 sentence executive narrative with specific numbers")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DSPy MODULES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BuildContextModule(dspy.Module):
    """
    Node 2 DSPy Module: extract synthesis context from three raw text inputs.
    Wraps FinancialJSONModule with _CONTEXT_PROMPT from InsightAgent.
    """

    def __init__(
        self,
        json_module: FinancialJSONModule,
        system_prompt: str,
        context_prompt_template: str,
    ):
        super().__init__()
        self._json_module = json_module
        self._system      = system_prompt
        self._template    = context_prompt_template

    async def acall(self, data: str, patterns: str, forecast: str) -> dict:
        """Returns dict for SynthesisContext.model_validate()."""
        user_prompt = self._template.format(
            data=data, patterns=patterns, forecast=forecast,
        )
        return await self._json_module.acall(self._system, user_prompt)


class GenerateInsightsModule(dspy.Module):
    """
    Node 3 DSPy Module: generate strategic insights for a CFO.
    Wraps FinancialJSONModule with _INSIGHTS_PROMPT template.
    Appends _RETRY_ADDENDUM when retry_errors are provided.
    """

    def __init__(
        self,
        json_module: FinancialJSONModule,
        system_prompt: str,
        insights_prompt_template: str,
        retry_addendum_template: str,
    ):
        super().__init__()
        self._json_module    = json_module
        self._system         = system_prompt
        self._template       = insights_prompt_template
        self._retry_addendum = retry_addendum_template

    async def acall(
        self,
        context:      str,
        data:         str,
        patterns:     str,
        forecast:     str,
        question:     str,
        retry_errors: Optional[list[str]] = None,
    ) -> dict:
        """Returns dict for InsightOutput.model_validate()."""
        user_prompt = self._template.format(
            context=context, data=data, patterns=patterns,
            forecast=forecast, question=question,
        )
        if retry_errors:
            user_prompt += self._retry_addendum.format(
                errors="\n".join(f"  - {e}" for e in retry_errors)
            )
        return await self._json_module.acall(self._system, user_prompt)


class InsightNarrativeModule(dspy.Module):
    """
    Node 5 DSPy Module: write CFO-ready executive narrative.
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
        insights_json:  str,
        data_period:    str,
        latest_revenue: str,
        health_score:   int,
        health_trend:   str,
        question:       str,
    ) -> dict:
        """Returns {"narrative": str}."""
        user_prompt = self._template.format(
            insights_json=insights_json,
            data_period=data_period,
            latest_revenue=latest_revenue,
            health_score=health_score,
            health_trend=health_trend,
            question=question,
        )
        return await self._json_module.acall(self._system, user_prompt)
