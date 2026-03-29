"""
agents/Pattern_Agent/dspy_modules.py
──────────────────────────────────────
DSPy Signatures and Modules for every LLM-calling node in the PatternAgent graph.

NODE MAP:
  Node 2 — parse_financials  → ParseFinancialsModule
  Node 3 — detect_patterns   → DetectPatternsModule
  Node 5 — enrich_narrative  → NarrativeModule

REPLACING GeminiAdapter:
  BEFORE: raw = await self._gemini.generate_json(system_prompt, user_prompt)
  AFTER:  raw = await self._parse_mod.acall(data=...)
          raw = await self._detect_mod.acall(parsed=..., question=...)
          raw = await self._narrative_mod.acall(patterns_json=..., ...)
"""

from __future__ import annotations

from typing import Optional

import dspy

from ..deep_agent.dspy_module import FinancialJSONModule


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DSPy SIGNATURES  (typed contracts per node)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ParseFinancialsSignature(dspy.Signature):
    """
    Extract structured financial time-series rows from raw data text.
    Returns periods with revenue, expenses, net_profit, margins, and growth rates.
    """
    data: str = dspy.InputField(desc="Raw financial data string containing time-series records")

    periods:      list[dict] = dspy.OutputField(desc="List of period dicts: period, revenue, expenses, net_profit, profit_margin, growth_rate")
    currency:     str        = dspy.OutputField(desc="Currency code e.g. USD")
    data_period:  str        = dspy.OutputField(desc="Human-readable period range e.g. 'Q1 2023 to Q4 2024'")
    num_periods:  int        = dspy.OutputField(desc="Number of distinct periods parsed")
    has_quarterly: bool      = dspy.OutputField(desc="True if quarterly data is present")
    has_annual:   bool       = dspy.OutputField(desc="True if annual data is present")
    parsing_notes: str       = dspy.OutputField(desc="Notes about data quality or gaps")


class DetectPatternsSignature(dspy.Signature):
    """
    Detect patterns, anomalies, and trends in structured financial time-series data.
    Every finding must cite specific numbers as evidence.
    """
    parsed:   str = dspy.InputField(desc="JSON-serialised ParsedFinancials object")
    question: str = dspy.InputField(desc="User's financial question for context")

    patterns:          list[dict]     = dspy.OutputField(desc="List of pattern dicts: name, description, evidence, severity, periods_affected")
    anomalies:         list[dict]     = dspy.OutputField(desc="List of anomaly dicts: description, period, magnitude, severity, possible_cause")
    trend_direction:   str            = dspy.OutputField(desc="growing|declining|stable|volatile|mixed")
    trend_explanation: str            = dspy.OutputField(desc="2-sentence explanation with specific numbers")
    yoy_comparisons:   list[dict]     = dspy.OutputField(desc="Year-over-year comparison dicts: metric, prior_value, current_value, change_pct, direction")
    key_finding:       str            = dspy.OutputField(desc="Most important finding with specific numbers (min 10 chars)")
    revenue_growth_rate: Optional[float] = dspy.OutputField(desc="Revenue growth rate as decimal (e.g. 0.08 for 8%)")
    expense_growth_rate: Optional[float] = dspy.OutputField(desc="Expense growth rate as decimal")
    margin_trend:      str            = dspy.OutputField(desc="expanding|contracting|stable")
    seasonal_patterns: list[str]      = dspy.OutputField(desc="List of seasonal pattern descriptions")


class PatternNarrativeSignature(dspy.Signature):
    """
    Write a 3-4 sentence analyst-quality financial pattern report paragraph.
    Must include specific numbers and state the dominant trend, key pattern, and key finding.
    """
    patterns_json: str = dspy.InputField(desc="JSON-serialised PatternOutput object")
    data_period:   str = dspy.InputField(desc="Period range covered by the analysis")
    num_periods:   int = dspy.InputField(desc="Number of periods analysed")
    question:      str = dspy.InputField(desc="User question for context")

    narrative: str = dspy.OutputField(desc="3-4 sentence analyst paragraph with specific numbers")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DSPy MODULES  (one per LLM node)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ParseFinancialsModule(dspy.Module):
    """
    Node 2 DSPy Module: extract structured time-series from raw data.

    Wraps FinancialJSONModule with the PatternAgent _PARSE_PROMPT template.
    ParseFinancialsSignature documents the typed output contract.
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
        """Returns parsed JSON dict for ParsedFinancials.model_validate()."""
        user_prompt = self._template.format(data=data)
        return await self._json_module.acall(self._system, user_prompt)


class DetectPatternsModule(dspy.Module):
    """
    Node 3 DSPy Module: detect patterns, anomalies, and trends.

    Wraps FinancialJSONModule with the PatternAgent _DETECT_PROMPT template.
    Supports retry augmentation: caller appends _RETRY_ADDENDUM when retry_count > 0.
    """

    def __init__(
        self,
        json_module: FinancialJSONModule,
        system_prompt: str,
        detect_prompt_template: str,
        retry_addendum_template: str,
    ):
        super().__init__()
        self._json_module    = json_module
        self._system         = system_prompt
        self._template       = detect_prompt_template
        self._retry_addendum = retry_addendum_template

    async def acall(
        self,
        parsed: str,
        question: str,
        retry_errors: Optional[list[str]] = None,
    ) -> dict:
        """Returns parsed JSON dict for PatternOutput.model_validate()."""
        user_prompt = self._template.format(parsed=parsed, question=question)
        if retry_errors:
            user_prompt += self._retry_addendum.format(
                errors="\n".join(f"  - {e}" for e in retry_errors)
            )
        return await self._json_module.acall(self._system, user_prompt)


class PatternNarrativeModule(dspy.Module):
    """
    Node 5 DSPy Module: write an analyst-quality pattern narrative paragraph.

    Wraps FinancialJSONModule with the _NARRATIVE_PROMPT template.
    Returns a dict with a single "narrative" key.
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
        patterns_json: str,
        data_period:   str,
        num_periods:   int,
        question:      str,
    ) -> dict:
        """Returns {"narrative": str}."""
        user_prompt = self._template.format(
            patterns_json=patterns_json,
            data_period=data_period,
            num_periods=num_periods,
            question=question,
        )
        return await self._json_module.acall(self._system, user_prompt)
