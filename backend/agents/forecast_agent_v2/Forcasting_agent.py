from __future__ import annotations

import json
import os
import time
from typing import Callable, Optional

from loguru import logger

from .gemini_adapter import GeminiAdapter
from .models import (
    FinancialSnapshot,
    ForecastAgentState,
    ForecastInput,
    ForecastOutput,
    ForecastResult,
    QuarterForecast,
    AnnualForecast,
)
from .validators import validate_inputs, validate_forecast


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROMPTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SYSTEM = """You are an expert financial forecasting specialist for Finora AI.
You analyse Salesforce financial data and produce precise, evidence-based forecasts.
Base every number on the actual data provided — never hallucinate figures.
Return ONLY valid JSON — no markdown fences, no explanation outside the JSON object.
"""

_PARSE_PROMPT = """
Parse the key financial numbers from this historical data.

Raw data:
{data}

Return EXACTLY this JSON:
{{
  "latest_revenue":       1234567.89,
  "latest_expenses":      987654.32,
  "latest_net_profit":    246913.57,
  "latest_profit_margin": 0.20,
  "avg_growth_rate":      0.12,
  "trend_direction":      "growing|declining|stable|volatile",
  "data_period":          "2021 Q1 to 2024 Q4",
  "currency":             "USD",
  "num_periods":          16,
  "parsing_notes":        "any caveats about data quality or gaps"
}}

Rules:
- All monetary values in the SAME currency unit (do not mix millions and billions)
- profit_margin as decimal: 20% → 0.20
- avg_growth_rate as decimal: 12% → 0.12
- If a field cannot be determined from the data, use null
- trend_direction: growing=consistent up, declining=consistent down,
                   stable=<5% change, volatile=large swings
"""

_FORECAST_PROMPT = """
Generate a structured financial forecast.

Parsed financial snapshot:
{snapshot}

Detected patterns:
{patterns}

User question: {question}

Based on the historical data and patterns above, return EXACTLY this JSON:
{{
  "short_term": {{
    "period":     "Q2 2025",
    "revenue":    1300000.00,
    "expenses":   1040000.00,
    "net_profit": 260000.00,
    "reasoning":  "one sentence grounded in the data"
  }},
  "annual": {{
    "year":           2025,
    "revenue":        5200000.00,
    "expenses":       4160000.00,
    "net_profit":     1040000.00,
    "profit_margin":  0.20,
    "reasoning":      "one sentence grounded in the data"
  }},
  "growth_rate":      0.12,
  "growth_reasoning": "explains why this growth rate is expected",
  "risks": [
    "Risk 1 — specific and grounded",
    "Risk 2 — specific and grounded",
    "Risk 3 — specific and grounded"
  ],
  "confidence":      "High|Medium|Low",
  "confidence_explanation": "why this confidence level",
  "assumptions": [
    "Assumption 1",
    "Assumption 2"
  ]
}}

Critical rules:
- Use EXACT numbers from the snapshot as the baseline — do not round aggressively
- growth_rate as decimal: 12% → 0.12
- profit_margin as decimal: 20% → 0.20
- annual.revenue must be ≈ 4× short_term.revenue (it covers a full year)
- risks must reference actual patterns, not generic statements
- Return ONLY valid JSON
"""

_RETRY_ADDENDUM = """

IMPORTANT — Previous forecast had these validation errors:
{errors}

Fix ALL of these errors in your new response.
Pay special attention to:
- annual.revenue should be ~4× short_term.revenue
- All monetary values in the same unit
- growth_rate as a decimal (0.12 not 12)
"""

_NARRATIVE_PROMPT = """
Write a concise, professional financial forecast narrative for a business audience.

Forecast data:
{forecast_json}

Historical context:
- Data period: {data_period}
- Latest revenue: {latest_revenue}
- Trend: {trend_direction}
- Patterns: {patterns}

User asked: {question}

Write 3–4 sentences that:
1. State the short-term and annual forecasts with actual numbers
2. Explain the growth rate estimate and its key drivers
3. Name the top 2 risks
4. Give the confidence level and why

Return EXACTLY this JSON:
{{
  "narrative": "your 3-4 sentence paragraph here"
}}

Rules:
- Use specific numbers from the forecast (e.g. '$1.3M', '12% growth')
- Professional but readable — not overly technical
- Do NOT use bullet points inside the narrative
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ForecastAgent:
    """
    Production DeepAgent for financial forecasting.

    Drop-in replacement for:
        def forecast_agent(data, patterns, question) -> str

    Usage:
        agent  = ForecastAgent()                  # reads GEMINI_API_KEY from env
        result = await agent.run(data, patterns, question)

        if result.success:
            print(result.short_term_revenue)      # e.g. 1_300_000.0
            print(result.growth_rate_pct)         # e.g. "12.0%"
            print(result.narrative)               # full paragraph
            print(result.risks)                   # ["Risk 1", "Risk 2"]
        else:
            print(result.errors)
    """

    def __init__(
        self,
        model: str       = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        max_retries: int = 2,
        temperature: float = 0.1,     # slight creativity for narrative
    ):
        resolved_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Set it in .env or pass api_key= to ForecastAgent()."
            )

        self.model       = model
        self.max_retries = max_retries
        self.agent_name  = "ForecastAgent"

        self._gemini = GeminiAdapter(
            api_key=resolved_key,
            model_name=model,
            temperature=temperature,
            max_output_tokens=3000,
        )

    # ──────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT — drop-in for forecast_agent()
    # ──────────────────────────────────────────────────────────

    async def run(
        self,
        data:     str,
        patterns: str,
        question: str,
    ) -> ForecastResult:
        """
        Exact same signature as the original forecast_agent().

        Args:
            data:     Historical financial data string.
            patterns: Patterns from Pattern Agent.
            question: User question / forecast context.

        Returns:
            ForecastResult — typed, validated, with full audit trail.
        """
        t_start = time.perf_counter()

        # Pre-validate with Pydantic before touching state
        try:
            validated = ForecastInput(data=data, patterns=patterns, question=question)
        except Exception as e:
            return ForecastResult(
                success=False,
                errors=[{"node": "pre_validation", "error": str(e)}],
                model_used=self.model,
                duration_ms=round((time.perf_counter() - t_start) * 1000, 1),
            )

        state = ForecastAgentState(
            data=validated.data,
            patterns=validated.patterns,
            question=validated.question,
            max_retries=self.max_retries,
        )

        logger.info(f"🤖 [{self.agent_name}] Starting | model={self.model}")
        logger.info(f"   Q: {question[:80]}{'...' if len(question) > 80 else ''}")

        state       = await self._execute_graph(state)
        duration_ms = round((time.perf_counter() - t_start) * 1000, 1)
        result      = self._build_result(state, duration_ms)

        if result.success:
            logger.success(
                f"✅ [{self.agent_name}] {duration_ms}ms | "
                f"confidence={result.confidence} | growth={result.growth_rate_pct}"
            )
        else:
            logger.error(
                f"❌ [{self.agent_name}] {duration_ms}ms | errors={result.errors}"
            )

        return result

    # ──────────────────────────────────────────────────────────
    # GRAPH EXECUTION ENGINE
    # ──────────────────────────────────────────────────────────

    async def _execute_graph(self, state: ForecastAgentState) -> ForecastAgentState:
        graph: list[tuple[str, Callable, bool]] = [
            ("validate_inputs",   self._node_validate_inputs,   True),
            ("parse_financials",  self._node_parse_financials,  True),
            ("generate_forecast", self._node_generate_forecast, True),
            ("validate_forecast", self._node_validate_forecast, False),
            ("enrich_narrative",  self._node_enrich_narrative,  False),
            ("format_output",     self._node_format_output,     False),
        ]

        for node_name, node_fn, is_critical in graph:
            t = time.perf_counter()
            logger.debug(f"  ▶ Node [{node_name}]")

            try:
                state = await node_fn(state)
                ms    = round((time.perf_counter() - t) * 1000, 1)
                state.node_history.append({
                    "node": node_name, "status": "completed", "ms": ms
                })
                logger.debug(f"  ✅ [{node_name}] {ms}ms")

            except Exception as exc:
                ms  = round((time.perf_counter() - t) * 1000, 1)
                msg = f"{type(exc).__name__}: {exc}"
                state.node_history.append({
                    "node": node_name, "status": "failed", "ms": ms, "error": msg
                })
                state.errors.append({"node": node_name, "error": msg})
                logger.error(f"  ❌ [{node_name}] FAILED: {msg}")

                if is_critical:
                    logger.error("  🛑 Critical node — aborting graph")
                    break

        return state

    # ──────────────────────────────────────────────────────────
    # NODE 1 — VALIDATE INPUTS  (no LLM)
    # ──────────────────────────────────────────────────────────

    async def _node_validate_inputs(self, state: ForecastAgentState) -> ForecastAgentState:
        """
        Pure validation — data has numbers, patterns is real,
        question is financial and injection-free.
        """
        state = validate_inputs(state)
        if not state.input_valid:
            raise ValueError(
                "Input validation failed:\n"
                + "\n".join(f"  • {e}" for e in state.input_errors)
            )
        logger.debug(f"    inputs valid | data={len(state.data)} chars | q='{state.question[:50]}'")
        return state

    # ──────────────────────────────────────────────────────────
    # NODE 2 — PARSE FINANCIALS  (Gemini call)
    # ──────────────────────────────────────────────────────────

    async def _node_parse_financials(self, state: ForecastAgentState) -> ForecastAgentState:
        """
        Ask Gemini to extract structured numbers from the raw data string.
        This gives Node 3 clean numeric inputs instead of a raw blob.
        """
        prompt = _PARSE_PROMPT.format(data=state.data)
        raw    = await self._llm_json(prompt)

        try:
            state.snapshot = FinancialSnapshot.model_validate(raw)
        except Exception as e:
            raise ValueError(f"Financial snapshot validation failed: {e} | raw={raw}")

        snap = state.snapshot
        logger.debug(
            f"    snapshot: revenue={snap.latest_revenue} "
            f"growth={snap.avg_growth_rate} "
            f"trend={snap.trend_direction} "
            f"periods={snap.num_periods}"
        )
        return state

    # ──────────────────────────────────────────────────────────
    # NODE 3 — GENERATE FORECAST  (Gemini call)
    # ──────────────────────────────────────────────────────────

    async def _node_generate_forecast(self, state: ForecastAgentState) -> ForecastAgentState:
        """
        Core forecasting call — Gemini produces the structured forecast.
        On retry: appends validation errors so Gemini can self-correct.
        """
        snapshot_str = (
            state.snapshot.model_dump_json(indent=2)
            if state.snapshot
            else "No snapshot available."
        )

        prompt = _FORECAST_PROMPT.format(
            snapshot=snapshot_str,
            patterns=state.patterns,
            question=state.question,
        )

        # On retry: add error context for self-correction
        if state.retry_count > 0 and state.forecast_errors:
            prompt += _RETRY_ADDENDUM.format(
                errors="\n".join(f"  - {e}" for e in state.forecast_errors)
            )

        raw = await self._llm_json(prompt)

        try:
            state.forecast = ForecastOutput.model_validate(raw)
        except Exception as e:
            raise ValueError(f"Forecast output validation failed: {e} | raw={raw}")

        fc = state.forecast
        logger.debug(
            f"    forecast: short={fc.short_term.period} rev={fc.short_term.revenue} "
            f"annual={fc.annual.year} growth={fc.growth_rate:.1%} "
            f"confidence={fc.confidence}"
        )
        return state

    # ──────────────────────────────────────────────────────────
    # NODE 4 — VALIDATE FORECAST  (no LLM, auto-retry)
    # ──────────────────────────────────────────────────────────

    async def _node_validate_forecast(self, state: ForecastAgentState) -> ForecastAgentState:
        """
        8 sanity checks on the forecast numbers.
        If invalid and retries remain → re-runs Node 3 with error context.
        """
        state = validate_forecast(state)

        if not state.forecast_valid:
            if state.retry_count <= state.max_retries:
                logger.warning(
                    f"    forecast invalid (attempt {state.retry_count}/{state.max_retries}): "
                    f"{state.forecast_errors}"
                )
                state = await self._node_generate_forecast(state)
                state = validate_forecast(state)

            if not state.forecast_valid:
                logger.error(
                    f"    forecast still invalid after {state.retry_count} retries: "
                    f"{state.forecast_errors}"
                )
                # Non-critical — graph continues with whatever we have
        else:
            logger.debug("    forecast valid ✓")

        return state

    # ──────────────────────────────────────────────────────────
    # NODE 5 — ENRICH NARRATIVE  (Gemini call)
    # ──────────────────────────────────────────────────────────

    async def _node_enrich_narrative(self, state: ForecastAgentState) -> ForecastAgentState:
        """
        Ask Gemini to write a human-readable forecast paragraph.
        This replaces the original function's raw text output.
        """
        if state.forecast is None:
            state.narrative = "Forecast could not be generated due to earlier errors."
            return state

        prompt = _NARRATIVE_PROMPT.format(
            forecast_json=state.forecast.model_dump_json(indent=2),
            data_period=state.snapshot.data_period if state.snapshot else "unknown",
            latest_revenue=state.snapshot.latest_revenue if state.snapshot else "unknown",
            trend_direction=state.snapshot.trend_direction if state.snapshot else "unknown",
            patterns=state.patterns[:500],   # truncate for token budget
            question=state.question,
        )

        raw = await self._llm_json(prompt)
        state.narrative = raw.get("narrative", "").strip()

        if not state.narrative:
            state.narrative = "Narrative generation returned empty response."

        logger.debug(f"    narrative: {state.narrative[:80]}...")
        return state

    # ──────────────────────────────────────────────────────────
    # NODE 6 — FORMAT OUTPUT
    # ──────────────────────────────────────────────────────────

    async def _node_format_output(self, state: ForecastAgentState) -> ForecastAgentState:
        """No LLM — _build_result() reads state directly."""
        return state

    # ──────────────────────────────────────────────────────────
    # GEMINI LLM HELPER
    # ──────────────────────────────────────────────────────────

    async def _llm_json(self, user_prompt: str) -> dict:
        """Single Gemini call → parsed dict. Same pattern as DataAgent."""
        return await self._gemini.generate_json(
            system_prompt=_SYSTEM,
            user_prompt=user_prompt,
        )

    # ──────────────────────────────────────────────────────────
    # RESULT BUILDER
    # ──────────────────────────────────────────────────────────

    def _build_result(self, state: ForecastAgentState, duration_ms: float) -> ForecastResult:
        """Convert final state into a ForecastResult."""
        input_ok  = state.input_valid
        plan_ok   = state.forecast_valid and state.forecast is not None
        no_errors = len(state.errors) == 0
        success   = input_ok and plan_ok and no_errors

        base = ForecastResult(
            success=success,
            model_used=self.model,
            duration_ms=duration_ms,
            retry_count=state.retry_count,
            node_history=state.node_history,
            errors=state.errors,
            narrative=state.narrative,
        )

        # Snapshot fields
        if state.snapshot:
            base.data_period     = state.snapshot.data_period
            base.latest_revenue  = state.snapshot.latest_revenue
            base.trend_direction = state.snapshot.trend_direction.value

        # Forecast fields
        if state.forecast:
            fc = state.forecast

            base.short_term_period   = fc.short_term.period
            base.short_term_revenue  = fc.short_term.revenue
            base.short_term_expenses = fc.short_term.expenses
            base.short_term_profit   = fc.short_term.net_profit
            base.short_term_reasoning = fc.short_term.reasoning

            base.annual_year      = fc.annual.year
            base.annual_revenue   = fc.annual.revenue
            base.annual_expenses  = fc.annual.expenses
            base.annual_profit    = fc.annual.net_profit
            base.annual_margin    = fc.annual.profit_margin
            base.annual_reasoning = fc.annual.reasoning

            base.growth_rate      = fc.growth_rate
            base.growth_rate_pct  = f"{fc.growth_rate * 100:.1f}%"
            base.growth_reasoning = fc.growth_reasoning

            base.risks               = fc.risks
            base.confidence          = fc.confidence.value
            base.confidence_explanation = fc.confidence_explanation
            base.assumptions         = fc.assumptions

        return base
        