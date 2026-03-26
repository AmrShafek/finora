from __future__ import annotations

import json
import os
import time
from typing import Callable, Optional

from loguru import logger

from .gemini_adapter import GeminiAdapter
from .models import (
    ParsedFinancials,
    PatternAgentState,
    PatternInput,
    PatternOutput,
    PatternResult,
)
from .validators import validate_inputs, validate_patterns


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROMPTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SYSTEM = """You are a financial pattern detection specialist for Finora AI.
You analyse time-series financial data to detect trends, anomalies, and patterns.
Base every finding on specific numbers from the provided data — never generalise.
Return ONLY valid JSON — no markdown fences, no explanation outside the JSON object.
"""

_PARSE_PROMPT = """
Extract the financial time-series rows from this data.

Raw data:
{data}

Return EXACTLY this JSON:
{{
  "periods": [
    {{
      "period":        "Q1 2023",
      "revenue":       1200000.0,
      "expenses":      960000.0,
      "net_profit":    240000.0,
      "profit_margin": 0.20,
      "growth_rate":   null
    }}
  ],
  "currency":      "USD",
  "data_period":   "Q1 2023 to Q1 2024",
  "num_periods":   5,
  "has_quarterly": true,
  "has_annual":    false,
  "parsing_notes": "any notes about data quality or gaps"
}}

Rules:
- One entry per distinct period (Q1 2023, Q2 2023, etc.)
- profit_margin as decimal: 20% → 0.20
- growth_rate: QoQ or YoY change as decimal (null if only one period)
- All monetary values in the same unit
- If a value is missing, use null — do not invent numbers
"""

_DETECT_PROMPT = """
Detect all patterns, anomalies, and trends in this financial time-series data.

Parsed financial periods:
{parsed}

User question context: "{question}"

Return EXACTLY this JSON:
{{
  "patterns": [
    {{
      "name":             "Pattern name in 4-6 words",
      "description":      "1-2 sentence explanation of the pattern",
      "evidence":         "specific numbers e.g. 'Revenue grew from $1.2M to $1.5M (+25%) over 5 quarters'",
      "severity":         "high|medium|low",
      "periods_affected": ["Q1 2023", "Q2 2023"]
    }}
  ],
  "anomalies": [
    {{
      "description":    "what happened",
      "period":         "Q4 2023",
      "magnitude":      "+32% vs prior quarter average",
      "severity":       "high|medium|low",
      "possible_cause": "possible explanation"
    }}
  ],
  "trend_direction":   "growing|declining|stable|volatile|mixed",
  "trend_explanation": "2 sentence explanation with specific numbers",
  "yoy_comparisons": [
    {{
      "metric":        "revenue",
      "prior_value":   1200000.0,
      "current_value": 1500000.0,
      "change_pct":    0.25,
      "direction":     "growing"
    }}
  ],
  "key_finding":          "most important finding in one sentence with specific numbers",
  "revenue_growth_rate":  0.08,
  "expense_growth_rate":  0.06,
  "margin_trend":         "expanding|contracting|stable",
  "seasonal_patterns":    ["Q4 typically shows higher expenses"]
}}

Detection rules:
- Patterns: consistent behaviour across ≥ 2 periods (trend, seasonality, leverage)
- Anomalies: one-period deviation of ≥ 15% from rolling average
- key_finding must include specific numbers, not generic statements
- growth rates as decimals: 8% → 0.08
- anomalies list can be empty [] if none detected
- Return ONLY valid JSON
"""

_RETRY_ADDENDUM = """

IMPORTANT — Previous response failed these quality checks:
{errors}

Fix ALL of these. Key requirements:
- Every pattern must have specific numbers in its evidence field
- key_finding must be ≥ 10 chars and include specific figures
- trend_explanation must be non-empty
"""

_NARRATIVE_PROMPT = """
Write a concise, analyst-quality financial pattern report.

Pattern findings:
{patterns_json}

Data period: {data_period}
Number of periods analysed: {num_periods}

User asked: "{question}"

Write a 3-4 sentence paragraph that:
1. States the dominant trend with specific numbers
2. Names the most important pattern or anomaly
3. Gives the key finding
4. Notes any seasonal patterns if present

Return EXACTLY this JSON:
{{
  "narrative": "your 3-4 sentence paragraph"
}}

Rules:
- Include specific numbers (e.g. '$1.5M', '+25%', '5 consecutive quarters')
- Professional analyst tone — objective and precise
- No bullet points inside the narrative
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PatternAgent:
    """
    Production DeepAgent for financial pattern detection.

    Drop-in replacement for:
        def pattern_agent(data: str, question: str) -> str

    Usage:
        agent  = PatternAgent()
        result = await agent.run(data, question)

        if result.success:
            print(result.key_finding)
            print(result.trend_direction)
            for p in result.patterns:
                print(f"  [{p['severity'].upper()}] {p['name']}: {p['evidence']}")
            for a in result.anomalies:
                print(f"  ANOMALY [{a['period']}]: {a['description']}")
    """

    def __init__(
        self,
        model:       str           = "gemini-2.5-flash",
        api_key:     Optional[str] = None,
        max_retries: int           = 2,
        temperature: float         = 0.0,
    ):
        resolved_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Set it in .env or pass api_key= to PatternAgent()."
            )

        self.model       = model
        self.max_retries = max_retries
        self.agent_name  = "PatternAgent"

        self._gemini = GeminiAdapter(
            api_key=resolved_key,
            model_name=model,
            temperature=temperature,
            max_output_tokens=3000,
        )

    # ──────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ──────────────────────────────────────────────────────────

    async def run(self, data: str, question: str) -> PatternResult:
        """
        Exact same signature as the original pattern_agent().

        Returns PatternResult — typed, validated, with full audit trail.
        """
        t_start = time.perf_counter()

        try:
            validated = PatternInput(data=data, question=question)
        except Exception as e:
            return PatternResult(
                success=False,
                errors=[{"node": "pre_validation", "error": str(e)}],
                model_used=self.model,
                duration_ms=round((time.perf_counter() - t_start) * 1000, 1),
            )

        state = PatternAgentState(
            data=validated.data,
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
                f"trend={result.trend_direction} | "
                f"patterns={len(result.patterns)} | "
                f"anomalies={len(result.anomalies)}"
            )
        else:
            logger.error(f"❌ [{self.agent_name}] {duration_ms}ms | errors={result.errors}")

        return result

    # ──────────────────────────────────────────────────────────
    # GRAPH ENGINE
    # ──────────────────────────────────────────────────────────

    async def _execute_graph(self, state: PatternAgentState) -> PatternAgentState:
        graph: list[tuple[str, Callable, bool]] = [
            ("validate_inputs",  self._node_validate_inputs,  True),
            ("parse_financials", self._node_parse_financials, True),
            ("detect_patterns",  self._node_detect_patterns,  True),
            ("validate_patterns",self._node_validate_patterns,False),
            ("enrich_narrative", self._node_enrich_narrative, False),
            ("format_output",    self._node_format_output,    False),
        ]

        for node_name, node_fn, is_critical in graph:
            t = time.perf_counter()
            logger.debug(f"  ▶ Node [{node_name}]")
            try:
                state = await node_fn(state)
                ms    = round((time.perf_counter() - t) * 1000, 1)
                state.node_history.append(
                    {"node": node_name, "status": "completed", "ms": ms}
                )
                logger.debug(f"  ✅ [{node_name}] {ms}ms")
            except Exception as exc:
                ms  = round((time.perf_counter() - t) * 1000, 1)
                msg = f"{type(exc).__name__}: {exc}"
                state.node_history.append(
                    {"node": node_name, "status": "failed", "ms": ms, "error": msg}
                )
                state.errors.append({"node": node_name, "error": msg})
                logger.error(f"  ❌ [{node_name}] FAILED: {msg}")
                if is_critical:
                    logger.error("  🛑 Critical node — aborting graph")
                    break

        return state

    # ──────────────────────────────────────────────────────────
    # NODE 1 — VALIDATE INPUTS
    # ──────────────────────────────────────────────────────────

    async def _node_validate_inputs(self, state: PatternAgentState) -> PatternAgentState:
        state = validate_inputs(state)
        if not state.input_valid:
            raise ValueError(
                "Input validation failed:\n"
                + "\n".join(f"  • {e}" for e in state.input_errors)
            )
        logger.debug(f"    inputs valid | data={len(state.data)}c")
        return state

    # ──────────────────────────────────────────────────────────
    # NODE 2 — PARSE FINANCIALS
    # ──────────────────────────────────────────────────────────

    async def _node_parse_financials(self, state: PatternAgentState) -> PatternAgentState:
        """Gemini extracts clean time-series rows — gives Node 3 structured numbers."""
        prompt = _PARSE_PROMPT.format(data=state.data)
        raw    = await self._llm_json(prompt)

        try:
            state.parsed = ParsedFinancials.model_validate(raw)
        except Exception as e:
            raise ValueError(f"ParsedFinancials validation failed: {e} | raw={raw}")

        logger.debug(
            f"    parsed: {state.parsed.num_periods} periods | "
            f"{state.parsed.data_period}"
        )
        return state

    # ──────────────────────────────────────────────────────────
    # NODE 3 — DETECT PATTERNS
    # ──────────────────────────────────────────────────────────

    async def _node_detect_patterns(self, state: PatternAgentState) -> PatternAgentState:
        """Core pattern detection — Gemini analyses the structured time-series."""
        parsed_str = (
            state.parsed.model_dump_json(indent=2)
            if state.parsed else "{}"
        )

        prompt = _DETECT_PROMPT.format(
            parsed=parsed_str,
            question=state.question,
        )

        if state.retry_count > 0 and state.patterns_errors:
            prompt += _RETRY_ADDENDUM.format(
                errors="\n".join(f"  - {e}" for e in state.patterns_errors)
            )

        raw = await self._llm_json(prompt)

        try:
            state.patterns = PatternOutput.model_validate(raw)
        except Exception as e:
            raise ValueError(f"PatternOutput validation failed: {e} | raw={raw}")

        out = state.patterns
        logger.debug(
            f"    detected: {len(out.patterns)} patterns | "
            f"{len(out.anomalies)} anomalies | "
            f"trend={out.trend_direction}"
        )
        return state

    # ──────────────────────────────────────────────────────────
    # NODE 4 — VALIDATE PATTERNS
    # ──────────────────────────────────────────────────────────

    async def _node_validate_patterns(self, state: PatternAgentState) -> PatternAgentState:
        """Quality checks with auto-retry."""
        state = validate_patterns(state)

        if not state.patterns_valid:
            if state.retry_count <= state.max_retries:
                logger.warning(
                    f"    patterns invalid (attempt {state.retry_count}/{state.max_retries}): "
                    f"{state.patterns_errors}"
                )
                state = await self._node_detect_patterns(state)
                state = validate_patterns(state)

            if not state.patterns_valid:
                logger.error(
                    f"    patterns still invalid after {state.retry_count} retries: "
                    f"{state.patterns_errors}"
                )
        else:
            logger.debug("    patterns valid ✓")

        return state

    # ──────────────────────────────────────────────────────────
    # NODE 5 — ENRICH NARRATIVE
    # ──────────────────────────────────────────────────────────

    async def _node_enrich_narrative(self, state: PatternAgentState) -> PatternAgentState:
        """Write analyst-quality summary paragraph."""
        if state.patterns is None:
            state.narrative = "Pattern detection could not be completed due to earlier errors."
            return state

        prompt = _NARRATIVE_PROMPT.format(
            patterns_json=state.patterns.model_dump_json(indent=2),
            data_period=state.parsed.data_period if state.parsed else "unknown",
            num_periods=state.parsed.num_periods if state.parsed else 0,
            question=state.question,
        )

        raw = await self._llm_json(prompt)
        state.narrative = raw.get("narrative", "").strip()

        if not state.narrative:
            state.narrative = state.patterns.key_finding

        logger.debug(f"    narrative: {state.narrative[:80]}...")
        return state

    # ──────────────────────────────────────────────────────────
    # NODE 6 — FORMAT OUTPUT
    # ──────────────────────────────────────────────────────────

    async def _node_format_output(self, state: PatternAgentState) -> PatternAgentState:
        return state

    # ──────────────────────────────────────────────────────────
    # LLM HELPER
    # ──────────────────────────────────────────────────────────

    async def _llm_json(self, user_prompt: str) -> dict:
        return await self._gemini.generate_json(
            system_prompt=_SYSTEM,
            user_prompt=user_prompt,
        )

    # ──────────────────────────────────────────────────────────
    # RESULT BUILDER
    # ──────────────────────────────────────────────────────────

    def _build_result(self, state: PatternAgentState, duration_ms: float) -> PatternResult:
        success = (
            state.input_valid
            and state.patterns_valid
            and state.patterns is not None
            and len(state.errors) == 0
        )

        base = PatternResult(
            success=success,
            model_used=self.model,
            duration_ms=duration_ms,
            retry_count=state.retry_count,
            node_history=state.node_history,
            errors=state.errors,
            narrative=state.narrative,
        )

        if state.parsed:
            base.data_period = state.parsed.data_period
            base.num_periods = state.parsed.num_periods

        if state.patterns:
            out = state.patterns
            base.key_finding       = out.key_finding
            base.trend_direction   = out.trend_direction.value
            base.trend_explanation = out.trend_explanation
            base.margin_trend      = out.margin_trend
            base.seasonal_patterns = out.seasonal_patterns
            base.has_anomalies     = len(out.anomalies) > 0

            base.revenue_growth_rate = out.revenue_growth_rate
            if out.revenue_growth_rate is not None:
                base.revenue_growth_pct = f"{out.revenue_growth_rate * 100:.1f}%"
            base.expense_growth_rate = out.expense_growth_rate

            base.patterns = [
                {
                    "name":             p.name,
                    "description":      p.description,
                    "evidence":         p.evidence,
                    "severity":         p.severity.value,
                    "periods_affected": p.periods_affected,
                }
                for p in out.patterns
            ]

            base.anomalies = [
                {
                    "description":   a.description,
                    "period":        a.period,
                    "magnitude":     a.magnitude,
                    "severity":      a.severity.value,
                    "possible_cause":a.possible_cause,
                }
                for a in out.anomalies
            ]

            base.yoy_comparisons = [
                {
                    "metric":        y.metric,
                    "prior_value":   y.prior_value,
                    "current_value": y.current_value,
                    "change_pct":    y.change_pct,
                    "direction":     y.direction.value,
                }
                for y in out.yoy_comparisons
            ]

        return base
