from __future__ import annotations

import json
import os
import time
from typing import Callable, Optional

from loguru import logger

from .gemini_adapter import GeminiAdapter
from .models import (
    InsightAgentState,
    InsightInput,
    InsightOutput,
    InsightResult,
    SynthesisContext,
)
from .validators import validate_inputs, validate_insights


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROMPTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SYSTEM = """You are a senior financial advisor and strategic insight specialist for Finora AI.
You synthesize financial data, patterns, and forecasts into clear, actionable recommendations for CFOs.
Every insight must be grounded in specific evidence from the provided data.
Return ONLY valid JSON — no markdown fences, no explanation outside the JSON object.
"""

_CONTEXT_PROMPT = """
Extract and synthesize the key signals from these three financial sources.

Financial data:
{data}

Detected patterns:
{patterns}

Forecast summary:
{forecast}

Return EXACTLY this JSON:
{{
  "latest_revenue":        1500000.0,
  "latest_net_profit":     350000.0,
  "latest_profit_margin":  0.233,
  "revenue_trend":         "growing|stable|declining|volatile",
  "data_period":           "Q1 2023 to Q1 2024",
  "currency":              "USD",
  "top_pattern":           "single most important pattern in one sentence",
  "pattern_count":         4,
  "has_anomaly":           false,
  "anomaly_description":   "",
  "forecast_growth_rate":  0.08,
  "forecast_confidence":   "High|Medium|Low",
  "forecast_period":       "Q2 2025",
  "forecast_revenue":      1620000.0,
  "overall_health_signal": "growing|stable|declining|volatile",
  "synthesis_notes":       "any important caveats or data quality notes"
}}

Rules:
- All monetary values in the same unit (do not mix millions and billions)
- latest_profit_margin as decimal: 23% → 0.23
- forecast_growth_rate as decimal: 8% → 0.08
- If a field cannot be determined, use null
"""

_INSIGHTS_PROMPT = """
Synthesize all findings into strategic insights for a CFO.

Context summary:
{context}

Original financial data:
{data}

Patterns:
{patterns}

Forecast:
{forecast}

User question: "{question}"

Return EXACTLY this JSON:
{{
  "direct_answer": "direct, specific answer to the user's question in 2-3 sentences using actual numbers",
  "insights": [
    {{
      "title":       "Insight title in 5-8 words",
      "explanation": "2-3 sentence explanation of the insight",
      "evidence":    "specific data point: e.g. 'Revenue grew from $1.2M to $1.5M (+25%) over 5 quarters'",
      "urgency":     "immediate|short_term|long_term|monitor"
    }},
    {{
      "title":       "Second insight title",
      "explanation": "Explanation...",
      "evidence":    "Specific evidence...",
      "urgency":     "short_term"
    }},
    {{
      "title":       "Third insight title",
      "explanation": "Explanation...",
      "evidence":    "Specific evidence...",
      "urgency":     "long_term"
    }}
  ],
  "actions": [
    {{
      "action":          "Specific action verb + what to do",
      "rationale":       "Why this action, grounded in the data",
      "expected_impact": "Quantified or qualified expected outcome",
      "urgency":         "immediate|short_term|long_term|monitor"
    }},
    {{
      "action":          "Second action",
      "rationale":       "Why...",
      "expected_impact": "Impact...",
      "urgency":         "short_term"
    }}
  ],
  "key_risk":           "One specific risk in 1-2 sentences with data reference",
  "health_score":       7,
  "health_explanation": "2 sentence explanation of why this score, with evidence",
  "health_trend":       "improving|stable|declining|volatile",
  "executive_summary":  "2-3 sentence CFO-level summary of overall financial position and outlook"
}}

Critical rules:
- direct_answer must reference actual numbers from the data
- Every insight must have specific evidence (no vague statements)
- health_score: 1=critical, 5=average, 8=strong, 10=exceptional
- Tone: confident, specific, senior financial advisor speaking to a CFO
- Return ONLY valid JSON
"""

_RETRY_ADDENDUM = """

IMPORTANT — Previous response had these quality issues:
{errors}

Fix ALL of these in your new response.
Pay special attention to:
- Every insight must have specific evidence (cite actual numbers)
- direct_answer must be at least 20 characters and include specific numbers
- executive_summary must be at least 30 characters
"""

_NARRATIVE_PROMPT = """
Write a polished, CFO-ready financial insight narrative.

Insights data:
{insights_json}

Context:
- Data period: {data_period}
- Latest revenue: {latest_revenue}
- Health score: {health_score}/10
- Health trend: {health_trend}

User asked: "{question}"

Write a 4-5 sentence executive-level narrative that:
1. Opens with the health score and overall position
2. States the top strategic insight with specific numbers
3. Names the top recommended action and its expected impact
4. Closes with the key risk to watch

Return EXACTLY this JSON:
{{
  "narrative": "your 4-5 sentence paragraph"
}}

Rules:
- Tone: senior financial advisor briefing a CFO — confident, specific, no hedging
- Use actual numbers (e.g. '$1.5M', '8% growth', '23% margin')
- No bullet points inside the narrative
- Write in present/future tense, not past
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class InsightAgent:
    """
    Production DeepAgent for financial insight synthesis.

    Drop-in replacement for:
        def insight_agent(data, patterns, forecast, question) -> str

    Usage:
        agent  = InsightAgent()              # reads GEMINI_API_KEY from env
        result = await agent.run(data, patterns, forecast, question)

        if result.success:
            print(f"Health: {result.health_score}/10")
            print(f"Answer: {result.direct_answer}")
            for ins in result.insights:
                print(f"  → {ins['title']}: {ins['evidence']}")
    """

    def __init__(
        self,
        model:       str            = "gemini-2.5-flash",
        api_key:     Optional[str]  = None,
        max_retries: int            = 2,
        temperature: float          = 0.1,
    ):
        resolved_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Set it in .env or pass api_key= to InsightAgent()."
            )

        self.model       = model
        self.max_retries = max_retries
        self.agent_name  = "InsightAgent"

        self._gemini = GeminiAdapter(
            api_key=resolved_key,
            model_name=model,
            temperature=temperature,
            max_output_tokens=3000,
        )

    # ──────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ──────────────────────────────────────────────────────────

    async def run(
        self,
        data:     str,
        patterns: str,
        forecast: str,
        question: str,
    ) -> InsightResult:
        """
        Exact same signature as the original insight_agent().

        Returns InsightResult — typed, validated, with full audit trail.
        """
        t_start = time.perf_counter()

        # Pre-validate with Pydantic before touching state
        try:
            validated = InsightInput(
                data=data, patterns=patterns,
                forecast=forecast, question=question
            )
        except Exception as e:
            return InsightResult(
                success=False,
                errors=[{"node": "pre_validation", "error": str(e)}],
                model_used=self.model,
                duration_ms=round((time.perf_counter() - t_start) * 1000, 1),
            )

        state = InsightAgentState(
            data=validated.data,
            patterns=validated.patterns,
            forecast=validated.forecast,
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
                f"health={result.health_score}/10 | insights={len(result.insights)}"
            )
        else:
            logger.error(
                f"❌ [{self.agent_name}] {duration_ms}ms | errors={result.errors}"
            )

        return result

    # ──────────────────────────────────────────────────────────
    # GRAPH ENGINE
    # ──────────────────────────────────────────────────────────

    async def _execute_graph(self, state: InsightAgentState) -> InsightAgentState:
        graph: list[tuple[str, Callable, bool]] = [
            ("validate_inputs",   self._node_validate_inputs,   True),
            ("build_context",     self._node_build_context,     True),
            ("generate_insights", self._node_generate_insights, True),
            ("validate_insights", self._node_validate_insights, False),
            ("enrich_narrative",  self._node_enrich_narrative,  False),
            ("format_output",     self._node_format_output,     False),
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

    async def _node_validate_inputs(self, state: InsightAgentState) -> InsightAgentState:
        state = validate_inputs(state)
        if not state.input_valid:
            raise ValueError(
                "Input validation failed:\n"
                + "\n".join(f"  • {e}" for e in state.input_errors)
            )
        logger.debug(
            f"    inputs valid | data={len(state.data)}c "
            f"patterns={len(state.patterns)}c "
            f"forecast={len(state.forecast)}c"
        )
        return state

    # ──────────────────────────────────────────────────────────
    # NODE 2 — BUILD CONTEXT
    # ──────────────────────────────────────────────────────────

    async def _node_build_context(self, state: InsightAgentState) -> InsightAgentState:
        """
        Gemini extracts structured key numbers from all three text inputs.
        Gives Node 3 a clean view instead of three raw text blobs.
        """
        prompt = _CONTEXT_PROMPT.format(
            data=state.data,
            patterns=state.patterns,
            forecast=state.forecast,
        )
        raw = await self._llm_json(prompt)

        try:
            state.context = SynthesisContext.model_validate(raw)
        except Exception as e:
            raise ValueError(f"Context validation failed: {e} | raw={raw}")

        ctx = state.context
        logger.debug(
            f"    context: revenue={ctx.latest_revenue} "
            f"margin={ctx.latest_profit_margin} "
            f"health={ctx.overall_health_signal} "
            f"anomaly={ctx.has_anomaly}"
        )
        return state

    # ──────────────────────────────────────────────────────────
    # NODE 3 — GENERATE INSIGHTS
    # ──────────────────────────────────────────────────────────

    async def _node_generate_insights(self, state: InsightAgentState) -> InsightAgentState:
        """Core insight generation — Gemini synthesizes all findings."""
        context_str = (
            state.context.model_dump_json(indent=2)
            if state.context else "{}"
        )

        prompt = _INSIGHTS_PROMPT.format(
            context=context_str,
            data=state.data,
            patterns=state.patterns,
            forecast=state.forecast,
            question=state.question,
        )

        # On retry: add quality-error context for self-correction
        if state.retry_count > 0 and state.insights_errors:
            prompt += _RETRY_ADDENDUM.format(
                errors="\n".join(f"  - {e}" for e in state.insights_errors)
            )

        raw = await self._llm_json(prompt)

        try:
            state.insights = InsightOutput.model_validate(raw)
        except Exception as e:
            raise ValueError(f"InsightOutput validation failed: {e} | raw={raw}")

        ins = state.insights
        logger.debug(
            f"    insights: score={ins.health_score}/10 "
            f"insights={len(ins.insights)} "
            f"actions={len(ins.actions)}"
        )
        return state

    # ──────────────────────────────────────────────────────────
    # NODE 4 — VALIDATE INSIGHTS
    # ──────────────────────────────────────────────────────────

    async def _node_validate_insights(self, state: InsightAgentState) -> InsightAgentState:
        """Quality checks — auto-retries Node 3 if output is low-quality."""
        state = validate_insights(state)

        if not state.insights_valid:
            if state.retry_count <= state.max_retries:
                logger.warning(
                    f"    insights invalid (attempt {state.retry_count}/{state.max_retries}): "
                    f"{state.insights_errors}"
                )
                state = await self._node_generate_insights(state)
                state = validate_insights(state)

            if not state.insights_valid:
                logger.error(
                    f"    insights still invalid after {state.retry_count} retries: "
                    f"{state.insights_errors}"
                )
        else:
            logger.debug("    insights valid ✓")

        return state

    # ──────────────────────────────────────────────────────────
    # NODE 5 — ENRICH NARRATIVE
    # ──────────────────────────────────────────────────────────

    async def _node_enrich_narrative(self, state: InsightAgentState) -> InsightAgentState:
        """Write a CFO-ready paragraph — replaces the original raw text output."""
        if state.insights is None:
            state.narrative = "Insights could not be generated due to earlier errors."
            return state

        ctx = state.context
        prompt = _NARRATIVE_PROMPT.format(
            insights_json=state.insights.model_dump_json(indent=2),
            data_period=ctx.data_period if ctx else "unknown",
            latest_revenue=ctx.latest_revenue if ctx else "unknown",
            health_score=state.insights.health_score,
            health_trend=state.insights.health_trend.value,
            question=state.question,
        )

        raw = await self._llm_json(prompt)
        state.narrative = raw.get("narrative", "").strip()

        if not state.narrative:
            state.narrative = state.insights.executive_summary

        logger.debug(f"    narrative: {state.narrative[:80]}...")
        return state

    # ──────────────────────────────────────────────────────────
    # NODE 6 — FORMAT OUTPUT
    # ──────────────────────────────────────────────────────────

    async def _node_format_output(self, state: InsightAgentState) -> InsightAgentState:
        return state  # _build_result reads state directly

    # ──────────────────────────────────────────────────────────
    # GEMINI LLM HELPER
    # ──────────────────────────────────────────────────────────

    async def _llm_json(self, user_prompt: str) -> dict:
        return await self._gemini.generate_json(
            system_prompt=_SYSTEM,
            user_prompt=user_prompt,
        )

    # ──────────────────────────────────────────────────────────
    # RESULT BUILDER
    # ──────────────────────────────────────────────────────────

    def _build_result(self, state: InsightAgentState, duration_ms: float) -> InsightResult:
        input_ok  = state.input_valid
        plan_ok   = state.insights_valid and state.insights is not None
        no_errors = len(state.errors) == 0
        success   = input_ok and plan_ok and no_errors

        base = InsightResult(
            success=success,
            model_used=self.model,
            duration_ms=duration_ms,
            retry_count=state.retry_count,
            node_history=state.node_history,
            errors=state.errors,
            narrative=state.narrative,
        )

        if state.context:
            ctx = state.context
            base.latest_revenue       = ctx.latest_revenue
            base.data_period          = ctx.data_period
            base.forecast_growth_rate = ctx.forecast_growth_rate
            if ctx.forecast_growth_rate is not None:
                base.forecast_growth_pct = f"{ctx.forecast_growth_rate * 100:.1f}%"

        if state.insights:
            ins = state.insights
            base.direct_answer      = ins.direct_answer
            base.key_risk           = ins.key_risk
            base.health_score       = ins.health_score
            base.health_explanation = ins.health_explanation
            base.health_trend       = ins.health_trend.value
            base.executive_summary  = ins.executive_summary

            base.insights = [
                {
                    "title":       i.title,
                    "explanation": i.explanation,
                    "evidence":    i.evidence,
                    "urgency":     i.urgency.value,
                }
                for i in ins.insights
            ]

            base.actions = [
                {
                    "action":          a.action,
                    "rationale":       a.rationale,
                    "expected_impact": a.expected_impact,
                    "urgency":         a.urgency.value,
                }
                for a in ins.actions
            ]

        return base
