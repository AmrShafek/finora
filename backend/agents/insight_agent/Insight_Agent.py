from __future__ import annotations

import os
import time
from typing import Optional, Annotated

from loguru import logger
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from operator import add

from .dspy_modules import (
    BuildContextModule,
    GenerateInsightsModule,
    InsightNarrativeModule,
)
from .models import (
    InsightAgentState,
    InsightInput,
    InsightOutput,
    InsightResult,
    SynthesisContext,
)
from .validators import validate_inputs, validate_insights
from ..deep_agent import make_dspy_lm, FinancialJSONModule


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
# LANGGRAPH STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class InsightAgentGraphState(TypedDict):
    """
    LangGraph state for the 6-node InsightAgent graph.
    node_history and errors use Annotated[list, add] for accumulation.
    """
    # Inputs
    data:             str
    patterns:         str
    forecast:         str
    question:         str
    max_retries:      int
    # Node 1
    input_valid:      bool
    input_errors:     list[str]
    # Node 2 — SynthesisContext as dict
    context:          Optional[dict]
    # Node 3 / 4 — InsightOutput as dict
    insights:         Optional[dict]
    insights_valid:   bool
    insights_errors:  list[str]
    retry_count:      int
    # Node 5
    narrative:        str
    # Graph control
    node_history:     Annotated[list[dict], add]
    errors:           Annotated[list[dict], add]
    abort:            bool


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class InsightAgent:
    """
    Production InsightAgent — refactored to DeepAgent + DSPy + LangGraph.

    ARCHITECTURE CHANGES:
      GeminiAdapter    → FinancialJSONModule(dspy.LM)   [DSPy]
      _execute_graph() → LangGraph StateGraph           [LangGraph]
      retry inline     → graph cycle via conditional edge

    PUBLIC INTERFACE — unchanged:
      agent  = InsightAgent()
      result = await agent.run(data, patterns, forecast, question)
    """

    def __init__(
        self,
        model:       str           = "gemini-2.5-flash",
        api_key:     Optional[str] = None,
        max_retries: int           = 2,
        temperature: float         = 0.1,
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

        # ── DSPy: replaces GeminiAdapter (max_tokens=3000 preserved) ─────────
        self._lm     = make_dspy_lm(
            model=model, api_key=resolved_key,
            temperature=temperature, max_tokens=3000,
        )
        self._module = FinancialJSONModule(lm=self._lm)

        # ── Agent-specific DSPy modules ───────────────────────────────────────
        self._context_mod   = BuildContextModule(
            json_module=self._module,
            system_prompt=_SYSTEM,
            context_prompt_template=_CONTEXT_PROMPT,
        )
        self._insights_mod  = GenerateInsightsModule(
            json_module=self._module,
            system_prompt=_SYSTEM,
            insights_prompt_template=_INSIGHTS_PROMPT,
            retry_addendum_template=_RETRY_ADDENDUM,
        )
        self._narrative_mod = InsightNarrativeModule(
            json_module=self._module,
            system_prompt=_SYSTEM,
            narrative_prompt_template=_NARRATIVE_PROMPT,
        )

        # ── LangGraph: replaces _execute_graph() ─────────────────────────────
        self._graph = self._build_graph()

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
        """Exact same signature as the original insight_agent()."""
        t_start = time.perf_counter()

        try:
            validated = InsightInput(
                data=data, patterns=patterns,
                forecast=forecast, question=question,
            )
        except Exception as e:
            return InsightResult(
                success=False,
                errors=[{"node": "pre_validation", "error": str(e)}],
                model_used=self.model,
                duration_ms=round((time.perf_counter() - t_start) * 1000, 1),
            )

        initial_state: InsightAgentGraphState = {
            "data":            validated.data,
            "patterns":        validated.patterns,
            "forecast":        validated.forecast,
            "question":        validated.question,
            "max_retries":     self.max_retries,
            "input_valid":     False,
            "input_errors":    [],
            "context":         None,
            "insights":        None,
            "insights_valid":  False,
            "insights_errors": [],
            "retry_count":     0,
            "narrative":       "",
            "node_history":    [],
            "errors":          [],
            "abort":           False,
        }

        logger.info(f"🤖 [{self.agent_name}] Starting | model={self.model}")
        logger.info(f"   Q: {question[:80]}{'...' if len(question) > 80 else ''}")

        final_state = await self._graph.ainvoke(initial_state)
        duration_ms = round((time.perf_counter() - t_start) * 1000, 1)
        result      = self._build_result(final_state, duration_ms)

        if result.success:
            logger.success(
                f"✅ [{self.agent_name}] {duration_ms}ms | "
                f"health={result.health_score}/10 | insights={len(result.insights)}"
            )
        else:
            logger.error(f"❌ [{self.agent_name}] {duration_ms}ms | errors={result.errors}")

        return result

    # ──────────────────────────────────────────────────────────
    # LANGGRAPH GRAPH BUILDER
    # ──────────────────────────────────────────────────────────

    def _build_graph(self):
        """
        Compile the 6-node InsightAgent LangGraph StateGraph.

        Topology:
          validate_inputs → build_context → generate_insights
            → validate_insights
              → (invalid & retries left?) generate_insights  [retry cycle]
              → enrich_narrative → format_output → END
        """
        sg = StateGraph(InsightAgentGraphState)

        sg.add_node("validate_inputs",   self._node_validate_inputs)
        sg.add_node("build_context",     self._node_build_context)
        sg.add_node("generate_insights", self._node_generate_insights)
        sg.add_node("validate_insights", self._node_validate_insights)
        sg.add_node("enrich_narrative",  self._node_enrich_narrative)
        sg.add_node("format_output",     self._node_format_output)

        sg.set_entry_point("validate_inputs")

        sg.add_conditional_edges("validate_inputs",
            lambda s: END if s.get("abort") else "build_context")
        sg.add_conditional_edges("build_context",
            lambda s: END if s.get("abort") else "generate_insights")
        sg.add_conditional_edges("generate_insights",
            lambda s: END if s.get("abort") else "validate_insights")
        # Retry cycle: validate_insights → generate_insights
        sg.add_conditional_edges(
            "validate_insights",
            lambda s: (
                "generate_insights"
                if not s.get("insights_valid") and s.get("retry_count", 0) < s.get("max_retries", 2)
                else "enrich_narrative"
            ),
        )
        sg.add_edge("enrich_narrative", "format_output")
        sg.add_edge("format_output", END)

        return sg.compile()

    # ──────────────────────────────────────────────────────────
    # NODE 1 — VALIDATE INPUTS  (no LLM)
    # ──────────────────────────────────────────────────────────

    async def _node_validate_inputs(self, state: InsightAgentGraphState) -> dict:
        t = time.perf_counter()
        try:
            tmp = InsightAgentState(
                data=state["data"], patterns=state["patterns"],
                forecast=state["forecast"], question=state["question"],
            )
            tmp = validate_inputs(tmp)
            ms  = round((time.perf_counter() - t) * 1000, 1)
            if not tmp.input_valid:
                err_msg = "Input validation failed: " + " | ".join(tmp.input_errors)
                logger.error(f"  ❌ [validate_inputs] {err_msg}")
                return {
                    "input_valid":  False, "input_errors": tmp.input_errors,
                    "node_history": [{"node": "validate_inputs", "status": "failed", "ms": ms}],
                    "errors":       [{"node": "validate_inputs", "error": err_msg}],
                    "abort":        True,
                }
            logger.debug(
                f"    inputs valid | data={len(state['data'])}c "
                f"patterns={len(state['patterns'])}c "
                f"forecast={len(state['forecast'])}c ({ms}ms)"
            )
            return {
                "input_valid":  True, "input_errors": [],
                "node_history": [{"node": "validate_inputs", "status": "completed", "ms": ms}],
                "abort":        False,
            }
        except Exception as exc:
            ms  = round((time.perf_counter() - t) * 1000, 1)
            msg = f"{type(exc).__name__}: {exc}"
            return {
                "input_valid":  False,
                "node_history": [{"node": "validate_inputs", "status": "failed", "ms": ms, "error": msg}],
                "errors":       [{"node": "validate_inputs", "error": msg}],
                "abort":        True,
            }

    # ──────────────────────────────────────────────────────────
    # NODE 2 — BUILD CONTEXT  (DSPy LM call)
    # ──────────────────────────────────────────────────────────

    async def _node_build_context(self, state: InsightAgentGraphState) -> dict:
        """
        LLM node: extract synthesis context from three text sources.
        BEFORE: self._llm_json(_CONTEXT_PROMPT.format(...))
        AFTER:  self._context_mod.acall(data=..., patterns=..., forecast=...)
        """
        t = time.perf_counter()
        try:
            raw = await self._context_mod.acall(
                data=state["data"],
                patterns=state["patterns"],
                forecast=state["forecast"],
            )
            ctx = SynthesisContext.model_validate(raw)
            ms  = round((time.perf_counter() - t) * 1000, 1)
            logger.debug(
                f"    context: rev={ctx.latest_revenue} "
                f"margin={ctx.latest_profit_margin} "
                f"health={ctx.overall_health_signal} ({ms}ms)"
            )
            return {
                "context":      ctx.model_dump(),
                "node_history": [{"node": "build_context", "status": "completed", "ms": ms}],
                "abort":        False,
            }
        except Exception as exc:
            ms  = round((time.perf_counter() - t) * 1000, 1)
            msg = f"{type(exc).__name__}: {exc}"
            logger.error(f"  ❌ [build_context] {msg}")
            return {
                "node_history": [{"node": "build_context", "status": "failed", "ms": ms, "error": msg}],
                "errors":       [{"node": "build_context", "error": msg}],
                "abort":        True,
            }

    # ──────────────────────────────────────────────────────────
    # NODE 3 — GENERATE INSIGHTS  (DSPy LM call)
    # ──────────────────────────────────────────────────────────

    async def _node_generate_insights(self, state: InsightAgentGraphState) -> dict:
        """
        LLM node: synthesize strategic insights for CFO.
        On retry: passes insights_errors for self-correction.
        BEFORE: self._llm_json(_INSIGHTS_PROMPT.format(...))
        AFTER:  self._insights_mod.acall(...)
        """
        t = time.perf_counter()
        try:
            ctx_dict     = state.get("context") or {}
            ctx_obj      = SynthesisContext.model_validate(ctx_dict) if ctx_dict else None
            context_str  = ctx_obj.model_dump_json(indent=2) if ctx_obj else "{}"
            retry_errors = state.get("insights_errors") if state.get("retry_count", 0) > 0 else None

            raw = await self._insights_mod.acall(
                context=context_str,
                data=state["data"],
                patterns=state["patterns"],
                forecast=state["forecast"],
                question=state["question"],
                retry_errors=retry_errors,
            )
            ins = InsightOutput.model_validate(raw)
            ms  = round((time.perf_counter() - t) * 1000, 1)
            logger.debug(
                f"    insights: score={ins.health_score}/10 "
                f"insights={len(ins.insights)} actions={len(ins.actions)} ({ms}ms)"
            )
            return {
                "insights":     ins.model_dump(),
                "node_history": [{"node": "generate_insights", "status": "completed", "ms": ms}],
                "abort":        False,
            }
        except Exception as exc:
            ms  = round((time.perf_counter() - t) * 1000, 1)
            msg = f"{type(exc).__name__}: {exc}"
            logger.error(f"  ❌ [generate_insights] {msg}")
            return {
                "node_history": [{"node": "generate_insights", "status": "failed", "ms": ms, "error": msg}],
                "errors":       [{"node": "generate_insights", "error": msg}],
                "abort":        True,
            }

    # ──────────────────────────────────────────────────────────
    # NODE 4 — VALIDATE INSIGHTS  (no LLM, retry via graph edge)
    # ──────────────────────────────────────────────────────────

    async def _node_validate_insights(self, state: InsightAgentGraphState) -> dict:
        """
        Quality checks on insight output.
        Retry expressed as LangGraph conditional edge:
          validate_insights → (invalid & retries left) → generate_insights
        """
        t            = time.perf_counter()
        insights_raw = state.get("insights")
        insights_obj = InsightOutput.model_validate(insights_raw) if insights_raw else None

        tmp = InsightAgentState(
            data=state["data"], patterns=state["patterns"],
            forecast=state["forecast"], question=state["question"],
            insights=insights_obj,
            retry_count=state.get("retry_count", 0),
            max_retries=state.get("max_retries", 2),
        )
        tmp = validate_insights(tmp)
        ms  = round((time.perf_counter() - t) * 1000, 1)

        if tmp.insights_valid:
            logger.debug(f"    insights valid ✓ ({ms}ms)")
        else:
            logger.warning(
                f"    insights invalid (attempt {tmp.retry_count}/{tmp.max_retries}): "
                f"{tmp.insights_errors} ({ms}ms)"
            )
        return {
            "insights_valid":  tmp.insights_valid,
            "insights_errors": tmp.insights_errors,
            "retry_count":     tmp.retry_count,
            "node_history":    [{"node": "validate_insights", "status": "completed", "ms": ms}],
        }

    # ──────────────────────────────────────────────────────────
    # NODE 5 — ENRICH NARRATIVE  (DSPy LM call)
    # ──────────────────────────────────────────────────────────

    async def _node_enrich_narrative(self, state: InsightAgentGraphState) -> dict:
        """LLM node: write CFO-ready executive narrative."""
        t            = time.perf_counter()
        insights_raw = state.get("insights")
        if not insights_raw:
            return {
                "narrative":    "Insights could not be generated due to earlier errors.",
                "node_history": [{"node": "enrich_narrative", "status": "skipped", "ms": 0}],
            }
        try:
            ctx_dict     = state.get("context") or {}
            ctx_obj      = SynthesisContext.model_validate(ctx_dict) if ctx_dict else None
            insights_obj = InsightOutput.model_validate(insights_raw)

            raw = await self._narrative_mod.acall(
                insights_json=insights_obj.model_dump_json(indent=2),
                data_period=ctx_obj.data_period if ctx_obj else "unknown",
                latest_revenue=str(ctx_obj.latest_revenue) if ctx_obj else "unknown",
                health_score=insights_obj.health_score,
                health_trend=insights_obj.health_trend.value,
                question=state["question"],
            )
            narrative = raw.get("narrative", "").strip() or insights_obj.executive_summary
            ms        = round((time.perf_counter() - t) * 1000, 1)
            logger.debug(f"    narrative: {narrative[:80]}... ({ms}ms)")
            return {
                "narrative":    narrative,
                "node_history": [{"node": "enrich_narrative", "status": "completed", "ms": ms}],
            }
        except Exception as exc:
            ms  = round((time.perf_counter() - t) * 1000, 1)
            msg = f"{type(exc).__name__}: {exc}"
            logger.error(f"  ❌ [enrich_narrative] {msg}")
            return {
                "node_history": [{"node": "enrich_narrative", "status": "failed", "ms": ms, "error": msg}],
                "errors":       [{"node": "enrich_narrative", "error": msg}],
            }

    # ──────────────────────────────────────────────────────────
    # NODE 6 — FORMAT OUTPUT  (no LLM)
    # ──────────────────────────────────────────────────────────

    async def _node_format_output(self, state: InsightAgentGraphState) -> dict:
        t  = time.perf_counter()
        ms = round((time.perf_counter() - t) * 1000, 1)
        return {"node_history": [{"node": "format_output", "status": "completed", "ms": ms}]}

    # ──────────────────────────────────────────────────────────
    # RESULT BUILDER
    # ──────────────────────────────────────────────────────────

    def _build_result(self, state: InsightAgentGraphState, duration_ms: float) -> InsightResult:
        """Convert final LangGraph state dict into a typed InsightResult."""
        insights_raw = state.get("insights")
        insights_obj = InsightOutput.model_validate(insights_raw) if insights_raw else None
        ctx_raw      = state.get("context")
        ctx_obj      = SynthesisContext.model_validate(ctx_raw) if ctx_raw else None

        success = (
            state.get("input_valid", False)
            and state.get("insights_valid", False)
            and insights_obj is not None
            and len(state.get("errors", [])) == 0
        )

        base = InsightResult(
            success=success,
            model_used=self.model,
            duration_ms=duration_ms,
            retry_count=state.get("retry_count", 0),
            node_history=state.get("node_history", []),
            errors=state.get("errors", []),
            narrative=state.get("narrative", ""),
        )

        if ctx_obj:
            base.latest_revenue       = ctx_obj.latest_revenue
            base.data_period          = ctx_obj.data_period
            base.forecast_growth_rate = ctx_obj.forecast_growth_rate
            if ctx_obj.forecast_growth_rate is not None:
                base.forecast_growth_pct = f"{ctx_obj.forecast_growth_rate * 100:.1f}%"

        if insights_obj:
            ins = insights_obj
            base.direct_answer      = ins.direct_answer
            base.key_risk           = ins.key_risk
            base.health_score       = ins.health_score
            base.health_explanation = ins.health_explanation
            base.health_trend       = ins.health_trend.value
            base.executive_summary  = ins.executive_summary
            base.insights = [
                {"title": i.title, "explanation": i.explanation,
                 "evidence": i.evidence, "urgency": i.urgency.value}
                for i in ins.insights
            ]
            base.actions = [
                {"action": a.action, "rationale": a.rationale,
                 "expected_impact": a.expected_impact, "urgency": a.urgency.value}
                for a in ins.actions
            ]

        return base
