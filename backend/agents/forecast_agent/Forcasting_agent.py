from __future__ import annotations

import os
import time
from typing import Optional, Annotated

from loguru import logger
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from operator import add

from .dspy_modules import (
    ParseSnapshotModule,
    GenerateForecastModule,
    ForecastNarrativeModule,
)
from .models import (
    FinancialSnapshot,
    ForecastAgentState,
    ForecastInput,
    ForecastOutput,
    ForecastResult,
)
from .validators import validate_inputs, validate_forecast
from ..deep_agent import make_dspy_lm, FinancialJSONModule


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
# LANGGRAPH STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ForecastAgentGraphState(TypedDict):
    """
    LangGraph state for the 6-node ForecastAgent graph.
    Mirrors ForecastAgentState (Pydantic) in a LangGraph-compatible TypedDict.
    """
    # Inputs
    data:             str
    patterns:         str
    question:         str
    max_retries:      int
    # Node 1
    input_valid:      bool
    input_errors:     list[str]
    # Node 2 — FinancialSnapshot as dict
    snapshot:         Optional[dict]
    # Node 3 / 4 — ForecastOutput as dict
    forecast:         Optional[dict]
    forecast_valid:   bool
    forecast_errors:  list[str]
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

class ForecastAgent:
    """
    Production ForecastAgent — refactored to DeepAgent + DSPy + LangGraph.

    ARCHITECTURE CHANGES:
      GeminiAdapter    → FinancialJSONModule(dspy.LM)   [DSPy]
      _execute_graph() → LangGraph StateGraph           [LangGraph]
      retry inline     → graph cycle via conditional edge

    PUBLIC INTERFACE — unchanged:
      agent  = ForecastAgent()
      result = await agent.run(data, patterns, question)
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
                "Set it in .env or pass api_key= to ForecastAgent()."
            )

        self.model       = model
        self.max_retries = max_retries
        self.agent_name  = "ForecastAgent"

        # ── DSPy: replaces GeminiAdapter (max_tokens=3000 preserved) ─────────
        self._lm     = make_dspy_lm(
            model=model, api_key=resolved_key,
            temperature=temperature, max_tokens=3000,
        )
        self._module = FinancialJSONModule(lm=self._lm)

        # ── Agent-specific DSPy modules ───────────────────────────────────────
        self._parse_mod     = ParseSnapshotModule(
            json_module=self._module,
            system_prompt=_SYSTEM,
            parse_prompt_template=_PARSE_PROMPT,
        )
        self._forecast_mod  = GenerateForecastModule(
            json_module=self._module,
            system_prompt=_SYSTEM,
            forecast_prompt_template=_FORECAST_PROMPT,
            retry_addendum_template=_RETRY_ADDENDUM,
        )
        self._narrative_mod = ForecastNarrativeModule(
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
        question: str,
    ) -> ForecastResult:
        """Exact same signature as the original forecast_agent()."""
        t_start = time.perf_counter()

        try:
            validated = ForecastInput(data=data, patterns=patterns, question=question)
        except Exception as e:
            return ForecastResult(
                success=False,
                errors=[{"node": "pre_validation", "error": str(e)}],
                model_used=self.model,
                duration_ms=round((time.perf_counter() - t_start) * 1000, 1),
            )

        initial_state: ForecastAgentGraphState = {
            "data":            validated.data,
            "patterns":        validated.patterns,
            "question":        validated.question,
            "max_retries":     self.max_retries,
            "input_valid":     False,
            "input_errors":    [],
            "snapshot":        None,
            "forecast":        None,
            "forecast_valid":  False,
            "forecast_errors": [],
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
                f"confidence={result.confidence} | growth={result.growth_rate_pct}"
            )
        else:
            logger.error(f"❌ [{self.agent_name}] {duration_ms}ms | errors={result.errors}")

        return result

    # ──────────────────────────────────────────────────────────
    # LANGGRAPH GRAPH BUILDER
    # ──────────────────────────────────────────────────────────

    def _build_graph(self):
        """
        Compile the 6-node ForecastAgent LangGraph StateGraph.

        Topology:
          validate_inputs → parse_financials → generate_forecast
            → validate_forecast
              → (invalid & retries left?) generate_forecast  [retry cycle]
              → enrich_narrative → format_output → END
        """
        sg = StateGraph(ForecastAgentGraphState)

        sg.add_node("validate_inputs",   self._node_validate_inputs)
        sg.add_node("parse_financials",  self._node_parse_financials)
        sg.add_node("generate_forecast", self._node_generate_forecast)
        sg.add_node("validate_forecast", self._node_validate_forecast)
        sg.add_node("enrich_narrative",  self._node_enrich_narrative)
        sg.add_node("format_output",     self._node_format_output)

        sg.set_entry_point("validate_inputs")

        sg.add_conditional_edges("validate_inputs",
            lambda s: END if s.get("abort") else "parse_financials")
        sg.add_conditional_edges("parse_financials",
            lambda s: END if s.get("abort") else "generate_forecast")
        sg.add_conditional_edges("generate_forecast",
            lambda s: END if s.get("abort") else "validate_forecast")
        # Retry cycle: validate_forecast → generate_forecast
        sg.add_conditional_edges(
            "validate_forecast",
            lambda s: (
                "generate_forecast"
                if not s.get("forecast_valid") and s.get("retry_count", 0) < s.get("max_retries", 2)
                else "enrich_narrative"
            ),
        )
        sg.add_edge("enrich_narrative", "format_output")
        sg.add_edge("format_output", END)

        return sg.compile()

    # ──────────────────────────────────────────────────────────
    # NODE 1 — VALIDATE INPUTS  (no LLM)
    # ──────────────────────────────────────────────────────────

    async def _node_validate_inputs(self, state: ForecastAgentGraphState) -> dict:
        t = time.perf_counter()
        try:
            tmp = ForecastAgentState(
                data=state["data"], patterns=state["patterns"], question=state["question"],
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
            logger.debug(f"    inputs valid | data={len(state['data'])}c ({ms}ms)")
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
    # NODE 2 — PARSE FINANCIALS  (DSPy LM call)
    # ──────────────────────────────────────────────────────────

    async def _node_parse_financials(self, state: ForecastAgentGraphState) -> dict:
        """
        LLM node: extract structured snapshot from raw data.
        BEFORE: self._llm_json(_PARSE_PROMPT.format(...))
        AFTER:  self._parse_mod.acall(data=...)
        """
        t = time.perf_counter()
        try:
            raw  = await self._parse_mod.acall(data=state["data"])
            snap = FinancialSnapshot.model_validate(raw)
            ms   = round((time.perf_counter() - t) * 1000, 1)
            logger.debug(
                f"    snapshot: rev={snap.latest_revenue} growth={snap.avg_growth_rate} "
                f"trend={snap.trend_direction} periods={snap.num_periods} ({ms}ms)"
            )
            return {
                "snapshot":     snap.model_dump(),
                "node_history": [{"node": "parse_financials", "status": "completed", "ms": ms}],
                "abort":        False,
            }
        except Exception as exc:
            ms  = round((time.perf_counter() - t) * 1000, 1)
            msg = f"{type(exc).__name__}: {exc}"
            logger.error(f"  ❌ [parse_financials] {msg}")
            return {
                "node_history": [{"node": "parse_financials", "status": "failed", "ms": ms, "error": msg}],
                "errors":       [{"node": "parse_financials", "error": msg}],
                "abort":        True,
            }

    # ──────────────────────────────────────────────────────────
    # NODE 3 — GENERATE FORECAST  (DSPy LM call)
    # ──────────────────────────────────────────────────────────

    async def _node_generate_forecast(self, state: ForecastAgentGraphState) -> dict:
        """
        LLM node: produce structured financial forecast.
        On retry: passes forecast_errors as context for self-correction.
        BEFORE: self._llm_json(_FORECAST_PROMPT.format(...))
        AFTER:  self._forecast_mod.acall(...)
        """
        t = time.perf_counter()
        try:
            snap_dict    = state.get("snapshot") or {}
            snap_obj     = FinancialSnapshot.model_validate(snap_dict) if snap_dict else None
            snapshot_str = snap_obj.model_dump_json(indent=2) if snap_obj else "No snapshot available."
            retry_errors = state.get("forecast_errors") if state.get("retry_count", 0) > 0 else None

            raw = await self._forecast_mod.acall(
                snapshot=snapshot_str,
                patterns=state["patterns"],
                question=state["question"],
                retry_errors=retry_errors,
            )
            fc  = ForecastOutput.model_validate(raw)
            ms  = round((time.perf_counter() - t) * 1000, 1)
            logger.debug(
                f"    forecast: {fc.short_term.period} rev={fc.short_term.revenue} "
                f"growth={fc.growth_rate:.1%} conf={fc.confidence} ({ms}ms)"
            )
            return {
                "forecast":     fc.model_dump(),
                "node_history": [{"node": "generate_forecast", "status": "completed", "ms": ms}],
                "abort":        False,
            }
        except Exception as exc:
            ms  = round((time.perf_counter() - t) * 1000, 1)
            msg = f"{type(exc).__name__}: {exc}"
            logger.error(f"  ❌ [generate_forecast] {msg}")
            return {
                "node_history": [{"node": "generate_forecast", "status": "failed", "ms": ms, "error": msg}],
                "errors":       [{"node": "generate_forecast", "error": msg}],
                "abort":        True,
            }

    # ──────────────────────────────────────────────────────────
    # NODE 4 — VALIDATE FORECAST  (no LLM, retry via graph edge)
    # ──────────────────────────────────────────────────────────

    async def _node_validate_forecast(self, state: ForecastAgentGraphState) -> dict:
        """
        8 sanity checks on the forecast numbers.
        Retry expressed as LangGraph conditional edge:
          validate_forecast → (invalid & retries left) → generate_forecast
        """
        t            = time.perf_counter()
        forecast_raw = state.get("forecast")
        forecast_obj = ForecastOutput.model_validate(forecast_raw) if forecast_raw else None

        tmp = ForecastAgentState(
            data=state["data"], patterns=state["patterns"], question=state["question"],
            forecast=forecast_obj,
            retry_count=state.get("retry_count", 0),
            max_retries=state.get("max_retries", 2),
        )
        tmp = validate_forecast(tmp)
        ms  = round((time.perf_counter() - t) * 1000, 1)

        if tmp.forecast_valid:
            logger.debug(f"    forecast valid ✓ ({ms}ms)")
        else:
            logger.warning(
                f"    forecast invalid (attempt {tmp.retry_count}/{tmp.max_retries}): "
                f"{tmp.forecast_errors} ({ms}ms)"
            )
        return {
            "forecast_valid":  tmp.forecast_valid,
            "forecast_errors": tmp.forecast_errors,
            "retry_count":     tmp.retry_count,
            "node_history":    [{"node": "validate_forecast", "status": "completed", "ms": ms}],
        }

    # ──────────────────────────────────────────────────────────
    # NODE 5 — ENRICH NARRATIVE  (DSPy LM call)
    # ──────────────────────────────────────────────────────────

    async def _node_enrich_narrative(self, state: ForecastAgentGraphState) -> dict:
        """LLM node: write professional forecast paragraph."""
        t            = time.perf_counter()
        forecast_raw = state.get("forecast")
        if not forecast_raw:
            return {
                "narrative":    "Forecast could not be generated due to earlier errors.",
                "node_history": [{"node": "enrich_narrative", "status": "skipped", "ms": 0}],
            }
        try:
            snap_dict    = state.get("snapshot") or {}
            snap_obj     = FinancialSnapshot.model_validate(snap_dict) if snap_dict else None
            forecast_obj = ForecastOutput.model_validate(forecast_raw)

            raw = await self._narrative_mod.acall(
                forecast_json=forecast_obj.model_dump_json(indent=2),
                data_period=snap_obj.data_period if snap_obj else "unknown",
                latest_revenue=str(snap_obj.latest_revenue) if snap_obj else "unknown",
                trend_direction=str(snap_obj.trend_direction) if snap_obj else "unknown",
                patterns=state["patterns"][:500],
                question=state["question"],
            )
            narrative = raw.get("narrative", "").strip() or "Narrative generation returned empty response."
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

    async def _node_format_output(self, state: ForecastAgentGraphState) -> dict:
        t  = time.perf_counter()
        ms = round((time.perf_counter() - t) * 1000, 1)
        return {"node_history": [{"node": "format_output", "status": "completed", "ms": ms}]}

    # ──────────────────────────────────────────────────────────
    # RESULT BUILDER
    # ──────────────────────────────────────────────────────────

    def _build_result(self, state: ForecastAgentGraphState, duration_ms: float) -> ForecastResult:
        """Convert final LangGraph state dict into a typed ForecastResult."""
        forecast_raw = state.get("forecast")
        forecast_obj = ForecastOutput.model_validate(forecast_raw) if forecast_raw else None
        snap_raw     = state.get("snapshot")
        snap_obj     = FinancialSnapshot.model_validate(snap_raw) if snap_raw else None

        success = (
            state.get("input_valid", False)
            and state.get("forecast_valid", False)
            and forecast_obj is not None
            and len(state.get("errors", [])) == 0
        )

        base = ForecastResult(
            success=success,
            model_used=self.model,
            duration_ms=duration_ms,
            retry_count=state.get("retry_count", 0),
            node_history=state.get("node_history", []),
            errors=state.get("errors", []),
            narrative=state.get("narrative", ""),
        )

        if snap_obj:
            base.data_period     = snap_obj.data_period
            base.latest_revenue  = snap_obj.latest_revenue
            base.trend_direction = snap_obj.trend_direction.value

        if forecast_obj:
            fc = forecast_obj
            base.short_term_period    = fc.short_term.period
            base.short_term_revenue   = fc.short_term.revenue
            base.short_term_expenses  = fc.short_term.expenses
            base.short_term_profit    = fc.short_term.net_profit
            base.short_term_reasoning = fc.short_term.reasoning
            base.annual_year          = fc.annual.year
            base.annual_revenue       = fc.annual.revenue
            base.annual_expenses      = fc.annual.expenses
            base.annual_profit        = fc.annual.net_profit
            base.annual_margin        = fc.annual.profit_margin
            base.annual_reasoning     = fc.annual.reasoning
            base.growth_rate          = fc.growth_rate
            base.growth_rate_pct      = f"{fc.growth_rate * 100:.1f}%"
            base.growth_reasoning     = fc.growth_reasoning
            base.risks                = fc.risks
            base.confidence           = fc.confidence.value
            base.confidence_explanation = fc.confidence_explanation
            base.assumptions          = fc.assumptions

        return base