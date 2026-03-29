from __future__ import annotations

import os
import time
from typing import Optional, Annotated

from loguru import logger
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from operator import add

from .dspy_modules import (
    ParseFinancialsModule,
    DetectPatternsModule,
    PatternNarrativeModule,
)
from .models import (
    ParsedFinancials,
    PatternAgentState,
    PatternInput,
    PatternOutput,
    PatternResult,
)
from .validators import validate_inputs, validate_patterns
from ..deep_agent import make_dspy_lm, FinancialJSONModule


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
# LANGGRAPH STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PatternAgentGraphState(TypedDict):
    """
    LangGraph state for the 6-node PatternAgent graph.
    node_history and errors use Annotated[list, add] so each node
    appends its entry without replacing the accumulated list.
    """
    # Inputs
    data:            str
    question:        str
    max_retries:     int
    # Node 1
    input_valid:     bool
    input_errors:    list[str]
    # Node 2 — stored as dict for JSON-serializability
    parsed:          Optional[dict]
    # Node 3 / 4
    patterns:        Optional[dict]
    patterns_valid:  bool
    patterns_errors: list[str]
    retry_count:     int
    # Node 5
    narrative:       str
    # Graph control
    node_history:    Annotated[list[dict], add]
    errors:          Annotated[list[dict], add]
    abort:           bool


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PatternAgent:
    """
    Production PatternAgent — refactored to DeepAgent + DSPy + LangGraph.

    ARCHITECTURE CHANGES:
      GeminiAdapter    → FinancialJSONModule(dspy.LM)   [DSPy]
      _execute_graph() → LangGraph StateGraph           [LangGraph]
      retry inline     → graph cycle via conditional edge

    PUBLIC INTERFACE — unchanged (drop-in replacement):
      agent  = PatternAgent()
      result = await agent.run(data, question)
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

        # ── DSPy: replaces GeminiAdapter ─────────────────────────────────────
        # max_tokens=3000 preserves the original max_output_tokens=3000
        self._lm     = make_dspy_lm(
            model=model, api_key=resolved_key,
            temperature=temperature, max_tokens=3000,
        )
        self._module = FinancialJSONModule(lm=self._lm)

        # ── Agent-specific DSPy modules (one per LLM node) ───────────────────
        self._parse_mod     = ParseFinancialsModule(
            json_module=self._module,
            system_prompt=_SYSTEM,
            parse_prompt_template=_PARSE_PROMPT,
        )
        self._detect_mod    = DetectPatternsModule(
            json_module=self._module,
            system_prompt=_SYSTEM,
            detect_prompt_template=_DETECT_PROMPT,
            retry_addendum_template=_RETRY_ADDENDUM,
        )
        self._narrative_mod = PatternNarrativeModule(
            json_module=self._module,
            system_prompt=_SYSTEM,
            narrative_prompt_template=_NARRATIVE_PROMPT,
        )

        # ── LangGraph: replaces _execute_graph() ─────────────────────────────
        self._graph = self._build_graph()

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

        initial_state: PatternAgentGraphState = {
            "data":            validated.data,
            "question":        validated.question,
            "max_retries":     self.max_retries,
            "input_valid":     False,
            "input_errors":    [],
            "parsed":          None,
            "patterns":        None,
            "patterns_valid":  False,
            "patterns_errors": [],
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
                f"trend={result.trend_direction} | "
                f"patterns={len(result.patterns)} | "
                f"anomalies={len(result.anomalies)}"
            )
        else:
            logger.error(f"❌ [{self.agent_name}] {duration_ms}ms | errors={result.errors}")

        return result

    # ──────────────────────────────────────────────────────────
    # LANGGRAPH GRAPH BUILDER
    # ──────────────────────────────────────────────────────────

    def _build_graph(self):
        """
        Compile the 6-node PatternAgent LangGraph StateGraph.

        Graph topology:
          validate_inputs → (abort?) END
            → parse_financials → (abort?) END
              → detect_patterns → (abort?) END
                → validate_patterns
                  → (invalid & retries left?) detect_patterns  [retry cycle]
                  → enrich_narrative → format_output → END
        """
        sg = StateGraph(PatternAgentGraphState)

        sg.add_node("validate_inputs",   self._node_validate_inputs)
        sg.add_node("parse_financials",  self._node_parse_financials)
        sg.add_node("detect_patterns",   self._node_detect_patterns)
        sg.add_node("validate_patterns", self._node_validate_patterns)
        sg.add_node("enrich_narrative",  self._node_enrich_narrative)
        sg.add_node("format_output",     self._node_format_output)

        sg.set_entry_point("validate_inputs")

        sg.add_conditional_edges(
            "validate_inputs",
            lambda s: END if s.get("abort") else "parse_financials",
        )
        sg.add_conditional_edges(
            "parse_financials",
            lambda s: END if s.get("abort") else "detect_patterns",
        )
        sg.add_conditional_edges(
            "detect_patterns",
            lambda s: END if s.get("abort") else "validate_patterns",
        )
        # Retry cycle: validate_patterns → detect_patterns
        sg.add_conditional_edges(
            "validate_patterns",
            lambda s: (
                "detect_patterns"
                if not s.get("patterns_valid") and s.get("retry_count", 0) < s.get("max_retries", 2)
                else "enrich_narrative"
            ),
        )
        sg.add_edge("enrich_narrative", "format_output")
        sg.add_edge("format_output", END)

        return sg.compile()

    # ──────────────────────────────────────────────────────────
    # NODE 1 — VALIDATE INPUTS  (no LLM)
    # ──────────────────────────────────────────────────────────

    async def _node_validate_inputs(self, state: PatternAgentGraphState) -> dict:
        t = time.perf_counter()
        try:
            tmp = PatternAgentState(
                data=state["data"], question=state["question"],
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

    async def _node_parse_financials(self, state: PatternAgentGraphState) -> dict:
        """
        LLM node: extract structured time-series from raw data.
        BEFORE: self._llm_json(_PARSE_PROMPT.format(...))
        AFTER:  self._parse_mod.acall(data=...)
        """
        t = time.perf_counter()
        try:
            raw    = await self._parse_mod.acall(data=state["data"])
            parsed = ParsedFinancials.model_validate(raw)
            ms     = round((time.perf_counter() - t) * 1000, 1)
            logger.debug(f"    parsed: {parsed.num_periods} periods | {parsed.data_period} ({ms}ms)")
            return {
                "parsed":       parsed.model_dump(),
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
    # NODE 3 — DETECT PATTERNS  (DSPy LM call)
    # ──────────────────────────────────────────────────────────

    async def _node_detect_patterns(self, state: PatternAgentGraphState) -> dict:
        """
        LLM node: detect patterns, anomalies, and trends.
        On retry: passes patterns_errors as context for self-correction.
        """
        t = time.perf_counter()
        try:
            parsed_dict  = state.get("parsed") or {}
            retry_errors = state.get("patterns_errors") if state.get("retry_count", 0) > 0 else None
            # Reconstruct Pydantic object to serialise properly
            parsed_obj   = ParsedFinancials.model_validate(parsed_dict) if parsed_dict else None
            parsed_str   = parsed_obj.model_dump_json(indent=2) if parsed_obj else "{}"

            raw  = await self._detect_mod.acall(
                parsed=parsed_str,
                question=state["question"],
                retry_errors=retry_errors,
            )
            out  = PatternOutput.model_validate(raw)
            ms   = round((time.perf_counter() - t) * 1000, 1)
            logger.debug(
                f"    detected: {len(out.patterns)} patterns | "
                f"{len(out.anomalies)} anomalies | trend={out.trend_direction} ({ms}ms)"
            )
            return {
                "patterns":     out.model_dump(),
                "node_history": [{"node": "detect_patterns", "status": "completed", "ms": ms}],
                "abort":        False,
            }
        except Exception as exc:
            ms  = round((time.perf_counter() - t) * 1000, 1)
            msg = f"{type(exc).__name__}: {exc}"
            logger.error(f"  ❌ [detect_patterns] {msg}")
            return {
                "node_history": [{"node": "detect_patterns", "status": "failed", "ms": ms, "error": msg}],
                "errors":       [{"node": "detect_patterns", "error": msg}],
                "abort":        True,
            }

    # ──────────────────────────────────────────────────────────
    # NODE 4 — VALIDATE PATTERNS  (no LLM, retry via graph edge)
    # ──────────────────────────────────────────────────────────

    async def _node_validate_patterns(self, state: PatternAgentGraphState) -> dict:
        """
        Pure Python quality checks on the pattern output.
        Retry logic expressed as a LangGraph conditional edge:
          validate_patterns → (invalid & retries left) → detect_patterns  [cycle]
        """
        t = time.perf_counter()
        patterns_raw = state.get("patterns")
        patterns_obj = PatternOutput.model_validate(patterns_raw) if patterns_raw else None

        tmp = PatternAgentState(
            data=state["data"], question=state["question"],
            patterns=patterns_obj,
            retry_count=state.get("retry_count", 0),
            max_retries=state.get("max_retries", 2),
        )
        tmp = validate_patterns(tmp)
        ms  = round((time.perf_counter() - t) * 1000, 1)

        if tmp.patterns_valid:
            logger.debug(f"    patterns valid ✓ ({ms}ms)")
        else:
            logger.warning(
                f"    patterns invalid (attempt {tmp.retry_count}/{tmp.max_retries}): "
                f"{tmp.patterns_errors} ({ms}ms)"
            )
        return {
            "patterns_valid":  tmp.patterns_valid,
            "patterns_errors": tmp.patterns_errors,
            "retry_count":     tmp.retry_count,
            "node_history":    [{"node": "validate_patterns", "status": "completed", "ms": ms}],
        }

    # ──────────────────────────────────────────────────────────
    # NODE 5 — ENRICH NARRATIVE  (DSPy LM call)
    # ──────────────────────────────────────────────────────────

    async def _node_enrich_narrative(self, state: PatternAgentGraphState) -> dict:
        """LLM node: write analyst-quality summary paragraph."""
        t = time.perf_counter()
        patterns_raw = state.get("patterns")
        if not patterns_raw:
            return {
                "narrative":    "Pattern detection could not be completed due to earlier errors.",
                "node_history": [{"node": "enrich_narrative", "status": "skipped", "ms": 0}],
            }
        try:
            parsed_dict  = state.get("parsed") or {}
            parsed_obj   = ParsedFinancials.model_validate(parsed_dict) if parsed_dict else None
            patterns_obj = PatternOutput.model_validate(patterns_raw)

            raw = await self._narrative_mod.acall(
                patterns_json=patterns_obj.model_dump_json(indent=2),
                data_period=parsed_obj.data_period if parsed_obj else "unknown",
                num_periods=parsed_obj.num_periods if parsed_obj else 0,
                question=state["question"],
            )
            narrative = raw.get("narrative", "").strip() or patterns_obj.key_finding
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

    async def _node_format_output(self, state: PatternAgentGraphState) -> dict:
        t  = time.perf_counter()
        ms = round((time.perf_counter() - t) * 1000, 1)
        return {"node_history": [{"node": "format_output", "status": "completed", "ms": ms}]}

    # ──────────────────────────────────────────────────────────
    # RESULT BUILDER
    # ──────────────────────────────────────────────────────────

    def _build_result(self, state: PatternAgentGraphState, duration_ms: float) -> PatternResult:
        """Convert final LangGraph state dict into a typed PatternResult."""
        patterns_raw   = state.get("patterns")
        patterns_obj   = PatternOutput.model_validate(patterns_raw) if patterns_raw else None
        parsed_raw     = state.get("parsed")
        parsed_obj     = ParsedFinancials.model_validate(parsed_raw) if parsed_raw else None

        success = (
            state.get("input_valid", False)
            and state.get("patterns_valid", False)
            and patterns_obj is not None
            and len(state.get("errors", [])) == 0
        )

        base = PatternResult(
            success=success,
            model_used=self.model,
            duration_ms=duration_ms,
            retry_count=state.get("retry_count", 0),
            node_history=state.get("node_history", []),
            errors=state.get("errors", []),
            narrative=state.get("narrative", ""),
        )

        if parsed_obj:
            base.data_period = parsed_obj.data_period
            base.num_periods = parsed_obj.num_periods

        if patterns_obj:
            out = patterns_obj
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
                {"name": p.name, "description": p.description, "evidence": p.evidence,
                 "severity": p.severity.value, "periods_affected": p.periods_affected}
                for p in out.patterns
            ]
            base.anomalies = [
                {"description": a.description, "period": a.period, "magnitude": a.magnitude,
                 "severity": a.severity.value, "possible_cause": a.possible_cause}
                for a in out.anomalies
            ]
            base.yoy_comparisons = [
                {"metric": y.metric, "prior_value": y.prior_value,
                 "current_value": y.current_value, "change_pct": y.change_pct,
                 "direction": y.direction.value}
                for y in out.yoy_comparisons
            ]
        return base
