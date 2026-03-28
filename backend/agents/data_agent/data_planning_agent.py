from __future__ import annotations

import json
import os
import time
from typing import Callable, Optional

from loguru import logger

from .gemini_adapter import GeminiAdapter
from .models import (
    AgentInput,
    DataAgentResult,
    DataAgentState,
    LLMDataPlanResponse,
    LLMIntentResponse,
)
from .validators import validate_input, validate_plan


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROMPTS  (identical to OpenAI version)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SYSTEM = """You are a financial data planning specialist for Finora AI.
You analyze financial questions and produce precise data retrieval plans.
You only answer about Salesforce financial data (2011–2026).
Return ONLY valid JSON — no markdown, no explanation outside JSON.
"""

_INTENT_PROMPT = """
Analyze this financial question and return a JSON intent classification.

Question: "{question}"

Available database:
{db_summary}

Return EXACTLY this JSON:
{{
  "intent": "trend|comparison|snapshot|forecast|anomaly|ranking|summary|unknown",
  "confidence": 0.95,
  "time_period": "natural language description e.g. '2023' or 'Q1-Q2 2024' or '2020-2024'",
  "year_start": 2020,
  "year_end": 2024,
  "quarter": "Q1 or null",
  "granularity": "annual|quarterly|monthly",
  "key_metrics": ["revenue", "net_profit"],
  "reasoning": "one sentence why you chose this intent"
}}

Rules:
- year_start and year_end must be integers between 2011 and 2026
- If no specific year is mentioned, use the full range: year_start=2011, year_end=2026
- key_metrics must only reference: revenue, expenses, net_profit, profit_margin, growth_rate, amount
- intent "unknown" only if the question is completely unrelated to finance
"""

_DATA_PLAN_PROMPT = """
Build a precise data retrieval plan for this financial question.

Question: "{question}"
Detected intent: {intent}
Time period: {time_period}
Key metrics: {key_metrics}
Year range: {year_start} to {year_end}

Available tables and columns:
- revenue:     year, quarter, amount, region, product_line
- expenses:    year, quarter, amount, category, department
- kpis:        year, revenue, expenses, net_profit, profit_margin, growth_rate
- ai_insights: year, quarter, insight_text, category, confidence_score

Return EXACTLY this JSON:
{{
  "tables_needed": ["kpis", "revenue"],
  "columns_needed": {{
    "kpis": ["year", "net_profit", "profit_margin"],
    "revenue": ["year", "quarter", "amount"]
  }},
  "filters": {{
    "year": "BETWEEN 2020 AND 2024",
    "quarter": "Q1"
  }},
  "aggregations": ["SUM(revenue.amount)", "AVG(kpis.profit_margin)"],
  "order_by": "year ASC",
  "limit": null,
  "explanation": "one sentence describing what data is being fetched and why"
}}

Rules:
- Only include tables actually needed to answer the question
- columns_needed must contain ONLY columns that exist in the table
- filters values are SQL fragments (e.g. "BETWEEN 2020 AND 2024", "= 'Q1'")
- aggregations can be empty list [] if raw rows are sufficient
- Return ONLY valid JSON
"""

_RETRY_ADDENDUM = """

IMPORTANT — Your previous attempt had these validation errors:
{errors}

Fix these errors in your response. The corrected JSON must pass all validation checks.
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DataAgent:
    """
    Production DeepAgent for financial data planning — Gemini edition.

    Drop-in replacement for the original data_agent() function.

    BEFORE:
        def data_agent(question: str, db_summary: str) -> str:
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            return response.text

    AFTER:
        agent  = DataAgent(model="gemini-2.5-flash")
        result = await agent.run(question, db_summary)
        # result.tables_needed, result.filters, result.intent ...
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        max_retries: int = 2,
        temperature: float = 0.0,
    ):
        """
        Args:
            model:       Gemini model name. Default: "gemini-2.5-flash"
            api_key:     GEMINI_API_KEY. Falls back to env var if not provided.
            max_retries: How many times to retry a failed data plan (default 2).
            temperature: LLM temperature. 0.0 = deterministic (recommended).
        """
        resolved_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Set it in .env or pass api_key= to DataAgent()."
            )

        self.model      = model
        self.max_retries = max_retries
        self.agent_name = "DataAgent"

        # GeminiAdapter is the ONLY place that knows about the Gemini SDK
        self._gemini = GeminiAdapter(
            api_key=resolved_key,
            model_name=model,
            temperature=temperature,
        )

    # ──────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ──────────────────────────────────────────────────────────

    async def run(self, question: str, db_summary: str) -> DataAgentResult:
        """
        Process a financial question through the full 5-node graph.

        Args:
            question:   Natural language financial question.
            db_summary: Current state of the database (tables, row counts, ranges).

        Returns:
            DataAgentResult — typed, validated, with full audit trail.
        """
        t_start = time.perf_counter()

        # Pre-validate inputs with Pydantic before touching state
        try:
            validated = AgentInput(question=question, db_summary=db_summary)
        except Exception as e:
            return DataAgentResult(
                success=False,
                errors=[{"node": "pre_validation", "error": str(e)}],
                model_used=self.model,
                duration_ms=round((time.perf_counter() - t_start) * 1000, 1),
            )

        state = DataAgentState(
            question=validated.question,
            db_summary=validated.db_summary,
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
                f"intent={result.intent} | tables={result.tables_needed}"
            )
        else:
            logger.error(
                f"❌ [{self.agent_name}] {duration_ms}ms | errors={result.errors}"
            )

        return result

    # ──────────────────────────────────────────────────────────
    # GRAPH EXECUTION ENGINE
    # ──────────────────────────────────────────────────────────

    async def _execute_graph(self, state: DataAgentState) -> DataAgentState:
        graph: list[tuple[str, Callable, bool]] = [
            ("validate_input",  self._node_validate_input,  True),
            ("classify_intent", self._node_classify_intent, True),
            ("build_data_plan", self._node_build_data_plan, True),
            ("validate_plan",   self._node_validate_plan,   False),
            ("format_output",   self._node_format_output,   False),
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
    # NODE 1 — VALIDATE INPUT  (no LLM)
    # ──────────────────────────────────────────────────────────

    async def _node_validate_input(self, state: DataAgentState) -> DataAgentState:
        state = validate_input(state)
        if not state.input_valid:
            raise ValueError(
                "Input validation failed: " + " | ".join(state.input_errors)
            )
        logger.debug(f"    input valid: {state.clean_question[:60]}...")
        return state

    # ──────────────────────────────────────────────────────────
    # NODE 2 — CLASSIFY INTENT  (Gemini call)
    # ──────────────────────────────────────────────────────────

    async def _node_classify_intent(self, state: DataAgentState) -> DataAgentState:
        prompt = _INTENT_PROMPT.format(
            question=state.clean_question,
            db_summary=state.db_summary,
        )
        raw = await self._llm_json(prompt)

        try:
            state.intent = LLMIntentResponse.model_validate(raw)
        except Exception as e:
            raise ValueError(f"Intent response failed validation: {e} | raw={raw}")

        logger.debug(
            f"    intent={state.intent.intent} "
            f"conf={state.intent.confidence} "
            f"period={state.intent.time_period}"
        )
        return state

    # ──────────────────────────────────────────────────────────
    # NODE 3 — BUILD DATA PLAN  (Gemini call)
    # ──────────────────────────────────────────────────────────

    async def _node_build_data_plan(self, state: DataAgentState) -> DataAgentState:
        prompt = _DATA_PLAN_PROMPT.format(
            question=state.clean_question,
            intent=state.intent.intent if state.intent else "unknown",
            time_period=state.intent.time_period if state.intent else "unspecified",
            key_metrics=state.intent.key_metrics if state.intent else [],
            year_start=state.intent.year_start if state.intent else 2011,
            year_end=state.intent.year_end if state.intent else 2026,
        )

        # On retry: add error context so Gemini can self-correct
        if state.retry_count > 0 and state.plan_errors:
            prompt += _RETRY_ADDENDUM.format(
                errors="\n".join(f"  - {e}" for e in state.plan_errors)
            )

        raw = await self._llm_json(prompt)

        try:
            state.data_plan = LLMDataPlanResponse.model_validate(raw)
        except Exception as e:
            raise ValueError(f"Data plan response failed validation: {e} | raw={raw}")

        logger.debug(
            f"    tables={state.data_plan.tables_needed} "
            f"cols={list(state.data_plan.columns_needed.keys())}"
        )
        return state

    # ──────────────────────────────────────────────────────────
    # NODE 4 — VALIDATE PLAN  (no LLM, auto-retry)
    # ──────────────────────────────────────────────────────────

    async def _node_validate_plan(self, state: DataAgentState) -> DataAgentState:
        state = validate_plan(state)

        if not state.plan_valid:
            if state.retry_count <= state.max_retries:
                logger.warning(
                    f"    plan invalid (attempt {state.retry_count}/{state.max_retries}), "
                    f"retrying: {state.plan_errors}"
                )
                state = await self._node_build_data_plan(state)
                state = validate_plan(state)

            if not state.plan_valid:
                logger.error(
                    f"    plan still invalid after {state.retry_count} retries: "
                    f"{state.plan_errors}"
                )
        else:
            logger.debug("    plan valid ✓")

        return state

    # ──────────────────────────────────────────────────────────
    # NODE 5 — FORMAT OUTPUT
    # ──────────────────────────────────────────────────────────

    async def _node_format_output(self, state: DataAgentState) -> DataAgentState:
        return state  # _build_result reads state directly

    # ──────────────────────────────────────────────────────────
    # GEMINI LLM HELPER  ← the only Gemini-specific method
    # ──────────────────────────────────────────────────────────

    async def _llm_json(self, user_prompt: str) -> dict:
        """
        Calls GeminiAdapter.generate_json() and returns a parsed dict.

        This is the ONLY method that differs from the OpenAI version.
        Everything above this line is identical.
        """
        return await self._gemini.generate_json(
            system_prompt=_SYSTEM,
            user_prompt=user_prompt,
        )

    # ──────────────────────────────────────────────────────────
    # RESULT BUILDER
    # ──────────────────────────────────────────────────────────

    def _build_result(self, state: DataAgentState, duration_ms: float) -> DataAgentResult:
        plan_ok   = state.plan_valid and state.data_plan is not None
        input_ok  = state.input_valid
        no_errors = len(state.errors) == 0
        success   = input_ok and plan_ok and no_errors

        base = DataAgentResult(
            success=success,
            model_used=self.model,
            duration_ms=duration_ms,
            retry_count=state.retry_count,
            node_history=state.node_history,
            errors=state.errors,
        )

        if state.intent:
            base.intent       = state.intent.intent.value
            base.time_period  = state.intent.time_period
            base.year_start   = state.intent.year_start
            base.year_end     = state.intent.year_end
            base.quarter      = state.intent.quarter
            base.granularity  = state.intent.granularity.value
            base.key_metrics  = state.intent.key_metrics

        if state.data_plan:
            base.tables_needed    = state.data_plan.tables_needed
            base.columns_needed   = state.data_plan.columns_needed
            base.filters          = state.data_plan.filters
            base.aggregations     = state.data_plan.aggregations
            base.data_explanation = state.data_plan.explanation

        return base
