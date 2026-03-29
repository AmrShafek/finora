from __future__ import annotations

import os
import time
from typing import Optional, Annotated

from loguru import logger
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from operator import add

from .dspy_modules import ClassifyIntentModule, BuildDataPlanModule
from .models import (
    AgentInput,
    DataAgentResult,
    DataAgentState,
    LLMDataPlanResponse,
    LLMIntentResponse,
)
from .validators import validate_input, validate_plan
from ..deep_agent import make_dspy_lm, FinancialJSONModule


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROMPTS  (unchanged from original)
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
# LANGGRAPH STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DataAgentGraphState(TypedDict):
    """
    LangGraph state flowing through all DataAgent graph nodes.

    Fields annotated with Annotated[list, add] are ACCUMULATED across nodes
    (LangGraph appends returned lists to the existing list).
    All other fields are REPLACED by each node's returned value.

    This TypedDict mirrors DataAgentState (Pydantic) but uses the LangGraph
    reducer pattern instead of mutable Pydantic model updates.
    """

    # ── Inputs ─────────────────────────────────────────────────
    question:        str
    db_summary:      str
    max_retries:     int

    # ── Node 1: validate_input ──────────────────────────────────
    clean_question:  str
    input_valid:     bool
    input_errors:    list[str]

    # ── Node 2: classify_intent ─────────────────────────────────
    # Stored as dict (LLMIntentResponse.model_dump()) for JSON-serializability
    intent:          Optional[dict]

    # ── Node 3: build_data_plan ─────────────────────────────────
    # Stored as dict (LLMDataPlanResponse.model_dump()) for JSON-serializability
    data_plan:       Optional[dict]

    # ── Node 4: validate_plan ───────────────────────────────────
    plan_valid:      bool
    plan_errors:     list[str]
    retry_count:     int

    # ── Graph control ───────────────────────────────────────────
    # Annotated[list, add] = each node appends its entry; LangGraph accumulates
    node_history:    Annotated[list[dict], add]
    errors:          Annotated[list[dict], add]
    abort:           bool   # set True by critical nodes on failure → routes to END


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DataAgent:
    """
    Production DataAgent — refactored to DeepAgent + DSPy + LangGraph.

    ARCHITECTURE CHANGES vs. original:
      GeminiAdapter    → FinancialJSONModule(dspy.LM)   [DSPy — provider-agnostic]
      _execute_graph() → LangGraph StateGraph           [LangGraph — proper DAG]
      manual loop      → compiled graph with ainvoke()  [LangGraph — async execution]
      retry inline     → graph cycle via conditional edge [LangGraph — explicit retry DAG]

    PUBLIC INTERFACE — unchanged (drop-in replacement):
      agent  = DataAgent(model="gemini-2.5-flash")
      result = await agent.run(question, db_summary)
      # result.tables_needed, result.filters, result.intent, result.node_history ...
    """

    def __init__(
        self,
        model:       str           = "gemini-2.5-flash",
        api_key:     Optional[str] = None,
        max_retries: int           = 2,
        temperature: float         = 0.0,
    ):
        """
        Args:
            model:       Gemini model name. Default: "gemini-2.5-flash".
            api_key:     GEMINI_API_KEY. Falls back to env var if not provided.
            max_retries: Max retry attempts for a failed data plan (default 2).
            temperature: LLM temperature. 0.0 = deterministic (recommended).

        DSPy replaces GeminiAdapter:
            BEFORE: self._gemini = GeminiAdapter(api_key, model_name, temperature)
            AFTER:  self._lm     = make_dspy_lm(model, api_key, temperature)
                    self._module = FinancialJSONModule(lm=self._lm)
        """
        resolved_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Set it in .env or pass api_key= to DataAgent()."
            )

        self.model       = model
        self.max_retries = max_retries
        self.agent_name  = "DataAgent"

        # ── DSPy: replaces GeminiAdapter ─────────────────────────────────────
        # make_dspy_lm() creates a dspy.LM backed by LiteLLM → Gemini REST API.
        # FinancialJSONModule wraps it with the same interface as GeminiAdapter:
        #   raw = await self._module.acall(system_prompt, user_prompt) → dict
        self._lm     = make_dspy_lm(
            model=model, api_key=resolved_key, temperature=temperature
        )
        self._module = FinancialJSONModule(lm=self._lm)

        # ── Agent-specific DSPy modules (one per LLM node) ───────────────────
        # These wrap _module with the exact prompt templates for each node,
        # making each LLM call explicitly typed and independently optimizable.
        self._classify_intent_mod = ClassifyIntentModule(
            json_module=self._module,
            system_prompt=_SYSTEM,
            intent_prompt_template=_INTENT_PROMPT,
        )
        self._build_data_plan_mod = BuildDataPlanModule(
            json_module=self._module,
            system_prompt=_SYSTEM,
            plan_prompt_template=_DATA_PLAN_PROMPT,
            retry_addendum_template=_RETRY_ADDENDUM,
        )

        # ── LangGraph: replaces _execute_graph() ─────────────────────────────
        # The compiled graph is re-used for every run() call.
        self._graph = self._build_graph()

    # ──────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ──────────────────────────────────────────────────────────

    async def run(self, question: str, db_summary: str) -> DataAgentResult:
        """
        Process a financial question through the full 5-node LangGraph.

        Args:
            question:   Natural language financial question.
            db_summary: Current state of the database (tables, row counts, ranges).

        Returns:
            DataAgentResult — typed, validated, with full audit trail.
        """
        t_start = time.perf_counter()

        # Pre-validate inputs with Pydantic before entering the graph
        try:
            validated = AgentInput(question=question, db_summary=db_summary)
        except Exception as e:
            return DataAgentResult(
                success=False,
                errors=[{"node": "pre_validation", "error": str(e)}],
                model_used=self.model,
                duration_ms=round((time.perf_counter() - t_start) * 1000, 1),
            )

        # Build the initial LangGraph state dict
        initial_state: DataAgentGraphState = {
            "question":       validated.question,
            "db_summary":     validated.db_summary,
            "max_retries":    self.max_retries,
            "clean_question": "",
            "input_valid":    False,
            "input_errors":   [],
            "intent":         None,
            "data_plan":      None,
            "plan_valid":     False,
            "plan_errors":    [],
            "retry_count":    0,
            "node_history":   [],
            "errors":         [],
            "abort":          False,
        }

        logger.info(f"🤖 [{self.agent_name}] Starting | model={self.model}")
        logger.info(f"   Q: {question[:80]}{'...' if len(question) > 80 else ''}")

        # ainvoke() runs the compiled LangGraph asynchronously
        final_state   = await self._graph.ainvoke(initial_state)
        duration_ms   = round((time.perf_counter() - t_start) * 1000, 1)
        result        = self._build_result(final_state, duration_ms)

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
    # LANGGRAPH GRAPH BUILDER
    # ──────────────────────────────────────────────────────────

    def _build_graph(self):
        """
        Compile the 5-node DataAgent LangGraph StateGraph.

        Graph topology:
          validate_input
            → (abort?) END
            → classify_intent
              → (abort?) END
              → build_data_plan
                → (abort?) END
                → validate_plan
                  → (invalid & retries left?) build_data_plan  [retry cycle]
                  → format_output → END

        Retry cycle: validate_plan → build_data_plan is a LangGraph CYCLE.
        It fires when plan_valid=False and retry_count < max_retries, allowing
        the LLM to self-correct using the validation error messages as context.
        """
        sg = StateGraph(DataAgentGraphState)

        # Register nodes (bound methods — LangGraph calls node(state) → dict)
        sg.add_node("validate_input",  self._node_validate_input)
        sg.add_node("classify_intent", self._node_classify_intent)
        sg.add_node("build_data_plan", self._node_build_data_plan)
        sg.add_node("validate_plan",   self._node_validate_plan)
        sg.add_node("format_output",   self._node_format_output)

        sg.set_entry_point("validate_input")

        # Critical nodes abort the graph on failure (abort=True → END)
        sg.add_conditional_edges(
            "validate_input",
            lambda s: END if s.get("abort") else "classify_intent",
        )
        sg.add_conditional_edges(
            "classify_intent",
            lambda s: END if s.get("abort") else "build_data_plan",
        )
        sg.add_conditional_edges(
            "build_data_plan",
            lambda s: END if s.get("abort") else "validate_plan",
        )

        # validate_plan → retry cycle or proceed to format_output
        sg.add_conditional_edges(
            "validate_plan",
            lambda s: (
                "build_data_plan"
                if not s.get("plan_valid") and s.get("retry_count", 0) < s.get("max_retries", 2)
                else "format_output"
            ),
        )

        sg.add_edge("format_output", END)

        return sg.compile()

    # ──────────────────────────────────────────────────────────
    # NODE 1 — VALIDATE INPUT  (no LLM)
    # ──────────────────────────────────────────────────────────

    async def _node_validate_input(self, state: DataAgentGraphState) -> dict:
        """
        Pure Python validation: checks question length, injection patterns,
        financial relevance, and db_summary presence.
        Uses the existing validate_input() validator unchanged.
        """
        t = time.perf_counter()
        try:
            # Bridge TypedDict → Pydantic state → call existing validator
            tmp = DataAgentState(
                question=state["question"],
                db_summary=state["db_summary"],
            )
            tmp = validate_input(tmp)
            ms  = round((time.perf_counter() - t) * 1000, 1)

            if not tmp.input_valid:
                err_msg = "Input validation failed: " + " | ".join(tmp.input_errors)
                logger.error(f"  ❌ [validate_input] {err_msg}")
                return {
                    "input_valid":    False,
                    "input_errors":   tmp.input_errors,
                    "clean_question": "",
                    "node_history":   [{"node": "validate_input", "status": "failed",    "ms": ms}],
                    "errors":         [{"node": "validate_input", "error": err_msg}],
                    "abort":          True,   # critical node
                }

            logger.debug(f"    input valid: {tmp.clean_question[:60]}... ({ms}ms)")
            return {
                "input_valid":    True,
                "input_errors":   [],
                "clean_question": tmp.clean_question,
                "node_history":   [{"node": "validate_input", "status": "completed", "ms": ms}],
                "abort":          False,
            }

        except Exception as exc:
            ms  = round((time.perf_counter() - t) * 1000, 1)
            msg = f"{type(exc).__name__}: {exc}"
            logger.error(f"  ❌ [validate_input] {msg}")
            return {
                "input_valid":  False,
                "node_history": [{"node": "validate_input", "status": "failed", "ms": ms, "error": msg}],
                "errors":       [{"node": "validate_input", "error": msg}],
                "abort":        True,
            }

    # ──────────────────────────────────────────────────────────
    # NODE 2 — CLASSIFY INTENT  (DSPy LM call)
    # ──────────────────────────────────────────────────────────

    async def _node_classify_intent(self, state: DataAgentGraphState) -> dict:
        """
        LLM node: classifies the financial question intent via DSPy.

        BEFORE: self._llm_json(prompt) → called GeminiAdapter directly.
        AFTER:  self._classify_intent_mod.acall(...) → calls FinancialJSONModule
                which calls dspy.LM (LiteLLM → Gemini).
        """
        t = time.perf_counter()
        try:
            raw = await self._classify_intent_mod.acall(
                question=state["clean_question"],
                db_summary=state["db_summary"],
            )
            intent_obj = LLMIntentResponse.model_validate(raw)
            ms         = round((time.perf_counter() - t) * 1000, 1)

            logger.debug(
                f"    intent={intent_obj.intent} "
                f"conf={intent_obj.confidence} "
                f"period={intent_obj.time_period} ({ms}ms)"
            )
            return {
                "intent":       intent_obj.model_dump(),
                "node_history": [{"node": "classify_intent", "status": "completed", "ms": ms}],
                "abort":        False,
            }

        except Exception as exc:
            ms  = round((time.perf_counter() - t) * 1000, 1)
            msg = f"{type(exc).__name__}: {exc}"
            logger.error(f"  ❌ [classify_intent] {msg}")
            return {
                "node_history": [{"node": "classify_intent", "status": "failed", "ms": ms, "error": msg}],
                "errors":       [{"node": "classify_intent", "error": msg}],
                "abort":        True,   # critical node
            }

    # ──────────────────────────────────────────────────────────
    # NODE 3 — BUILD DATA PLAN  (DSPy LM call)
    # ──────────────────────────────────────────────────────────

    async def _node_build_data_plan(self, state: DataAgentGraphState) -> dict:
        """
        LLM node: builds a SQL data retrieval plan from the classified intent.

        On retry (retry_count > 0): passes plan_errors as context so the LLM
        can self-correct — same behaviour as the original inline retry logic.
        """
        t = time.perf_counter()
        try:
            intent_dict  = state.get("intent") or {}
            retry_errors = state.get("plan_errors") if state.get("retry_count", 0) > 0 else None

            raw = await self._build_data_plan_mod.acall(
                question=state["clean_question"],
                intent=intent_dict.get("intent", "unknown"),
                time_period=intent_dict.get("time_period", "unspecified"),
                key_metrics=intent_dict.get("key_metrics", []),
                year_start=intent_dict.get("year_start", 2011),
                year_end=intent_dict.get("year_end", 2026),
                retry_errors=retry_errors,
            )
            plan_obj = LLMDataPlanResponse.model_validate(raw)
            ms       = round((time.perf_counter() - t) * 1000, 1)

            logger.debug(
                f"    tables={plan_obj.tables_needed} "
                f"cols={list(plan_obj.columns_needed.keys())} ({ms}ms)"
            )
            return {
                "data_plan":    plan_obj.model_dump(),
                "node_history": [{"node": "build_data_plan", "status": "completed", "ms": ms}],
                "abort":        False,
            }

        except Exception as exc:
            ms  = round((time.perf_counter() - t) * 1000, 1)
            msg = f"{type(exc).__name__}: {exc}"
            logger.error(f"  ❌ [build_data_plan] {msg}")
            return {
                "node_history": [{"node": "build_data_plan", "status": "failed", "ms": ms, "error": msg}],
                "errors":       [{"node": "build_data_plan", "error": msg}],
                "abort":        True,   # critical node
            }

    # ──────────────────────────────────────────────────────────
    # NODE 4 — VALIDATE PLAN  (no LLM, retry cycle handled by graph edges)
    # ──────────────────────────────────────────────────────────

    async def _node_validate_plan(self, state: DataAgentGraphState) -> dict:
        """
        Pure Python validation: 6 sanity checks on the data plan.

        The retry logic that was previously INLINE (calling _node_build_data_plan
        directly) is now expressed as a LangGraph CONDITIONAL EDGE:
          validate_plan → (invalid & retries left) → build_data_plan [cycle]
        This node only increments retry_count and sets plan_valid / plan_errors.
        The graph's conditional edge decides whether to loop or proceed.
        """
        t = time.perf_counter()

        # Bridge TypedDict → Pydantic for the existing validator
        intent_dict   = state.get("intent") or {}
        data_plan_raw = state.get("data_plan")

        intent_obj    = LLMIntentResponse.model_validate(intent_dict)   if intent_dict   else None
        plan_obj      = LLMDataPlanResponse.model_validate(data_plan_raw) if data_plan_raw else None

        tmp = DataAgentState(
            question=state["question"],
            db_summary=state["db_summary"],
            intent=intent_obj,
            data_plan=plan_obj,
            retry_count=state.get("retry_count", 0),
            max_retries=state.get("max_retries", 2),
        )
        tmp = validate_plan(tmp)   # mutates tmp.plan_valid, tmp.plan_errors, tmp.retry_count
        ms  = round((time.perf_counter() - t) * 1000, 1)

        if tmp.plan_valid:
            logger.debug(f"    plan valid ✓ ({ms}ms)")
        else:
            logger.warning(
                f"    plan invalid (attempt {tmp.retry_count}/{tmp.max_retries}): "
                f"{tmp.plan_errors} ({ms}ms)"
            )

        return {
            "plan_valid":   tmp.plan_valid,
            "plan_errors":  tmp.plan_errors,
            "retry_count":  tmp.retry_count,   # validator incremented this on failure
            "node_history": [{"node": "validate_plan", "status": "completed", "ms": ms}],
        }

    # ──────────────────────────────────────────────────────────
    # NODE 5 — FORMAT OUTPUT  (no LLM)
    # ──────────────────────────────────────────────────────────

    async def _node_format_output(self, state: DataAgentGraphState) -> dict:
        """
        No-op node — _build_result() reads the final state directly.
        Exists to make the graph topology explicit (same as original).
        """
        t  = time.perf_counter()
        ms = round((time.perf_counter() - t) * 1000, 1)
        return {
            "node_history": [{"node": "format_output", "status": "completed", "ms": ms}],
        }

    # ──────────────────────────────────────────────────────────
    # RESULT BUILDER
    # ──────────────────────────────────────────────────────────

    def _build_result(
        self,
        state: DataAgentGraphState,
        duration_ms: float,
    ) -> DataAgentResult:
        """Convert the final LangGraph state dict into a typed DataAgentResult."""
        plan_ok   = state.get("plan_valid", False) and state.get("data_plan") is not None
        input_ok  = state.get("input_valid", False)
        no_errors = len(state.get("errors", [])) == 0
        success   = input_ok and plan_ok and no_errors

        base = DataAgentResult(
            success=success,
            model_used=self.model,
            duration_ms=duration_ms,
            retry_count=state.get("retry_count", 0),
            node_history=state.get("node_history", []),
            errors=state.get("errors", []),
        )

        intent_dict = state.get("intent")
        if intent_dict:
            intent_obj       = LLMIntentResponse.model_validate(intent_dict)
            base.intent      = intent_obj.intent.value
            base.time_period = intent_obj.time_period
            base.year_start  = intent_obj.year_start
            base.year_end    = intent_obj.year_end
            base.quarter     = intent_obj.quarter
            base.granularity = intent_obj.granularity.value
            base.key_metrics = intent_obj.key_metrics

        plan_dict = state.get("data_plan")
        if plan_dict:
            plan_obj              = LLMDataPlanResponse.model_validate(plan_dict)
            base.tables_needed    = plan_obj.tables_needed
            base.columns_needed   = plan_obj.columns_needed
            base.filters          = plan_obj.filters
            base.aggregations     = plan_obj.aggregations
            base.data_explanation = plan_obj.explanation

        return base
