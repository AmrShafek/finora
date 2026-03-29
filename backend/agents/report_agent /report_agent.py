"""
src/agent.py — ReportAgent (5-node DeepAgent)

BEFORE:
    result = report_agent(question, data_analysis, patterns, forecast, insights)
    # str — no word check, no structure

AFTER:
    agent  = ReportAgent()
    result = await agent.run(question, data_analysis, patterns, forecast, insights)
    print(result.text)           # drop-in str
    print(result.word_count)     # guaranteed ≤ 300
    print(result.key_findings)   # list
    print(result.recommendations)

Graph:
  Node 1  validate_inputs  → 5 inputs checked (no LLM)
  Node 2  plan_content     → Gemini plans structure before writing
  Node 3  write_report     → Gemini writes using the plan
  Node 4  validate_report  → word count + quality (no LLM, auto-retry)
  Node 5  format_output    → assemble final text
"""

from __future__ import annotations

import os
import time
from typing import Optional, Annotated

from loguru import logger
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from operator import add

from .dspy_modules import PlanContentModule, WriteReportModule
from .models import ContentPlan, ReportAgentState, ReportInput, ReportOutput, ReportResult
from .validators import validate_inputs, validate_report
from ..deep_agent import make_dspy_lm, FinancialJSONModule


_SYSTEM = """You are a financial report writer for Finora AI.
Transform technical agent outputs into clear, professional reports for business users.
Rules:
  - Answer the user's question directly in the first paragraph
  - Plain English — no jargon
  - Total under 300 words
  - Never invent numbers not present in the inputs
Return ONLY valid JSON — no markdown fences, no explanation outside JSON.
"""

_PLAN_PROMPT = """
Before writing, create a content plan.

User question: "{question}"
Content available: data_analysis ({da_len}c), patterns ({pat_len}c), forecast ({fc_len}c), insights ({ins_len}c)

Return EXACTLY this JSON:
{{
  "question_type": "trend|forecast|comparison|risk|performance|general",
  "direct_answer_point": "the single most important fact that answers the question (≤ 20 words)",
  "top_findings": ["Finding 1 (≤15w)", "Finding 2 (≤15w)", "Finding 3 optional"],
  "top_recommendations": ["Action 1 (≤15w)", "Action 2 (≤15w)"],
  "key_number": "most important number e.g. '$1.5M revenue' or null",
  "word_limit": 300,
  "tone": "professional",
  "include_bullets": true,
  "include_forecast": true,
  "include_risk": true
}}
"""

_WRITE_PROMPT = """
Write the report following this plan.

Plan: {plan}

Source material (use ONLY facts/numbers from these):
DATA ANALYSIS: {data_analysis}
PATTERNS: {patterns}
FORECAST: {forecast}
INSIGHTS: {insights}

User question: "{question}"

Return EXACTLY this JSON:
{{
  "direct_answer": "2-3 sentences directly answering the question. Include {key_number}. Plain English.",
  "key_findings": [
    "Finding with specific number",
    "Finding with specific number",
    "Optional third finding"
  ],
  "recommendations": [
    "Verb-led action — specific (≤25 words)",
    "Second action (≤25 words)"
  ],
  "summary_sentence": "One closing sentence summarising the financial position (≤20 words)."
}}

RULES:
- Total word count across all fields: UNDER 300
- direct_answer: min 20 chars, max 3 sentences
- Each finding: ≥ 5 chars, include a number
- Each recommendation: ≥ 10 chars, starts with verb
- summary_sentence: exactly 1 sentence
- No bullet symbols inside the strings
"""

_RETRY = """
PREVIOUS REPORT FAILED THESE CHECKS:
{errors}
Fix all issues. Keep total words under 300.
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LANGGRAPH STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ReportAgentGraphState(TypedDict):
    """
    LangGraph state for the 5-node ReportAgent graph.
    node_history and errors use Annotated[list, add] for accumulation.
    """
    # Inputs
    question:       str
    data_analysis:  str
    patterns:       str
    forecast:       str
    insights:       str
    max_retries:    int
    # Node 1
    input_valid:    bool
    input_errors:   list[str]
    # Node 2 — ContentPlan as dict
    plan:           Optional[dict]
    # Node 3 / 4 — ReportOutput as dict
    report:         Optional[dict]
    report_valid:   bool
    report_errors:  list[str]
    retry_count:    int
    # Node 5
    formatted_text: str
    # Graph control
    node_history:   Annotated[list[dict], add]
    errors:         Annotated[list[dict], add]
    abort:          bool


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ReportAgent:
    """
    Production ReportAgent — refactored to DeepAgent + DSPy + LangGraph.

    ARCHITECTURE CHANGES:
      GeminiAdapter    → FinancialJSONModule(dspy.LM)   [DSPy]
      _execute_graph() → LangGraph StateGraph           [LangGraph]
      retry inline     → graph cycle via conditional edge

    PUBLIC INTERFACE — unchanged:
      agent  = ReportAgent()
      result = await agent.run(question, data_analysis, patterns, forecast, insights)
    """

    def __init__(
        self,
        model:       str           = "gemini-2.5-flash",
        api_key:     Optional[str] = None,
        max_retries: int           = 2,
        temperature: float         = 0.2,
    ):
        resolved_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not resolved_key:
            raise ValueError("GEMINI_API_KEY not found.")

        self.model       = model
        self.max_retries = max_retries
        self.agent_name  = "ReportAgent"

        # ── DSPy: replaces GeminiAdapter (max_tokens=1500 preserved) ─────────
        self._lm     = make_dspy_lm(
            model=model, api_key=resolved_key,
            temperature=temperature, max_tokens=1500,
        )
        self._module = FinancialJSONModule(lm=self._lm)

        # ── Agent-specific DSPy modules ───────────────────────────────────────
        self._plan_mod  = PlanContentModule(
            json_module=self._module,
            system_prompt=_SYSTEM,
            plan_prompt_template=_PLAN_PROMPT,
        )
        self._write_mod = WriteReportModule(
            json_module=self._module,
            system_prompt=_SYSTEM,
            write_prompt_template=_WRITE_PROMPT,
            retry_template=_RETRY,
        )

        # ── LangGraph: replaces _execute_graph() ─────────────────────────────
        self._graph = self._build_graph()

    async def run(
        self,
        question:      str,
        data_analysis: str,
        patterns:      str,
        forecast:      str,
        insights:      str,
    ) -> ReportResult:
        """Exact same signature as the original ReportAgent.run()."""
        t = time.perf_counter()

        try:
            validated = ReportInput(
                question=question, data_analysis=data_analysis,
                patterns=patterns, forecast=forecast, insights=insights,
            )
        except Exception as e:
            return ReportResult(
                success=False,
                errors=[{"node": "pre_validation", "error": str(e)}],
                model_used=self.model,
                duration_ms=round((time.perf_counter() - t) * 1000, 1),
            )

        initial_state: ReportAgentGraphState = {
            "question":       validated.question,
            "data_analysis":  validated.data_analysis,
            "patterns":       validated.patterns,
            "forecast":       validated.forecast,
            "insights":       validated.insights,
            "max_retries":    self.max_retries,
            "input_valid":    False,
            "input_errors":   [],
            "plan":           None,
            "report":         None,
            "report_valid":   False,
            "report_errors":  [],
            "retry_count":    0,
            "formatted_text": "",
            "node_history":   [],
            "errors":         [],
            "abort":          False,
        }

        logger.info(f"🤖 [{self.agent_name}] | model={self.model}")
        final_state = await self._graph.ainvoke(initial_state)
        duration_ms = round((time.perf_counter() - t) * 1000, 1)
        result      = self._build_result(final_state, duration_ms)

        if result.success:
            logger.success(f"✅ [{self.agent_name}] {duration_ms}ms | words={result.word_count}")
        else:
            logger.error(f"❌ [{self.agent_name}] {duration_ms}ms | {result.errors}")

        return result

    # ── LangGraph Builder ─────────────────────────────────────────────

    def _build_graph(self):
        """
        Compile the 5-node ReportAgent LangGraph StateGraph.

        Topology:
          validate_inputs → plan_content → write_report
            → validate_report
              → (invalid & retries left?) write_report  [retry cycle]
              → format_output → END
        """
        sg = StateGraph(ReportAgentGraphState)

        sg.add_node("validate_inputs", self._node_validate_inputs)
        sg.add_node("plan_content",    self._node_plan_content)
        sg.add_node("write_report",    self._node_write_report)
        sg.add_node("validate_report", self._node_validate_report)
        sg.add_node("format_output",   self._node_format_output)

        sg.set_entry_point("validate_inputs")

        sg.add_conditional_edges("validate_inputs",
            lambda s: END if s.get("abort") else "plan_content")
        sg.add_conditional_edges("plan_content",
            lambda s: END if s.get("abort") else "write_report")
        sg.add_conditional_edges("write_report",
            lambda s: END if s.get("abort") else "validate_report")
        # Retry cycle: validate_report → write_report
        sg.add_conditional_edges(
            "validate_report",
            lambda s: (
                "write_report"
                if not s.get("report_valid") and s.get("retry_count", 0) < s.get("max_retries", 2)
                else "format_output"
            ),
        )
        sg.add_edge("format_output", END)

        return sg.compile()

    # ── Nodes ──────────────────────────────────────────────────

    async def _node_validate_inputs(self, state: ReportAgentGraphState) -> dict:
        t = time.perf_counter()
        try:
            tmp = ReportAgentState(
                question=state["question"], data_analysis=state["data_analysis"],
                patterns=state["patterns"], forecast=state["forecast"],
                insights=state["insights"],
            )
            tmp = validate_inputs(tmp)
            ms  = round((time.perf_counter() - t) * 1000, 1)
            if not tmp.input_valid:
                err_msg = "Input validation: " + " | ".join(tmp.input_errors)
                return {
                    "input_valid":  False, "input_errors": tmp.input_errors,
                    "node_history": [{"node": "validate_inputs", "status": "failed", "ms": ms}],
                    "errors":       [{"node": "validate_inputs", "error": err_msg}],
                    "abort":        True,
                }
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

    async def _node_plan_content(self, state: ReportAgentGraphState) -> dict:
        """
        LLM node: create content plan.
        BEFORE: self._llm_json(_PLAN_PROMPT.format(...))
        AFTER:  self._plan_mod.acall(...)
        """
        t = time.perf_counter()
        try:
            raw  = await self._plan_mod.acall(
                question=state["question"],
                da_len=len(state["data_analysis"]),
                pat_len=len(state["patterns"]),
                fc_len=len(state["forecast"]),
                ins_len=len(state["insights"]),
            )
            plan = ContentPlan.model_validate(raw)
            ms   = round((time.perf_counter() - t) * 1000, 1)
            logger.debug(f"    plan: type={plan.question_type} key={plan.key_number} ({ms}ms)")
            return {
                "plan":         plan.model_dump(),
                "node_history": [{"node": "plan_content", "status": "completed", "ms": ms}],
                "abort":        False,
            }
        except Exception as exc:
            ms  = round((time.perf_counter() - t) * 1000, 1)
            msg = f"{type(exc).__name__}: {exc}"
            logger.error(f"  ❌ [plan_content] {msg}")
            return {
                "node_history": [{"node": "plan_content", "status": "failed", "ms": ms, "error": msg}],
                "errors":       [{"node": "plan_content", "error": msg}],
                "abort":        True,
            }

    async def _node_write_report(self, state: ReportAgentGraphState) -> dict:
        """
        LLM node: write the final concise report.
        On retry: passes report_errors for self-correction.
        BEFORE: self._llm_json(_WRITE_PROMPT.format(...))
        AFTER:  self._write_mod.acall(...)
        """
        t = time.perf_counter()
        try:
            plan_dict    = state.get("plan") or {}
            plan_obj     = ContentPlan.model_validate(plan_dict) if plan_dict else None
            plan_str     = plan_obj.model_dump_json(indent=2) if plan_obj else "{}"
            key_num      = (plan_obj.key_number or "the key metric") if plan_obj else "the key metric"
            retry_errors = state.get("report_errors") if state.get("retry_count", 0) > 0 else None

            raw = await self._write_mod.acall(
                plan=plan_str,
                data_analysis=state["data_analysis"][:3000],
                patterns=state["patterns"][:1500],
                forecast=state["forecast"][:1500],
                insights=state["insights"][:1500],
                question=state["question"],
                key_number=key_num,
                retry_errors=retry_errors,
            )
            report = ReportOutput.model_validate(raw)
            ms     = round((time.perf_counter() - t) * 1000, 1)
            logger.debug(f"    report: words={report.word_count} ({ms}ms)")
            return {
                "report":       report.model_dump(),
                "node_history": [{"node": "write_report", "status": "completed", "ms": ms}],
                "abort":        False,
            }
        except Exception as exc:
            ms  = round((time.perf_counter() - t) * 1000, 1)
            msg = f"{type(exc).__name__}: {exc}"
            logger.error(f"  ❌ [write_report] {msg}")
            return {
                "node_history": [{"node": "write_report", "status": "failed", "ms": ms, "error": msg}],
                "errors":       [{"node": "write_report", "error": msg}],
                "abort":        True,
            }

    async def _node_validate_report(self, state: ReportAgentGraphState) -> dict:
        """
        Pure Python quality + word-count checks.
        Retry expressed as LangGraph conditional edge:
          validate_report → (invalid & retries left) → write_report
        """
        t           = time.perf_counter()
        report_raw  = state.get("report")
        report_obj  = ReportOutput.model_validate(report_raw) if report_raw else None

        tmp = ReportAgentState(
            question=state["question"], data_analysis=state["data_analysis"],
            patterns=state["patterns"], forecast=state["forecast"],
            insights=state["insights"],
            report=report_obj,
            retry_count=state.get("retry_count", 0),
            max_retries=state.get("max_retries", 2),
        )
        tmp = validate_report(tmp)
        ms  = round((time.perf_counter() - t) * 1000, 1)

        if tmp.report_valid:
            logger.debug(f"    report valid ✓ ({ms}ms)")
        else:
            logger.warning(f"    report invalid (attempt {tmp.retry_count}/{tmp.max_retries}): {tmp.report_errors} ({ms}ms)")
        return {
            "report_valid":  tmp.report_valid,
            "report_errors": tmp.report_errors,
            "retry_count":   tmp.retry_count,
            "node_history":  [{"node": "validate_report", "status": "completed", "ms": ms}],
        }

    async def _node_format_output(self, state: ReportAgentGraphState) -> dict:
        """
        No LLM — assemble formatted text from ReportOutput fields.
        Preserves exact original format_output logic.
        """
        t          = time.perf_counter()
        report_raw = state.get("report")
        if not report_raw:
            ms = round((time.perf_counter() - t) * 1000, 1)
            return {
                "formatted_text": "Report generation failed.",
                "node_history":   [{"node": "format_output", "status": "skipped", "ms": ms}],
            }
        r     = ReportOutput.model_validate(report_raw)
        lines = [r.direct_answer, ""]
        if r.key_findings:
            lines.append("Key findings:")
            lines += [f"• {f}" for f in r.key_findings]
            lines.append("")
        if r.recommendations:
            lines.append("Recommended actions:")
            lines += [f"• {rec}" for rec in r.recommendations]
            lines.append("")
        lines.append(r.summary_sentence)
        ms = round((time.perf_counter() - t) * 1000, 1)
        return {
            "formatted_text": "\n".join(lines).strip(),
            "node_history":   [{"node": "format_output", "status": "completed", "ms": ms}],
        }

    def _build_result(self, state: ReportAgentGraphState, duration_ms: float) -> ReportResult:
        """Convert final LangGraph state dict into a typed ReportResult."""
        report_raw = state.get("report")
        report_obj = ReportOutput.model_validate(report_raw) if report_raw else None
        plan_raw   = state.get("plan")
        plan_obj   = ContentPlan.model_validate(plan_raw) if plan_raw else None

        success = (
            state.get("input_valid", False)
            and state.get("report_valid", False)
            and report_obj is not None
            and not state.get("errors", [])
        )
        base = ReportResult(
            success=success,
            text=state.get("formatted_text", ""),
            model_used=self.model,
            duration_ms=duration_ms,
            retry_count=state.get("retry_count", 0),
            node_history=state.get("node_history", []),
            errors=state.get("errors", []),
        )
        if plan_obj:
            base.question_type = plan_obj.question_type
            base.tone          = plan_obj.tone
        if report_obj:
            base.direct_answer    = report_obj.direct_answer
            base.key_findings     = report_obj.key_findings
            base.recommendations  = report_obj.recommendations
            base.summary_sentence = report_obj.summary_sentence
            base.word_count       = report_obj.word_count
        return base
