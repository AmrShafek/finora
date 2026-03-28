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
from typing import Callable, Optional

from loguru import logger

from .gemini_adapter import GeminiAdapter
from .models import ContentPlan, ReportAgentState, ReportInput, ReportOutput, ReportResult
from .validators import validate_inputs, validate_report


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


class ReportAgent:

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
        self._gemini = GeminiAdapter(
            api_key=resolved_key, model_name=model,
            temperature=temperature, max_output_tokens=1500,
        )

    async def run(
        self,
        question:      str,
        data_analysis: str,
        patterns:      str,
        forecast:      str,
        insights:      str,
    ) -> ReportResult:
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

        state = ReportAgentState(
            question=validated.question, data_analysis=validated.data_analysis,
            patterns=validated.patterns, forecast=validated.forecast,
            insights=validated.insights, max_retries=self.max_retries,
        )

        logger.info(f"🤖 [{self.agent_name}] | model={self.model}")
        state       = await self._execute_graph(state)
        duration_ms = round((time.perf_counter() - t) * 1000, 1)
        result      = self._build_result(state, duration_ms)

        if result.success:
            logger.success(f"✅ [{self.agent_name}] {duration_ms}ms | words={result.word_count}")
        else:
            logger.error(f"❌ [{self.agent_name}] {duration_ms}ms | {result.errors}")

        return result

    async def _execute_graph(self, state: ReportAgentState) -> ReportAgentState:
        graph: list[tuple[str, Callable, bool]] = [
            ("validate_inputs", self._node_validate_inputs, True),
            ("plan_content",    self._node_plan_content,    True),
            ("write_report",    self._node_write_report,    True),
            ("validate_report", self._node_validate_report, False),
            ("format_output",   self._node_format_output,   False),
        ]
        for name, fn, critical in graph:
            t0 = time.perf_counter()
            logger.debug(f"  ▶ [{name}]")
            try:
                state = await fn(state)
                ms = round((time.perf_counter() - t0) * 1000, 1)
                state.node_history.append({"node": name, "status": "completed", "ms": ms})
                logger.debug(f"  ✅ [{name}] {ms}ms")
            except Exception as exc:
                ms  = round((time.perf_counter() - t0) * 1000, 1)
                msg = f"{type(exc).__name__}: {exc}"
                state.node_history.append({"node": name, "status": "failed", "ms": ms, "error": msg})
                state.errors.append({"node": name, "error": msg})
                logger.error(f"  ❌ [{name}] {msg}")
                if critical:
                    break
        return state

    # ── Nodes ──────────────────────────────────────────────────

    async def _node_validate_inputs(self, state: ReportAgentState) -> ReportAgentState:
        state = validate_inputs(state)
        if not state.input_valid:
            raise ValueError("Input validation: " + " | ".join(state.input_errors))
        return state

    async def _node_plan_content(self, state: ReportAgentState) -> ReportAgentState:
        raw = await self._llm_json(_PLAN_PROMPT.format(
            question=state.question,
            da_len=len(state.data_analysis), pat_len=len(state.patterns),
            fc_len=len(state.forecast), ins_len=len(state.insights),
        ))
        try:
            state.plan = ContentPlan.model_validate(raw)
        except Exception as e:
            raise ValueError(f"ContentPlan failed: {e}")
        logger.debug(f"    plan: type={state.plan.question_type} key={state.plan.key_number}")
        return state

    async def _node_write_report(self, state: ReportAgentState) -> ReportAgentState:
        plan_str = state.plan.model_dump_json(indent=2) if state.plan else "{}"
        key_num  = (state.plan.key_number or "the key metric") if state.plan else "the key metric"

        prompt = _WRITE_PROMPT.format(
            plan=plan_str,
            data_analysis=state.data_analysis[:3000],
            patterns=state.patterns[:1500],
            forecast=state.forecast[:1500],
            insights=state.insights[:1500],
            question=state.question,
            key_number=key_num,
        )
        if state.retry_count > 0 and state.report_errors:
            prompt += _RETRY.format(errors="\n".join(f"  - {e}" for e in state.report_errors))

        raw = await self._llm_json(prompt)
        try:
            state.report = ReportOutput.model_validate(raw)
        except Exception as e:
            raise ValueError(f"ReportOutput failed: {e}")
        logger.debug(f"    report: words={state.report.word_count}")
        return state

    async def _node_validate_report(self, state: ReportAgentState) -> ReportAgentState:
        state = validate_report(state)
        if not state.report_valid and state.retry_count <= state.max_retries:
            logger.warning(f"    report invalid, retrying: {state.report_errors}")
            state = await self._node_write_report(state)
            state = validate_report(state)
        if not state.report_valid:
            logger.error(f"    still invalid after {state.retry_count} retries")
        return state

    async def _node_format_output(self, state: ReportAgentState) -> ReportAgentState:
        if state.report is None:
            state.formatted_text = "Report generation failed."
            return state
        r = state.report
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
        state.formatted_text = "\n".join(lines).strip()
        return state

    async def _llm_json(self, prompt: str) -> dict:
        return await self._gemini.generate_json(system_prompt=_SYSTEM, user_prompt=prompt)

    def _build_result(self, state: ReportAgentState, duration_ms: float) -> ReportResult:
        success = state.input_valid and state.report_valid and state.report is not None and not state.errors
        base = ReportResult(
            success=success, text=state.formatted_text,
            model_used=self.model, duration_ms=duration_ms,
            retry_count=state.retry_count, node_history=state.node_history, errors=state.errors,
        )
        if state.plan:
            base.question_type = state.plan.question_type
            base.tone          = state.plan.tone
        if state.report:
            base.direct_answer    = state.report.direct_answer
            base.key_findings     = state.report.key_findings
            base.recommendations  = state.report.recommendations
            base.summary_sentence = state.report.summary_sentence
            base.word_count       = state.report.word_count
        return base
