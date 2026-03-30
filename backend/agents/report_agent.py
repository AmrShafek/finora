"""
Report Agent - Financial report generation specialist
New 5-node architecture with validators and OpenRouter
"""

import json
import re
import time
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = "mistralai/mistral-small-24b-instruct-2501"


@dataclass
class ReportAgentState:
    question: str = ""
    data_analysis: str = ""
    patterns: str = ""
    forecast: str = ""
    insights: str = ""
    max_retries: int = 2
    
    input_valid: bool = False
    input_errors: list = field(default_factory=list)
    
    plan: Optional[Dict] = None
    report: Optional[Dict] = None
    report_valid: bool = False
    report_errors: list = field(default_factory=list)
    retry_count: int = 0
    
    formatted_text: str = ""
    node_history: list = field(default_factory=list)
    errors: list = field(default_factory=list)


@dataclass
class ReportAgentResult:
    success: bool = False
    text: str = ""
    direct_answer: str = ""
    key_findings: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    summary_sentence: str = ""
    word_count: int = 0
    model_used: str = ""
    duration_ms: float = 0
    retry_count: int = 0
    node_history: list = field(default_factory=list)
    errors: list = field(default_factory=list)


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
  "top_findings": ["Finding 1", "Finding 2", "Finding 3"],
  "top_recommendations": ["Action 1", "Action 2"],
  "key_number": "most important number or null",
  "word_limit": 300,
  "tone": "professional"
}}
"""

_WRITE_PROMPT = """
Write the report following this plan.

Plan: {plan}

Source material:
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
    "Finding with specific number"
  ],
  "recommendations": [
    "Verb-led action — specific",
    "Second action"
  ],
  "summary_sentence": "One closing sentence summarising the financial position."
}}

RULES:
- Total word count across all fields: UNDER 300
- direct_answer: min 20 chars
- Each finding: ≥ 5 chars, include a number
- Each recommendation: ≥ 10 chars, starts with verb
- summary_sentence: exactly 1 sentence
"""

_RETRY_ADDENDUM = """

PREVIOUS REPORT FAILED THESE CHECKS:
{errors}
Fix all issues. Keep total words under 300.
"""


def _llm_json(prompt: str, system: str = _SYSTEM) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
    )
    raw = response.choices[0].message.content.strip()
    
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    
    return json.loads(raw)


def validate_inputs(state: ReportAgentState) -> ReportAgentState:
    """Node 1: Validate inputs"""
    errors = []
    
    question = (state.question or "").strip()
    if not question or len(question) < 3:
        errors.append("question is too short.")
    elif len(question) > 2000:
        errors.append("question is too long.")
    
    for label, val, min_len in [
        ("data_analysis", state.data_analysis, 10),
        ("patterns", state.patterns, 5),
        ("forecast", state.forecast, 5),
        ("insights", state.insights, 5),
    ]:
        v = (val or "").strip()
        if not v:
            errors.append(f"{label} is empty.")
        elif len(v) < min_len:
            errors.append(f"{label} is too short.")
    
    if errors:
        state.input_valid = False
        state.input_errors = errors
    else:
        state.input_valid = True
        state.input_errors = []
    
    return state


def validate_report(state: ReportAgentState) -> ReportAgentState:
    """Node 4: Validate report output"""
    errors = []
    out = state.report
    
    if out is None:
        state.report_valid = False
        state.report_errors = ["No report produced."]
        state.retry_count += 1
        return state
    
    if len(out.get("direct_answer", "").strip()) < 20:
        errors.append("direct_answer is too short.")
    
    if not out.get("key_findings"):
        errors.append("key_findings is empty.")
    
    if not out.get("recommendations"):
        errors.append("recommendations is empty.")
    
    if len(out.get("summary_sentence", "").strip()) < 10:
        errors.append("summary_sentence is too short.")
    
    word_count = len(out.get("direct_answer", "").split()) + \
                 len(out.get("key_findings", [])) + \
                 len(out.get("recommendations", [])) + \
                 len(out.get("summary_sentence", "").split())
    
    if word_count > 350:
        errors.append(f"Report is too long ({word_count} words).")
    
    if errors:
        state.report_valid = False
        state.report_errors = errors
        state.retry_count += 1
    else:
        state.report_valid = True
        state.report_errors = []
    
    return state


class ReportAgent:
    """Report Agent - Financial report generation specialist"""
    
    def __init__(self, max_retries: int = 2, temperature: float = 0.2):
        self.max_retries = max_retries
        self.temperature = temperature
        self.agent_name = "ReportAgent"
    
    async def run(
        self,
        question: str,
        data_analysis: str,
        patterns: str,
        forecast: str,
        insights: str,
    ) -> ReportAgentResult:
        """Execute the 5-node report agent graph"""
        t_start = time.perf_counter()
        
        state = ReportAgentState(
            question=question,
            data_analysis=data_analysis,
            patterns=patterns,
            forecast=forecast,
            insights=insights,
            max_retries=self.max_retries,
        )
        
        nodes = [
            ("validate_inputs", self._node_validate_inputs),
            ("plan_content", self._node_plan_content),
            ("write_report", self._node_write_report),
            ("validate_report", self._node_validate_report),
            ("format_output", self._node_format_output),
        ]
        
        for node_name, node_fn in nodes:
            t = time.perf_counter()
            try:
                state = await node_fn(state)
                ms = round((time.perf_counter() - t) * 1000, 1)
                state.node_history.append({"node": node_name, "status": "completed", "ms": ms})
            except Exception as e:
                ms = round((time.perf_counter() - t) * 1000, 1)
                state.node_history.append({"node": node_name, "status": "failed", "ms": ms, "error": str(e)})
                state.errors.append({"node": node_name, "error": str(e)})
                break
        
        duration_ms = round((time.perf_counter() - t_start) * 1000, 1)
        
        return self._build_result(state, duration_ms)
    
    async def _node_validate_inputs(self, state: ReportAgentState) -> ReportAgentState:
        state = validate_inputs(state)
        if not state.input_valid:
            raise ValueError("Input validation failed: " + " | ".join(state.input_errors))
        return state
    
    async def _node_plan_content(self, state: ReportAgentState) -> ReportAgentState:
        prompt = _PLAN_PROMPT.format(
            question=state.question,
            da_len=len(state.data_analysis),
            pat_len=len(state.patterns),
            fc_len=len(state.forecast),
            ins_len=len(state.insights),
        )
        state.plan = _llm_json(prompt)
        return state
    
    async def _node_write_report(self, state: ReportAgentState) -> ReportAgentState:
        plan_str = json.dumps(state.plan) if state.plan else "{}"
        key_num = (state.plan.get("key_number") or "the key metric") if state.plan else "the key metric"
        
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
            prompt += _RETRY_ADDENDUM.format(errors="\n".join(f"  - {e}" for e in state.report_errors))
        
        state.report = _llm_json(prompt)
        return state
    
    async def _node_validate_report(self, state: ReportAgentState) -> ReportAgentState:
        state = validate_report(state)
        
        if not state.report_valid and state.retry_count <= self.max_retries:
            state = await self._node_write_report(state)
            state = validate_report(state)
        
        return state
    
    async def _node_format_output(self, state: ReportAgentState) -> ReportAgentState:
        if not state.report:
            state.formatted_text = "Report generation failed."
            return state
        
        r = state.report
        lines = [r.get("direct_answer", ""), ""]
        
        if r.get("key_findings"):
            lines.append("Key findings:")
            lines += [f"- {f}" for f in r["key_findings"]]
            lines.append("")
        
        if r.get("recommendations"):
            lines.append("Recommended actions:")
            lines += [f"- {rec}" for f in r["recommendations"]]
            lines.append("")
        
        if r.get("summary_sentence"):
            lines.append(r["summary_sentence"])
        
        state.formatted_text = "\n".join(lines).strip()
        return state
    
    def _build_result(self, state: ReportAgentState, duration_ms: float) -> ReportAgentResult:
        success = state.input_valid and state.report_valid and len(state.errors) == 0
        
        result = ReportAgentResult(
            success=success,
            text=state.formatted_text,
            model_used=MODEL,
            duration_ms=duration_ms,
            retry_count=state.retry_count,
            node_history=state.node_history,
            errors=state.errors,
        )
        
        if state.report:
            r = state.report
            result.direct_answer = r.get("direct_answer", "")
            result.key_findings = r.get("key_findings", [])
            result.recommendations = r.get("recommendations", [])
            result.summary_sentence = r.get("summary_sentence", "")
            
            result.word_count = (
                len(result.direct_answer.split()) +
                sum(len(f.split()) for f in result.key_findings) +
                sum(len(r.split()) for r in result.recommendations) +
                len(result.summary_sentence.split())
            )
        
        return result


def report_agent(
    question: str,
    data_analysis: str,
    patterns: str,
    forecast: str,
    insights: str,
) -> str:
    """Synchronous wrapper for ReportAgent"""
    import asyncio
    
    async def run():
        agent = ReportAgent()
        result = await agent.run(question, data_analysis, patterns, forecast, insights)
        return result
    
    result = asyncio.run(run())
    
    return json.dumps({
        "success": result.success,
        "text": result.text,
        "direct_answer": result.direct_answer,
        "key_findings": result.key_findings,
        "recommendations": result.recommendations,
        "summary_sentence": result.summary_sentence,
        "word_count": result.word_count,
    })
