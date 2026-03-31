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
    executive_summary: str = ""
    direct_answer: str = ""
    key_findings: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    detailed_insights: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    risk_assessment: list = field(default_factory=list)
    financial_health: dict = field(default_factory=dict)
    summary_sentence: str = ""
    word_count: int = 0
    model_used: str = ""
    duration_ms: float = 0
    retry_count: int = 0
    node_history: list = field(default_factory=list)
    errors: list = field(default_factory=list)


_SYSTEM = """You are a senior financial report writer for Finora AI.
Transform technical agent outputs into comprehensive, detailed reports for business executives.
Rules:
  - Answer the user's question directly in the first paragraph
  - Provide deep analysis with multiple data points and evidence
  - Include detailed findings, evidence, and insights
  - Plain English — no jargon
  - Total under 3000 words
  - Never invent numbers not present in the inputs
Return ONLY valid JSON — no markdown fences, no explanation outside JSON.
"""

_PLAN_PROMPT = """
Before writing, create a comprehensive content plan.

User question: "{question}"
Content available: data_analysis ({da_len}c), patterns ({pat_len}c), forecast ({fc_len}c), insights ({ins_len}c)

Return EXACTLY this JSON:
{{
  "question_type": "trend|forecast|comparison|risk|performance|general",
  "direct_answer_point": "the key fact that answers the question (≤ 30 words)",
  "top_findings": ["Finding 1", "Finding 2", "Finding 3", "Finding 4", "Finding 5"],
  "evidence": ["Evidence 1 with specific data", "Evidence 2 with specific data", "Evidence 3"],
  "top_recommendations": ["Action 1", "Action 2", "Action 3", "Action 4"],
  "key_number": "most important number or null",
  "word_limit": 3000,
  "tone": "professional"
}}
"""

_WRITE_PROMPT = """
Write a comprehensive, detailed financial report covering all aspects.

Plan: {plan}

Source material:
DATA ANALYSIS: {data_analysis}
PATTERNS: {patterns}
FORECAST: {forecast}
INSIGHTS: {insights}

User question: "{question}"

Return EXACTLY this JSON with comprehensive details:
{{
  "executive_summary": "5-6 sentence comprehensive summary covering all key aspects of the financial position",
  "direct_answer": "3-4 sentences directly answering the question with specific numbers and context",
  "key_findings": [
    "Detailed finding 1: specific metric, comparison, and what it means",
    "Detailed finding 2: trend analysis with percentage change",
    "Detailed finding 3: pattern interpretation and business impact",
    "Detailed finding 4: risk assessment with supporting data",
    "Detailed finding 5: opportunity identification with evidence",
    "Detailed finding 6: year-over-year comparison insights",
    "Detailed finding 7: operational efficiency metrics"
  ],
  "evidence": [
    "Evidence from data: specific revenue figure with context",
    "Evidence from patterns: trend direction and magnitude",
    "Evidence from forecast: projected values with confidence",
    "Evidence from insights: health score rationale",
    "Evidence from expenses: category breakdown and analysis",
    "Evidence from profit margin: calculation and interpretation"
  ],
  "detailed_insights": [
    "Strategic insight 1: business implications",
    "Strategic insight 2: market position analysis",
    "Strategic insight 3: growth drivers identification",
    "Strategic insight 4: cost optimization opportunities",
    "Strategic insight 5: risk mitigation recommendations"
  ],
  "recommendations": [
    "Immediate action with specific steps and timeline",
    "Short-term recommendation with expected impact",
    "Medium-term strategic initiative with resource requirements",
    "Long-term recommendation for sustainable growth",
    "Risk mitigation action with implementation plan"
  ],
  "risk_assessment": [
    "High priority risk with probability and impact",
    "Medium priority risk with mitigation strategy",
    "Low priority risk with monitoring recommendation"
  ],
  "financial_health": {{
    "revenue_health": "Excellent/Good/Stable/Concerning with explanation",
    "profit_health": "Excellent/Good/Stable/Concerning with explanation",
    "margin_health": "Excellent/Good/Stable/Concerning with explanation",
    "overall_score": "1-10 score with rationale"
  }},
  "summary_sentence": "2-3 sentence comprehensive summary of overall financial position and outlook."
}}

RULES:
- Total word count across all fields: UNDER 3000
- direct_answer: min 50 chars, include specific numbers
- executive_summary: min 100 chars
- Each finding: ≥ 30 chars with specific numbers and interpretation
- Each evidence: ≥ 25 chars with specific data references
- Each recommendation: ≥ 25 chars with specific actions
- summary_sentence: 2-3 sentences
- Include ALL findings, evidence, insights, and recommendations requested
"""

_RETRY_ADDENDUM = """

PREVIOUS REPORT FAILED THESE CHECKS:
{errors}
Fix all issues. Make the report more comprehensive with more details, evidence, and insights. Keep total words under 3000.
"""


def _llm_json(prompt: str, system: str = _SYSTEM) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3000
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
                 sum(len(f.split()) for f in out.get("key_findings", [])) + \
                 sum(len(r.split()) for r in out.get("recommendations", [])) + \
                 len(out.get("summary_sentence", "").split())
    
    if word_count > 3500:
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
        lines = []
        
        # Executive Summary
        if r.get("executive_summary"):
            lines.append("=" * 50)
            lines.append("EXECUTIVE SUMMARY")
            lines.append("=" * 50)
            lines.append(r["executive_summary"])
            lines.append("")
        
        # Direct Answer
        if r.get("direct_answer"):
            lines.append("=" * 50)
            lines.append("DIRECT ANSWER")
            lines.append("=" * 50)
            lines.append(r["direct_answer"])
            lines.append("")
        
        # Key Findings
        if r.get("key_findings"):
            lines.append("=" * 50)
            lines.append("KEY FINDINGS & ANALYSIS")
            lines.append("=" * 50)
            for i, finding in enumerate(r["key_findings"], 1):
                lines.append(f"{i}. {finding}")
            lines.append("")
        
        # Evidence
        if r.get("evidence"):
            lines.append("=" * 50)
            lines.append("SUPPORTING EVIDENCE")
            lines.append("=" * 50)
            for i, ev in enumerate(r["evidence"], 1):
                lines.append(f"{i}. {ev}")
            lines.append("")
        
        # Detailed Insights
        if r.get("detailed_insights"):
            lines.append("=" * 50)
            lines.append("STRATEGIC INSIGHTS")
            lines.append("=" * 50)
            for i, insight in enumerate(r["detailed_insights"], 1):
                lines.append(f"{i}. {insight}")
            lines.append("")
        
        # Financial Health
        if r.get("financial_health"):
            fh = r["financial_health"]
            lines.append("=" * 50)
            lines.append("FINANCIAL HEALTH SCORE")
            lines.append("=" * 50)
            lines.append(f"Overall Score: {fh.get('overall_score', 'N/A')}/10")
            lines.append(f"Revenue Health: {fh.get('revenue_health', 'N/A')}")
            lines.append(f"Profit Health: {fh.get('profit_health', 'N/A')}")
            lines.append(f"Margin Health: {fh.get('margin_health', 'N/A')}")
            lines.append("")
        
        # Risk Assessment
        if r.get("risk_assessment"):
            lines.append("=" * 50)
            lines.append("RISK ASSESSMENT")
            lines.append("=" * 50)
            for i, risk in enumerate(r["risk_assessment"], 1):
                lines.append(f"{i}. {risk}")
            lines.append("")
        
        # Recommendations
        if r.get("recommendations"):
            lines.append("=" * 50)
            lines.append("RECOMMENDATIONS")
            lines.append("=" * 50)
            for i, rec in enumerate(r["recommendations"], 1):
                lines.append(f"{i}. {rec}")
            lines.append("")
        
        # Summary
        if r.get("summary_sentence"):
            lines.append("=" * 50)
            lines.append("SUMMARY")
            lines.append("=" * 50)
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
            result.executive_summary = r.get("executive_summary", "")
            result.direct_answer = r.get("direct_answer", "")
            result.key_findings = r.get("key_findings", [])
            result.evidence = r.get("evidence", [])
            result.detailed_insights = r.get("detailed_insights", [])
            result.recommendations = r.get("recommendations", [])
            result.risk_assessment = r.get("risk_assessment", [])
            result.financial_health = r.get("financial_health", {})
            result.summary_sentence = r.get("summary_sentence", "")
            
            result.word_count = (
                len(result.executive_summary.split()) +
                len(result.direct_answer.split()) +
                sum(len(f.split()) for f in result.key_findings) +
                sum(len(e.split()) for e in result.evidence) +
                sum(len(i.split()) for i in result.detailed_insights) +
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
